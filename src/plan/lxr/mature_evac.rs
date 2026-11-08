use std::sync::Mutex;
use std::{cell::UnsafeCell, marker::PhantomData};

use crate::plan::concurrent::Pause;
use crate::plan::global::Plan;
use crate::plan::lxr::gc_work::mature_evac::SelectDefragBlocks;
use crate::plan::lxr::gc_work::mature_evac::SELECT_DEFRAG_BLOCK_JOB_COUNTER;
use crate::policy::immix::block::{Block, BlockState};
use crate::policy::immix::line::Line;
use crate::policy::immix::ImmixSpace;
use crate::policy::space::Space;
use crate::scheduler::WorkBucketStage;
use crate::scheduler::GCWorker;
use crate::util::linear_scan::{Region, UnstraddlableRegion};
use crate::util::metadata::side_metadata::spec_defs::{IX_LINE_REUSE_COUNT, LOS_PAGE_REUSE_COUNT};
use crate::util::rc;
use crate::util::ObjectReference;
use crate::util::VMThread;
use crate::{
    plan::lxr::LXR,
    scheduler::GCWork,
    vm::{slot::Slot, VMBinding},
};

use super::gc_work::mature_evac::EvacuateMatureObjects;
use super::gc_work::tracing::LXRStopTheWorldProcessEdges;
use crate::util::constants::LOG_BYTES_IN_PAGE;
use atomic::Ordering;
use crossbeam::queue::SegQueue;
use std::sync::atomic::AtomicUsize;

#[repr(C)]
pub struct RemSetEntry<VM: VMBinding>(VM::VMSlot, u8);

impl<VM: VMBinding> RemSetEntry<VM> {
    // Only the Immix and large object spaces reuse memory, so only their slots carry a reuse count; a slot in any other space (such as the VM space) is always the slot it was.
    fn encode(slot: VM::VMSlot, ix: bool, los: bool) -> Self {
        let reuse = if ix {
            IX_LINE_REUSE_COUNT.load_atomic::<u8>(slot.to_address(), Ordering::SeqCst)
        } else if los {
            LOS_PAGE_REUSE_COUNT.load_atomic::<u8>(slot.to_address(), Ordering::SeqCst)
        } else {
            0
        };
        Self(slot, reuse)
    }

    pub fn decode(&self) -> (VM::VMSlot, u8) {
        (self.0, self.1)
    }
}

pub struct MatureEvecRemSet<VM: VMBinding> {
    pub gc_buffers: Vec<UnsafeCell<Vec<RemSetEntry<VM>>>>,
    pub global_packets: Mutex<Vec<Box<dyn GCWork<VM>>>>,
    local_packets: Vec<UnsafeCell<Vec<Box<dyn GCWork<VM>>>>>,
    _p: PhantomData<VM>,
    size: AtomicUsize,
}

unsafe impl<VM: VMBinding> Send for MatureEvecRemSet<VM> {}
unsafe impl<VM: VMBinding> Sync for MatureEvecRemSet<VM> {}

impl<VM: VMBinding> MatureEvecRemSet<VM> {
    pub fn new(workers: usize) -> Self {
        let mut rs = Self {
            gc_buffers: vec![],
            global_packets: Mutex::new(vec![]),
            local_packets: vec![],
            _p: PhantomData,
            size: AtomicUsize::new(0),
        };
        rs.gc_buffers
            .resize_with(workers, || UnsafeCell::new(vec![]));
        rs.local_packets
            .resize_with(workers, || UnsafeCell::new(vec![]));
        rs
    }

    #[allow(clippy::mut_from_ref)]
    fn gc_buffer(&self, id: usize) -> &mut Vec<RemSetEntry<VM>> {
        unsafe { &mut *self.gc_buffers[id].get() }
    }

    pub fn flush_all(&self) {
        let mut mature_evac_remsets = self.global_packets.lock().unwrap();
        self.size.store(0, Ordering::SeqCst);
        for id in 0..self.gc_buffers.len() {
            if !self.gc_buffer(id).is_empty() {
                let remset = std::mem::take(self.gc_buffer(id));
                mature_evac_remsets.push(Box::new(EvacuateMatureObjects::new(remset)));
            }
        }
        for id in 0..self.local_packets.len() {
            let buf = unsafe { &mut *self.local_packets[id].get() };
            if !buf.is_empty() {
                let packets = std::mem::take(buf);
                for p in packets {
                    mature_evac_remsets.push(p);
                }
            }
        }
    }

    pub fn take_global_packets(&self) -> Vec<Box<dyn GCWork<VM>>> {
        let mut mature_evac_remsets = self.global_packets.lock().unwrap();
        std::mem::take(&mut *mature_evac_remsets)
    }

    #[cold]
    fn flush(&self, id: usize) {
        if !self.gc_buffer(id).is_empty() {
            let remset = std::mem::take(self.gc_buffer(id));
            self.size.fetch_add(remset.len(), Ordering::SeqCst);
            let w = EvacuateMatureObjects::new(remset);
            let packet_buffer = unsafe { &mut *self.local_packets[id].get() };
            packet_buffer.push(Box::new(w));
        }
    }

    pub fn record(&self, s: VM::VMSlot, _o: ObjectReference, lxr: &LXR<VM>) {
        let id = crate::scheduler::current_worker_ordinal().unwrap();
        let ix = lxr.immix_space.address_in_space(s.to_address());
        let los = !ix && lxr.los().address_in_space(s.to_address());
        self.gc_buffer(id).push(RemSetEntry::<VM>::encode(s, ix, los));
        if self.gc_buffer(id).len() >= EvacuateMatureObjects::<VM>::CAPACITY {
            self.flush(id)
        }
    }
}

#[derive(Default)]
pub struct MatureEvacuationSet {
    pub fragmented_blocks: SegQueue<Vec<(Block, usize)>>,
    pub fragmented_blocks_size: AtomicUsize,
    pub blocks_in_fragmented_chunks: SegQueue<Vec<(Block, usize)>>,
    pub blocks_in_fragmented_chunks_size: AtomicUsize,
    pub defrag_blocks: Mutex<Vec<Block>>,
    pub num_defrag_blocks: AtomicUsize,
    /// Blocks holding objects that a full pause must not move, left out when the pause selects its set.
    pinned_blocks: SegQueue<Block>,
}

impl MatureEvacuationSet {
    /// Release all the mature defrag source blocks
    pub fn sweep_mature_evac_candidates<VM: VMBinding>(&self, space: &ImmixSpace<VM>) {
        let mut defrag_blocks: Vec<Block> =
            std::mem::take(&mut *self.defrag_blocks.lock().unwrap());
        if defrag_blocks.is_empty() {
            return;
        }
        while let Some(block) = defrag_blocks.pop() {
            if !block.is_defrag_source() || block.get_state() == BlockState::Unallocated {
                // This block has been eagerly released (probably be reused again). Skip it.
                continue;
            }
            block.clear_rc_table();
            block.clear_striddle_table();
            block.rc_sweep_mature::<VM>(space, true);
            assert!(!block.is_defrag_source());
        }
    }

    pub fn schedule_defrag_selection_packets<VM: VMBinding>(&self, space: &ImmixSpace<VM>) {
        let tasks =
            space
                .chunk_map
                .generate_tasks_batched(space.scheduler().num_workers(), |chunks| {
                    Box::new(SelectDefragBlocks {
                        chunks,
                        defrag_threshold: 1,
                    })
                });
        self.fragmented_blocks_size.store(0, Ordering::SeqCst);
        SELECT_DEFRAG_BLOCK_JOB_COUNTER.store(tasks.len(), Ordering::SeqCst);
        space.scheduler().work_buckets[WorkBucketStage::Unconstrained].bulk_add(tasks);
    }

    pub fn skip_block(b: Block) -> bool {
        let s = b.get_state();
        b.is_defrag_source() || s == BlockState::Unallocated || s == BlockState::Nursery
    }

    /// Keeps objects that must not move, such as those native code holds by address, where they are. A full pause has yet to select its set and leaves their blocks out of it. A final-mark pause selected its set when the mark began, so it takes their blocks out of it again, and traces the slots of those blocks' marked objects that point into the set: the mark recorded none of them, since the objects of a block in the set are copied and scanned again.
    pub fn keep_in_place<VM: VMBinding>(&self, objects: &[ObjectReference], lxr: &LXR<VM>, worker: &mut GCWorker<VM>) {
        let pause = lxr.current_pause().unwrap();
        for &object in objects {
            if !lxr.immix_space.in_space(object) {
                continue;
            }
            let block = Block::containing(object);
            if pause == Pause::Full {
                self.pinned_blocks.push(block);
            } else if pause == Pause::FinalMark && block.attempt_unset_defrag_source() {
                let slots = Self::marked_slots_into_set(block, lxr);
                if !slots.is_empty() {
                    worker.add_work(WorkBucketStage::Closure, LXRStopTheWorldProcessEdges::<VM, false>::new_remset(slots, worker.mmtk));
                }
            }
        }
    }

    fn marked_slots_into_set<VM: VMBinding>(block: Block, lxr: &LXR<VM>) -> Vec<VM::VMSlot> {
        let mut slots = vec![];
        let mut cursor = block.start();
        while cursor < block.end() {
            let granule = cursor;
            cursor += rc::MIN_OBJECT_SIZE;
            let count = lxr.rc.count_by_address(granule);
            if count == 0 {
                continue;
            }
            let object = unsafe { ObjectReference::from_raw_address_unchecked(granule) };
            if Line::is_aligned(granule) && count == 1 && lxr.rc.object_is_in_straddle_line_no_rc_check(object) {
                continue;
            }
            if lxr.immix_space.is_marked(object) {
                object.iterate_fields::<VM, _>(VMThread::UNINITIALIZED, |slot| {
                    if slot.load().is_some_and(|target| lxr.in_defrag(target)) {
                        slots.push(slot);
                    }
                });
            }
        }
        slots
    }

    fn select_fragmented_blocks(
        &self,
        selected_blocks: &mut Vec<Block>,
        copy_bytes: &mut usize,
        max_copy_bytes: usize,
        kept_blocks: &[crate::util::Address],
    ) {
        let mut blocks = Vec::with_capacity(self.fragmented_blocks_size.load(Ordering::SeqCst));
        while let Some(mut x) = self.fragmented_blocks.pop() {
            blocks.append(&mut x);
        }
        blocks.sort_by_key(|x| x.1);
        while let Some((block, _dead_bytes)) = blocks.pop() {
            if Self::skip_block(block) || kept_blocks.binary_search(&block.start()).is_ok() {
                continue;
            }
            block.set_as_defrag_source(true);
            selected_blocks.push(block);
            *copy_bytes += (Block::BYTES - (block.calc_dead_lines() << Line::LOG_BYTES)) >> 1;
            if *copy_bytes >= max_copy_bytes {
                break;
            }
        }
    }

    #[allow(clippy::assertions_on_constants)]
    pub fn select_mature_evacuation_candidates<VM: VMBinding>(&self, lxr: &LXR<VM>) {
        debug_assert!(crate::plan::lxr::MATURE_EVACUATION);
        if lxr.current_pause().unwrap() == Pause::Full {
            // Make sure LOS sweeping finishes before evac selection begin
            // FIXME: This can be done in parallel with SelectDefragBlocksInChunk packets
            let los = lxr.common().get_los();
            los.release_rc_nursery_objects();
        }
        // Select mature defrag blocks
        let available_clean_pages_for_defrag = if lxr.current_pause().unwrap() == Pause::Full {
            lxr.get_total_pages()
                .saturating_sub(lxr.get_used_pages())
                .max(lxr.immix_space.defrag_headroom_pages())
        } else {
            lxr.immix_space.defrag_headroom_pages()
        };
        let max_copy_bytes = available_clean_pages_for_defrag << LOG_BYTES_IN_PAGE;
        let mut copy_bytes = 0usize;
        let mut selected_blocks = vec![];
        // Blocks this pause copied objects into are left out, since emptying them again in the same pause would copy their objects twice, and so are blocks holding objects that must not move.
        let mut kept_blocks = lxr.block_allocation.copy_target_blocks();
        while let Some(block) = self.pinned_blocks.pop() {
            kept_blocks.push(block.start());
        }
        kept_blocks.sort_unstable();
        self.select_fragmented_blocks(&mut selected_blocks, &mut copy_bytes, max_copy_bytes, &kept_blocks);
        self.num_defrag_blocks
            .store(selected_blocks.len(), Ordering::SeqCst);
        let mut defrag_blocks = self.defrag_blocks.lock().unwrap();
        *defrag_blocks = selected_blocks;
        // cleanup
        assert!(self.fragmented_blocks.is_empty());
        assert!(self.blocks_in_fragmented_chunks.is_empty());
        self.fragmented_blocks_size.store(0, Ordering::SeqCst);
        self.blocks_in_fragmented_chunks_size
            .store(0, Ordering::SeqCst);
    }
}
