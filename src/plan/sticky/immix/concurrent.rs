//! Concurrent full-heap marking for StickyImmix. An initial-mark pause, which is also a nursery collection, takes the snapshot: every object live at its end is old. Collector threads then mark old objects while mutators run and nursery collections go on, the write barrier records what an old object holds before its first change, and a final-mark pause, again a nursery collection, finishes the mark and sweeps the old space. Old objects never move while the mark runs, and young ones are never traced: they were allocated after the snapshot, so they are live until a nursery collection decides otherwise.

use std::marker::PhantomData;

use crate::global_state::GcStatus;
use crate::plan::barriers::BarrierSemantics;
use crate::plan::generational::barrier::GenObjectBarrierSemantics;
use crate::plan::generational::gc_work::GenNurseryTrace;
use crate::plan::tracing::gc_work::closure::ProcessSlots;
use crate::plan::tracing::gc_work::root::DefaultRootsWorkFactory;
use crate::plan::{ConcurrentPlan, VectorQueue};
use crate::policy::gc_work::{DEFAULT_TRACE, TRACE_KIND_TRANSITIVE_PIN};
use crate::scheduler::gc_work::RootKind;
use crate::scheduler::{GCWork, GCWorker, WorkBucketStage};
use crate::util::{ObjectReference, VMMutatorThread};
use crate::vm::slot::Slot;
use crate::vm::{RootsWorkFactory, Scanning, VMBinding};
use crate::MMTK;

use super::global::StickyImmix;

type NurseryTrace<VM> = GenNurseryTrace<VM, StickyImmix<VM>, DEFAULT_TRACE>;
type NurseryPinningTrace<VM> = GenNurseryTrace<VM, StickyImmix<VM>, TRACE_KIND_TRANSITIVE_PIN>;

// How many objects a marking packet traces between checks for a requested pause.
const YIELD_INTERVAL: usize = 256;
// A marking packet whose queue grows past this hands half of it to a new packet, which idle collector threads can take.
const SPLIT_LENGTH: usize = 4096;

fn sticky_immix<VM: VMBinding>(mmtk: &'static MMTK<VM>) -> &'static StickyImmix<VM> {
    mmtk.get_plan().downcast_ref::<StickyImmix<VM>>().unwrap()
}

// ── Write barrier ───

/// StickyImmix's object barrier. It logs old objects for nursery collections, and while a concurrent mark runs it first records every unmarked old object the object holds, before the write that could lose one.
pub struct StickyImmixBarrierSemantics<VM: VMBinding> {
    generational: GenObjectBarrierSemantics<VM, StickyImmix<VM>>,
    plan: &'static StickyImmix<VM>,
    tls: VMMutatorThread,
    snapshot: VectorQueue<ObjectReference>,
}

impl<VM: VMBinding> StickyImmixBarrierSemantics<VM> {
    pub fn new(mmtk: &'static MMTK<VM>, tls: VMMutatorThread) -> Self {
        let plan = sticky_immix(mmtk);
        Self {
            generational: GenObjectBarrierSemantics::new(mmtk, plan),
            plan,
            tls,
            snapshot: VectorQueue::new(),
        }
    }

    fn record(&mut self, object: ObjectReference) {
        if self.plan.is_unmarked_old_object(object) {
            self.snapshot.push(object);
            if self.snapshot.is_full() {
                self.flush_snapshot();
            }
        }
    }

    // A full-heap pause that replaces the mark drops what is still queued for it.
    fn flush_snapshot(&mut self) {
        if !self.snapshot.is_empty() {
            let objects = self.snapshot.take();
            if self.plan.is_concurrent_marking() {
                self.plan.add_marking_work(ConcurrentMark::<VM>::new(objects, false));
            }
        }
    }
}

impl<VM: VMBinding> BarrierSemantics for StickyImmixBarrierSemantics<VM> {
    type VM = VM;

    fn flush(&mut self) {
        self.generational.flush();
        self.flush_snapshot();
    }

    fn object_reference_write_slow(&mut self, src: ObjectReference, slot: VM::VMSlot, target: Option<ObjectReference>) {
        self.generational.object_reference_write_slow(src, slot, target);
    }

    fn memory_region_copy_slow(&mut self, src: VM::VMMemorySlice, dst: VM::VMMemorySlice) {
        self.generational.memory_region_copy_slow(src, dst);
    }

    fn object_probable_write_slow(&mut self, obj: ObjectReference) {
        self.generational.object_probable_write_slow(obj);
    }

    fn before_logging(&mut self, src: ObjectReference) {
        if self.plan.is_concurrent_marking() {
            src.iterate_fields::<VM, _>(self.tls.0, |slot| {
                if let Some(target) = slot.load() {
                    self.record(target);
                }
            });
        }
    }

    // A referent read while the mark runs may become strongly reachable after the snapshot, which the mark would otherwise miss.
    fn load_weak_reference(&mut self, referent: ObjectReference) {
        if self.plan.is_concurrent_marking() {
            self.record(referent);
        }
    }
}

// ── Marking ───

/// Marks old objects for the concurrent mark and traces what they hold.
pub(super) struct ConcurrentMark<VM: VMBinding> {
    objects: Vec<ObjectReference>,
    /// Whether `objects` are scanned as they are: objects this mark has already marked, or image-heap objects, which are roots.
    marked: bool,
    phantom: PhantomData<VM>,
}

impl<VM: VMBinding> ConcurrentMark<VM> {
    pub(super) fn new(objects: Vec<ObjectReference>, marked: bool) -> Self {
        Self {
            objects,
            marked,
            phantom: PhantomData,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for ConcurrentMark<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let plan = sticky_immix(mmtk);
        let mut stack = std::mem::take(&mut self.objects);
        if !self.marked {
            stack.retain(|&object| plan.attempt_concurrent_mark(object));
        }
        // Only the final-mark pause runs marking work inside a pause, and it must finish it.
        let concurrent = plan.current_pause().is_none();
        let stage = if concurrent { WorkBucketStage::ConcurrentResumable } else { WorkBucketStage::FinishConcurrentWork };
        let mut traced = 0;
        while let Some(object) = stack.pop() {
            <VM::VMScanning as Scanning<VM>>::scan_object(worker.tls, object, &mut |slot: VM::VMSlot| {
                if let Some(child) = slot.load() {
                    if plan.attempt_concurrent_mark(child) {
                        stack.push(child);
                    }
                }
            });
            traced += 1;
            if stack.len() > SPLIT_LENGTH {
                let half = stack.split_off(stack.len() / 2);
                worker.add_work(stage, ConcurrentMark::<VM>::new(half, true));
            }
            // A pause waits for every collector thread to park, so a long trace hands back what it has left.
            if concurrent && traced % YIELD_INTERVAL == 0 && mmtk.state.gc_status.load() == GcStatus::PauseRequested {
                if !stack.is_empty() {
                    plan.add_marking_work(ConcurrentMark::<VM>::new(stack, true));
                }
                return;
            }
        }
    }
}

// ── Initial-mark roots ───

/// Root scanning for the initial-mark pause. Roots go to the nursery collection as usual, and what each root holds once the collection is done with it goes to the concurrent mark, so the stacks are walked once.
pub(super) struct InitialMarkRootsWorkFactory<VM: VMBinding> {
    mmtk: &'static MMTK<VM>,
    nursery: DefaultRootsWorkFactory<VM, NurseryTrace<VM>, NurseryPinningTrace<VM>>,
}

impl<VM: VMBinding> InitialMarkRootsWorkFactory<VM> {
    pub(super) fn new(mmtk: &'static MMTK<VM>) -> Self {
        Self {
            mmtk,
            nursery: DefaultRootsWorkFactory::new(mmtk),
        }
    }

    // Root objects never move, so they go to the mark as they are. An image-heap object among them is scanned by the mark, since what it holds is a root.
    fn mark_root_objects(&self, nodes: &[ObjectReference]) -> Vec<ObjectReference> {
        let plan = sticky_immix(self.mmtk);
        let (collected, image_heap): (Vec<_>, Vec<_>) = nodes.iter().partition(|&&object| plan.is_in_collected_space(object));
        if !image_heap.is_empty() {
            plan.add_marking_work(ConcurrentMark::<VM>::new(image_heap, true));
        }
        if !collected.is_empty() {
            plan.add_marking_work(ConcurrentMark::<VM>::new(collected.clone(), false));
        }
        collected
    }
}

impl<VM: VMBinding> Clone for InitialMarkRootsWorkFactory<VM> {
    fn clone(&self) -> Self {
        Self {
            mmtk: self.mmtk,
            nursery: self.nursery.clone(),
        }
    }
}

impl<VM: VMBinding> RootsWorkFactory<VM::VMSlot> for InitialMarkRootsWorkFactory<VM> {
    fn create_process_roots_work_with_root_kind(&mut self, slots: Vec<VM::VMSlot>, _kind: RootKind) {
        #[cfg(feature = "sanity")]
        self.mmtk.sanity_checker.lock().unwrap().add_root_slots(slots.clone());
        crate::memory_manager::add_work_packet(self.mmtk, WorkBucketStage::Closure, ProcessRootSlotsForMarking::<VM> { slots });
    }

    #[cfg(feature = "non_moving_root")]
    fn create_process_pinning_roots_work(&mut self, nodes: Vec<ObjectReference>) {
        let collected = self.mark_root_objects(&nodes);
        if !collected.is_empty() {
            self.nursery.create_process_pinning_roots_work(collected);
        }
    }

    #[cfg(feature = "non_moving_root")]
    fn create_process_tpinning_roots_work(&mut self, nodes: Vec<ObjectReference>) {
        let collected = self.mark_root_objects(&nodes);
        if !collected.is_empty() {
            self.nursery.create_process_tpinning_roots_work(collected);
        }
    }
}

/// The nursery collection's work for a packet of root slots, after which each slot holds where its object now lives, which the mark takes as a root.
struct ProcessRootSlotsForMarking<VM: VMBinding> {
    slots: Vec<VM::VMSlot>,
}

impl<VM: VMBinding> GCWork<VM> for ProcessRootSlotsForMarking<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let slots = std::mem::take(&mut self.slots);
        GCWork::do_work(&mut ProcessSlots::<NurseryTrace<VM>>::new(slots.clone(), WorkBucketStage::Closure), worker, mmtk);
        let roots: Vec<ObjectReference> = slots.iter().filter_map(|slot| slot.load()).collect();
        if !roots.is_empty() {
            sticky_immix(mmtk).add_marking_work(ConcurrentMark::<VM>::new(roots, false));
        }
    }
}
