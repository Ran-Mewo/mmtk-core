use super::defrag::StatsForDefrag;
use super::line::*;
use super::{block::*, defrag::Defrag};
use crate::plan::tracing::OptionObjectQueue;
use crate::plan::Pause;
use crate::policy::gc_work::{TraceKind, DEFAULT_TRACE, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::sft::GCWorkerMutRef;
use crate::policy::sft::SFT;
use crate::policy::sft_map::SFTMap;
use crate::policy::space::{CommonSpace, Space};
use crate::scheduler::gc_work::PrepareCollector;
use crate::util::alloc::allocator::AllocationOptions;
use crate::util::alloc::allocator::AllocatorContext;
use crate::util::constants::LOG_BYTES_IN_PAGE;
use crate::util::heap::chunk_map::*;
use crate::util::heap::BlockPageResource;
use crate::util::heap::PageResource;
use crate::util::linear_scan::{Region, RegionIterator, UnstraddlableRegion};
use crate::util::metadata::log_bit::UnlogBitsOperation;
use crate::util::metadata::side_metadata::spec_defs::{CONC_COVERED, CONC_MARK, IX_LINE_REUSE_COUNT};
use crate::util::metadata::side_metadata::*;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::{self, MetadataSpec};
use crate::util::object_enum::ObjectEnumerator;
use crate::util::object_forwarding;
use crate::util::rc::RefCountHelper;
use crate::util::{copy::*, epilogue, object_enum};
use crate::util::{Address, ObjectReference};
use crate::vm::*;
use crate::{
    plan::ObjectQueue,
    scheduler::{GCWork, GCWorkScheduler, GCWorker, WorkBucketStage},
    util::opaque_pointer::{VMThread, VMWorkerThread},
    MMTK,
};
use atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::OnceLock;
use std::sync::{atomic::AtomicU8, Arc};

pub(crate) const TRACE_KIND_FAST: TraceKind = 0;
pub(crate) const TRACE_KIND_DEFRAG: TraceKind = 1;

/// Whether RC-mode mature-space evacuation is compiled in.
const LXR_MATURE_EVACUATION: bool = !cfg!(feature = "lxr_no_mature_evac");

/// Plan-level hooks invoked by ImmixSpace during mutator allocation.
/// Default impls are no-ops; LXR provides the concrete implementation.
pub trait ImmixHooks<VM: VMBinding>: Send + Sync {
    /// Called after a fresh clean block is acquired. `copy` distinguishes
    /// mutator vs. GC-copy allocation. The hook owns any plan-specific
    /// per-block bookkeeping (e.g. nursery list, mark-table init).
    fn on_clean_block_acquired(&self, _block: Block, _copy: bool) {}
    /// Called after a reusable block is handed out to a mutator.
    fn on_reusable_block_acquired(&self, _block: Block, _copy: bool) {}
    /// Whether tracing is in progress; consulted on the mutator
    /// reused-line fast path so newly handed-out lines can be marked.
    fn cm_in_progress_or_final_mark(&self) -> bool {
        false
    }
}

pub struct ImmixSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: BlockPageResource<VM, Block>,
    /// Allocation status for all chunks in immix space
    pub chunk_map: ChunkMap,
    /// Current line mark state
    pub line_mark_state: AtomicU8,
    /// Line mark state in previous GC
    line_unavail_state: AtomicU8,
    /// A list of all reusable blocks
    pub reusable_blocks: ReusableBlockPool,
    /// Blocks handed to an allocator since the last GC. A nursery GC of a mixed-age space sweeps
    /// only these: no other block holds young objects, so their line marks and states are the
    /// same as when they were last swept.
    blocks_allocated_since_gc: std::sync::Mutex<Vec<Block>>,
    /// The sweep a nursery collection left for after its pause, see [`Self::release_nursery_later`].
    nursery_sweep: std::sync::Mutex<NurserySweep>,
    /// Chunks of `nursery_sweep` not swept yet, counting those being swept.
    nursery_sweep_chunks_left: AtomicUsize,
    /// Whether nursery GCs have swept only some blocks since the last full sweep, so that
    /// `defrag.mark_histograms` no longer covers every block.
    mark_histograms_stale: bool,
    /// Defrag utilities
    pub(super) defrag: Defrag,
    /// How many lines have been consumed since last GC?
    lines_consumed: AtomicUsize,
    reused_lines_consumed: AtomicUsize,
    /// Object mark state
    mark_state: u8,
    /// Work packet scheduler
    scheduler: Arc<GCWorkScheduler<VM>>,
    /// Some settings for this space
    space_args: ImmixSpaceArgs,
    hooks: OnceLock<&'static dyn ImmixHooks<VM>>,
    pub rc_enabled: bool,
    pub is_end_of_satb_or_full_gc: bool,
    pub rc: RefCountHelper<VM>,
    /// Whether the plan marks the old space concurrently, which maps the `CONC_MARK` table.
    uses_concurrent_marking: bool,
    /// Whether a concurrent full-heap mark is running, from the end of its initial pause to the end of its final one. An object promoted meanwhile is live for it.
    concurrent_marking: AtomicBool,
    /// Whether the final-mark pause is deciding liveness, when an old object is live only if the concurrent mark reached it.
    final_mark: AtomicBool,
    /// Whether the coming full-heap collection replaces an unfinished concurrent mark, whose live lines then carry either of two epochs.
    concurrent_mark_replaced: AtomicBool,
    /// Sweep packets still to run after the last concurrent full-heap mark.
    sweep_packets_left: AtomicUsize,
}

/// Some arguments for Immix Space.
pub struct ImmixSpaceArgs {
    /// Whether this ImmixSpace instance contains both young and old objects.
    /// This affects the updating of valid-object bits.  If some lines or blocks of this ImmixSpace
    /// instance contain young objects, their VO bits need to be updated during this GC.  Currently
    /// only StickyImmix is affected.  GenImmix allocates young objects in a separete CopySpace
    /// nursery and its VO bits can be cleared in bulk.
    pub mixed_age: bool,
    /// Disable copying for this Immix space.
    pub never_move_objects: bool,
}

unsafe impl<VM: VMBinding> Sync for ImmixSpace<VM> {}

impl<VM: VMBinding> SFT for ImmixSpace<VM> {
    fn name(&self) -> &'static str {
        self.get_name()
    }

    fn get_forwarded_object(&self, object: ObjectReference) -> Option<ObjectReference> {
        // If we never move objects, look no further.
        if !self.is_movable() {
            return None;
        }

        if object_forwarding::is_forwarded::<VM>(object) {
            Some(object_forwarding::read_forwarding_pointer::<VM>(object))
        } else {
            None
        }
    }

    fn is_live(&self, object: ObjectReference) -> bool {
        if self.rc_enabled {
            if self.is_end_of_satb_or_full_gc {
                if self.is_marked(object) {
                    let block = Block::containing(object);
                    if block.is_defrag_source() {
                        if object_forwarding::is_forwarded::<VM>(object) {
                            let forwarded =
                                object_forwarding::read_forwarding_pointer::<VM>(object);
                            return self.is_marked(forwarded) && self.rc.count(forwarded) > 0;
                        } else {
                            return false;
                        }
                    }
                    return self.rc.count(object) > 0;
                } else if object_forwarding::is_forwarded::<VM>(object) {
                    let forwarded = object_forwarding::read_forwarding_pointer::<VM>(object);
                    debug_assert!(
                        forwarded.to_raw_address().is_mapped(),
                        "Invalid forwarded object: {:?} -> {:?}",
                        object,
                        forwarded
                    );
                    return self.is_marked(forwarded) && self.rc.count(forwarded) > 0;
                } else {
                    return false;
                }
            }
            return self.rc.count(object) > 0 || object_forwarding::is_forwarded::<VM>(object);
        }
        // If the mark bit is set, it is live.
        if self.is_marked(object) {
            return !self.final_mark.load(Ordering::Relaxed) || self.is_concurrent_marked(object);
        }

        // If we never move objects, look no further.
        if !self.is_movable() {
            return false;
        }

        // If the object is forwarded, it is live, too.
        object_forwarding::is_forwarded::<VM>(object)
    }

    fn is_reachable(&self, object: ObjectReference) -> bool {
        if self.rc_enabled {
            if object_forwarding::is_forwarded::<VM>(object) {
                let forwarded = object_forwarding::read_forwarding_pointer::<VM>(object);
                return self.is_marked(forwarded) && self.rc.count(forwarded) > 0;
            }
            self.is_marked(object) && self.rc.count(object) > 0
        } else {
            self.is_live(object)
        }
    }
    #[cfg(feature = "object_pinning")]
    fn pin_object(&self, object: ObjectReference) -> bool {
        if self.space_args.never_move_objects {
            false
        } else {
            VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.pin_object::<VM>(object)
        }
    }
    #[cfg(feature = "object_pinning")]
    fn unpin_object(&self, object: ObjectReference) -> bool {
        if self.space_args.never_move_objects {
            false
        } else {
            VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.unpin_object::<VM>(object)
        }
    }
    #[cfg(feature = "object_pinning")]
    fn is_object_pinned(&self, object: ObjectReference) -> bool {
        if self.space_args.never_move_objects {
            true
        } else {
            VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.is_object_pinned::<VM>(object)
        }
    }
    fn is_movable(&self) -> bool {
        !self.space_args.never_move_objects
    }

    #[cfg(feature = "sanity")]
    fn is_sane(&self) -> bool {
        true
    }
    fn initialize_object_metadata(&self, _object: ObjectReference, _bytes: usize) {
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::set_vo_bit(_object);
    }
    #[cfg(feature = "vo_bit")]
    fn is_mmtk_object(&self, addr: Address) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::is_vo_bit_set_for_addr(addr)
    }
    #[cfg(feature = "vo_bit")]
    fn find_object_from_internal_pointer(
        &self,
        ptr: Address,
        max_search_bytes: usize,
    ) -> Option<ObjectReference> {
        // We don't need to search more than the max object size in the immix space.
        let search_bytes = usize::min(super::MAX_IMMIX_OBJECT_SIZE, max_search_bytes);
        crate::util::metadata::vo_bit::find_object_from_internal_pointer::<VM>(ptr, search_bytes)
    }
    fn sft_trace_object(
        &self,
        _queue: &mut OptionObjectQueue,
        _object: ObjectReference,
        _worker: GCWorkerMutRef,
    ) -> ObjectReference {
        panic!("We do not use SFT to trace objects for Immix. sft_trace_object() cannot be used.")
    }

    fn debug_print_object_info(&self, object: ObjectReference) {
        println!("marked  = {}", self.is_marked(object));
        // The line mark table isn't mapped when RC is enabled (LXR tracks liveness via
        // block state and reference counts instead), so skip it in that case.
        if !self.rc_enabled {
            println!(
                "line marked = {}",
                Line::from_unaligned_address(object.to_raw_address()).is_marked(self.mark_state)
            );
        }
        println!(
            "block state = {:?}",
            Block::from_unaligned_address(object.to_raw_address()).get_state()
        );
        object_forwarding::debug_print_object_forwarding_info::<VM>(object);
        self.common.debug_print_object_global_info(object);
    }
}

impl<VM: VMBinding> Space<VM> for ImmixSpace<VM> {
    fn as_space(&self) -> &dyn Space<VM> {
        self
    }
    fn as_sft(&self) -> &(dyn SFT + Sync + 'static) {
        self
    }
    fn get_page_resource(&self) -> &dyn PageResource<VM> {
        &self.pr
    }
    fn maybe_get_page_resource_mut(&mut self) -> Option<&mut dyn PageResource<VM>> {
        Some(&mut self.pr)
    }
    fn common(&self) -> &CommonSpace<VM> {
        &self.common
    }
    fn initialize_sft(&self, sft_map: &mut dyn SFTMap) {
        self.common().initialize_sft(self.as_sft(), sft_map)
    }
    fn release_multiple_pages(&mut self, _start: Address) {
        panic!("immixspace only releases pages enmasse")
    }
    fn set_copy_for_sft_trace(&mut self, _semantics: Option<CopySemantics>) {
        panic!("We do not use SFT to trace objects for Immix. set_copy_context() cannot be used.")
    }

    fn enumerate_objects(&self, enumerator: &mut dyn ObjectEnumerator) {
        object_enum::enumerate_blocks_from_chunk_map::<Block>(enumerator, &self.chunk_map);
    }

    fn clear_side_log_bits(&self) {
        // Remove the following warning if we have a legitimate use case.
        warn!("ImmixSpace::clear_side_log_bits is single-treaded.  Consider clearing side metadata in per-chunk work packets.");

        let log_bit = VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.extract_side_spec();
        for chunk in self.chunk_map.all_chunks() {
            log_bit.bzero_metadata(chunk.start(), Chunk::BYTES);
        }
    }

    fn set_side_log_bits(&self) {
        // Remove the following warning if we have a legitimate use case.
        warn!("ImmixSpace::set_side_log_bits is single-treaded.  Consider setting side metadata in per-chunk work packets.");

        let log_bit = VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.extract_side_spec();
        for chunk in self.chunk_map.all_chunks() {
            log_bit.bset_metadata(chunk.start(), Chunk::BYTES);
        }
    }
}

impl<VM: VMBinding> crate::policy::gc_work::PolicyTraceObject<VM> for ImmixSpace<VM> {
    fn trace_object<Q: ObjectQueue, const KIND: TraceKind>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        copy: Option<CopySemantics>,
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        if KIND == TRACE_KIND_TRANSITIVE_PIN {
            self.trace_object_without_moving(queue, object)
        } else if KIND == TRACE_KIND_DEFRAG {
            if Block::containing(object).is_defrag_source() {
                debug_assert!(self.in_defrag());
                debug_assert!(
                    !crate::plan::is_nursery_gc(worker.mmtk.get_plan()),
                    "Calling PolicyTraceObject on Immix in nursery GC"
                );
                self.trace_object_with_opportunistic_copy(
                    queue,
                    object,
                    copy.unwrap(),
                    worker,
                    // This should not be nursery collection. Nursery collection does not use PolicyTraceObject.
                    false,
                )
            } else {
                self.trace_object_without_moving(queue, object)
            }
        } else if KIND == TRACE_KIND_FAST {
            self.trace_object_without_moving(queue, object)
        } else {
            unreachable!()
        }
    }

    fn post_scan_object(&self, object: ObjectReference) {
        if super::MARK_LINE_AT_SCAN_TIME && !super::BLOCK_ONLY {
            debug_assert!(self.in_space(object));
            self.mark_lines(object);
        }
    }

    #[allow(clippy::if_same_then_else)] // DEFAULT_TRACE needs a workaround which is documented below.
    fn may_move_objects<const KIND: TraceKind>() -> bool {
        if KIND == TRACE_KIND_DEFRAG {
            true
        } else if KIND == TRACE_KIND_FAST || KIND == TRACE_KIND_TRANSITIVE_PIN {
            false
        } else if KIND == DEFAULT_TRACE {
            // FIXME: This is hacky. When we do a default trace, this should be a nonmoving space.
            // The only exception is the nursery GC for sticky immix, for which, we use default trace.
            // This function is only used for PlanProcessEdges, and for sticky immix nursery GC, we use
            // GenNurseryProcessEdges. So it still works. But this is quite hacky anyway.
            // See https://github.com/mmtk/mmtk-core/issues/1314 for details.
            false
        } else {
            unreachable!()
        }
    }
}

impl<VM: VMBinding> ImmixSpace<VM> {
    #[allow(unused)]
    const UNMARKED_STATE: u8 = 0;
    const MARKED_STATE: u8 = 1;

    /// Get side metadata specs
    fn side_metadata_specs(rc_enabled: bool) -> Vec<SideMetadataSpec> {
        if rc_enabled {
            let meta = vec![
                MetadataSpec::OnSide(Block::DEFRAG_STATE_TABLE),
                MetadataSpec::OnSide(Block::MARK_TABLE),
                *VM::VMObjectModel::LOCAL_MARK_BIT_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC,
                MetadataSpec::OnSide(crate::util::rc::RC_STRADDLE_LINES),
                MetadataSpec::OnSide(Block::LOG_TABLE),
                MetadataSpec::OnSide(Block::NURSERY_PROMOTION_STATE_TABLE),
                MetadataSpec::OnSide(IX_LINE_REUSE_COUNT),
            ];
            return metadata::extract_side_metadata(&meta);
        }
        metadata::extract_side_metadata(&if super::BLOCK_ONLY {
            vec![
                MetadataSpec::OnSide(Block::DEFRAG_STATE_TABLE),
                MetadataSpec::OnSide(Block::MARK_TABLE),
                *VM::VMObjectModel::LOCAL_MARK_BIT_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC,
                #[cfg(feature = "object_pinning")]
                *VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC,
            ]
        } else {
            vec![
                MetadataSpec::OnSide(Line::MARK_TABLE),
                MetadataSpec::OnSide(Block::DEFRAG_STATE_TABLE),
                MetadataSpec::OnSide(Block::MARK_TABLE),
                *VM::VMObjectModel::LOCAL_MARK_BIT_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC,
                #[cfg(feature = "object_pinning")]
                *VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC,
            ]
        })
    }

    pub fn new(
        args: crate::policy::space::PlanCreateSpaceArgs<VM>,
        mut space_args: ImmixSpaceArgs,
    ) -> Self {
        if args.unlog_traced_object {
            assert!(
                args.constraints.needs_log_bit,
                "Invalid args when the plan does not use log bit"
            );
        }

        // Make sure we override the space args if we force non moving Immix
        if cfg!(feature = "immix_non_moving") && !space_args.never_move_objects {
            info!(
                "Overriding never_moves_objects for Immix Space {}, as the immix_non_moving feature is set. Block size: 2^{}",
                args.name,
                Block::LOG_BYTES,
            );
            space_args.never_move_objects = true;
        }

        // validate features
        if super::BLOCK_ONLY {
            assert!(
                space_args.never_move_objects,
                "Block-only immix must not move objects"
            );
        }
        assert!(
            Block::LINES / 2 <= u8::MAX as usize - 2,
            "Number of lines in a block should not exceed BlockState::MARK_MARKED"
        );

        // TODO: The VO bit strategy is only relevant to tracing GC.
        // LXR currently ignores the strategy.
        #[cfg(feature = "vo_bit")]
        if !args.constraints.rc_enabled {
            vo_bit::helper::validate_config::<VM>();
        }

        let vm_map = args.vm_map;
        let scheduler = args.scheduler.clone();
        let rc_enabled = args.constraints.rc_enabled;
        let common = CommonSpace::new(args.into_policy_args(
            true,
            false,
            Self::side_metadata_specs(rc_enabled),
        ));
        let space_index = common.descriptor.get_index();
        ImmixSpace {
            pr: if common.vmrequest.is_discontiguous() {
                BlockPageResource::new_discontiguous(
                    Block::LOG_PAGES,
                    vm_map,
                    scheduler.num_workers(),
                )
            } else {
                BlockPageResource::new_contiguous(
                    Block::LOG_PAGES,
                    common.start,
                    common.extent,
                    vm_map,
                    scheduler.num_workers(),
                )
            },
            common,
            chunk_map: ChunkMap::new(space_index),
            line_mark_state: AtomicU8::new(Line::RESET_MARK_STATE),
            line_unavail_state: AtomicU8::new(Line::RESET_MARK_STATE),
            lines_consumed: AtomicUsize::new(0),
            reused_lines_consumed: AtomicUsize::new(0),
            reusable_blocks: ReusableBlockPool::new(scheduler.num_workers()),
            blocks_allocated_since_gc: std::sync::Mutex::new(Vec::new()),
            nursery_sweep: std::sync::Mutex::new(NurserySweep::default()),
            nursery_sweep_chunks_left: AtomicUsize::new(0),
            mark_histograms_stale: false,
            defrag: Defrag::default(),
            // Set to the correct mark state when inititialized. We cannot rely on prepare to set it (prepare may get skipped in nursery GCs).
            mark_state: Self::MARKED_STATE,
            scheduler,
            space_args,
            hooks: OnceLock::new(),
            rc_enabled,
            is_end_of_satb_or_full_gc: false,
            rc: RefCountHelper::NEW,
            uses_concurrent_marking: false,
            concurrent_marking: AtomicBool::new(false),
            final_mark: AtomicBool::new(false),
            concurrent_mark_replaced: AtomicBool::new(false),
            sweep_packets_left: AtomicUsize::new(0),
        }
    }

    /// Flush the thread-local queues in BlockPageResource
    pub fn flush_page_resource(&self) {
        // FIXME: Do we need this for LXR? We observed this to cause fails on conix.
        if !self.rc_enabled {
            self.reusable_blocks.flush_all();
        }
        #[cfg(target_pointer_width = "64")]
        self.pr.flush_all()
    }

    /// Get the number of defrag headroom pages.
    pub fn defrag_headroom_pages(&self) -> usize {
        self.defrag.defrag_headroom_pages(self)
    }

    /// Check if current GC is a defrag GC.
    pub fn in_defrag(&self) -> bool {
        self.defrag.in_defrag()
    }

    /// check if the current GC should do defragmentation.
    pub fn decide_whether_to_defrag(
        &self,
        emergency_collection: bool,
        collect_whole_heap: bool,
        collection_attempts: usize,
        user_triggered_collection: bool,
        full_heap_system_gc: bool,
    ) -> bool {
        self.defrag.decide_whether_to_defrag(
            self.is_defrag_enabled(),
            emergency_collection,
            collect_whole_heap,
            collection_attempts,
            user_triggered_collection,
            self.reusable_blocks.len() == 0,
            full_heap_system_gc,
            self.rc_enabled,
            *self.common.options.immix_always_defrag,
        );
        self.defrag.in_defrag()
    }

    /// Get work packet scheduler
    pub fn scheduler(&self) -> &GCWorkScheduler<VM> {
        &self.scheduler
    }

    /// Install the plan-level hooks. Called once by the owning plan during `gc_init`.
    pub fn install_hooks(&self, hooks: &'static dyn ImmixHooks<VM>) {
        self.hooks
            .set(hooks)
            .unwrap_or_else(|_| panic!("ImmixSpace::install_hooks called more than once"));
    }

    fn hooks(&self) -> Option<&'static dyn ImmixHooks<VM>> {
        self.hooks.get().copied()
    }

    pub fn prepare_rc(&mut self, pause: Pause) {
        // Initialize mark state for tracing
        if pause == Pause::Full || pause == Pause::InitialMark {
            // Update mark_state
            if VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.is_on_side() {
                self.mark_state = Self::MARKED_STATE;
            } else {
                // For header metadata, we use cyclic mark bits.
                unimplemented!("cyclic mark bits is not supported at the moment");
            }
        }
        // Release nursery blocks
        if pause != Pause::RefCount {
            if pause == Pause::Full {
                // Reset worker TLABs.
                // The block of the current worker TLAB may be selected as part of the mature evacuation set.
                for w in &self.scheduler().worker_group.workers_shared {
                    let result = w.designated_work.push(Box::new(PrepareCollector));
                    debug_assert!(result.is_ok());
                }
            }
            self.flush_page_resource();
        }
        if pause == Pause::FinalMark || pause == Pause::Full {
            self.is_end_of_satb_or_full_gc = true;
        }
    }

    pub fn release_rc(&mut self) {
        self.flush_page_resource();
        self.rc.reset_inc_buffer_size();
        self.is_end_of_satb_or_full_gc = false;
        self.reused_lines_consumed.store(0, Ordering::Relaxed);
    }

    pub(crate) fn prepare(
        &mut self,
        major_gc: bool,
        plan_stats: Option<StatsForDefrag>,
        unlog_bits_op: UnlogBitsOperation,
    ) {
        // This function should not be called during RC.
        // Otherwise the VO bit handling will be incorrect.
        debug_assert!(!self.rc_enabled);

        if major_gc {
            // Update mark_state
            if VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.is_on_side() {
                self.mark_state = Self::MARKED_STATE;
            } else {
                // For header metadata, we use cyclic mark bits.
                unimplemented!("cyclic mark bits is not supported at the moment");
            }

            // Prepare defrag info
            if self.is_defrag_enabled() {
                if !super::BLOCK_ONLY && self.mark_histograms_stale && self.in_defrag() {
                    self.recount_mark_histograms();
                }
                self.defrag.prepare(self, plan_stats.unwrap());
            }

            // Prepare each block for GC
            let threshold = self.defrag.defrag_spill_threshold.load(Ordering::Acquire);
            // A full collection that replaces an unfinished concurrent mark finds live lines with either the mark's epoch or the one before. The hole search can keep only one epoch unavailable besides the current one, so lines of the older epoch take the mark's.
            let relabel_lines = if !super::BLOCK_ONLY && self.concurrent_mark_replaced.swap(false, Ordering::SeqCst) {
                let older = self.line_unavail_state.load(Ordering::Acquire);
                let marking = self.line_mark_state.load(Ordering::Acquire);
                self.line_unavail_state.store(marking, Ordering::Release);
                Some((older, marking))
            } else {
                None
            };
            // # Safety: ImmixSpace reference is always valid within this collection cycle.
            let space = unsafe { &*(self as *const Self) };
            let work_packets = self.chunk_map.generate_tasks(|chunk| {
                Box::new(PrepareBlockState {
                    space,
                    chunk,
                    defrag_threshold: if space.in_defrag() {
                        Some(threshold)
                    } else {
                        None
                    },
                    unlog_bits_op,
                    relabel_lines,
                })
            });
            self.scheduler().work_buckets[WorkBucketStage::Prepare].bulk_add(work_packets);

            if !super::BLOCK_ONLY {
                self.line_mark_state.fetch_add(1, Ordering::AcqRel);
                if self.line_mark_state.load(Ordering::Acquire) > Line::MAX_MARK_STATE {
                    self.line_mark_state
                        .store(Line::RESET_MARK_STATE, Ordering::Release);
                }
            }
        }

        #[cfg(feature = "vo_bit")]
        if vo_bit::helper::need_to_clear_vo_bits_before_tracing::<VM>() {
            let maybe_scope = if major_gc {
                // If it is major GC, we always clear all VO bits because we are doing full-heap
                // tracing.
                Some(VOBitsClearingScope::FullGC)
            } else if self.space_args.mixed_age {
                // StickyImmix nursery GC.
                // Some lines (or blocks) contain only young objects,
                // while other lines (or blocks) contain only old objects.
                if super::BLOCK_ONLY {
                    // Block only.  Young objects are only allocated into fully empty blocks.
                    // Only clear unmarked blocks.
                    Some(VOBitsClearingScope::BlockOnly)
                } else {
                    // Young objects are allocated into empty lines.
                    // Only clear unmarked lines.
                    let line_mark_state = self.line_mark_state.load(Ordering::SeqCst);
                    Some(VOBitsClearingScope::Line {
                        state: line_mark_state,
                    })
                }
            } else {
                // GenImmix nursery GC.  We do nothing to the ImmixSpace because the nursery is a
                // separate CopySpace.  It'll clear its own VO bits.
                None
            };

            if let Some(scope) = maybe_scope {
                let work_packets = self
                    .chunk_map
                    .generate_tasks(|chunk| Box::new(ClearVOBitsAfterPrepare { chunk, scope }));
                self.scheduler.work_buckets[WorkBucketStage::ClearVOBits].bulk_add(work_packets);
            }
        }
    }

    /// Release for the immix space.
    pub(crate) fn release(&mut self, major_gc: bool, unlog_bits_op: UnlogBitsOperation) {
        debug_assert!(!self.rc_enabled);
        if major_gc {
            // Update line_unavail_state for hole searching after this GC.
            if !super::BLOCK_ONLY {
                self.line_unavail_state.store(
                    self.line_mark_state.load(Ordering::Acquire),
                    Ordering::Release,
                );
            }
        }
        let sweep_allocated_blocks_only =
            !major_gc && self.sweeps_allocated_blocks_only_in_nursery_gc();
        if sweep_allocated_blocks_only {
            // Blocks that are not swept keep their place in the reusable blocks list.
            debug_assert!(unlog_bits_op == UnlogBitsOperation::NoOp);
            let work_packets = self.generate_allocated_block_sweep_tasks();
            self.scheduler().work_buckets[WorkBucketStage::Release].bulk_add(work_packets);
            self.mark_histograms_stale = true;
        } else {
            // Clear reusable blocks list
            if !super::BLOCK_ONLY {
                self.reusable_blocks.reset();
            }
            // Sweep chunks and blocks
            let work_packets = self.generate_sweep_tasks(unlog_bits_op, false);
            self.scheduler().work_buckets[WorkBucketStage::Release].bulk_add(work_packets);
            self.blocks_allocated_since_gc.lock().unwrap().clear();
            self.mark_histograms_stale = false;
        }

        self.lines_consumed.store(0, Ordering::Relaxed);
    }

    /// Release at the end of a concurrent full-heap mark: sweep every block, keeping only lines the mark reached or that hold objects promoted while it ran. Nothing cleared the mark and log bits of the old space before the mark, as a full collection's prepare does, so the sweep clears them for every line it frees.
    pub(crate) fn release_after_concurrent_marking(&mut self) {
        debug_assert!(self.concurrent_marking.load(Ordering::Relaxed));
        self.line_unavail_state.store(self.line_mark_state.load(Ordering::Acquire), Ordering::Release);
        self.reusable_blocks.reset();
        let work_packets = self.generate_sweep_tasks(UnlogBitsOperation::NoOp, true);
        self.scheduler().work_buckets[WorkBucketStage::Release].bulk_add(work_packets);
        self.blocks_allocated_since_gc.lock().unwrap().clear();
        self.mark_histograms_stale = false;
        self.lines_consumed.store(0, Ordering::Relaxed);
    }

    /// The same release, with the sweep done after the pause: the returned packets sweep the blocks in use now, which no allocator can take until they are swept, while mutators allocate from clean blocks. Freed and reusable blocks become available when the last packet finishes.
    pub(crate) fn release_after_concurrent_marking_later(&mut self) -> Vec<Box<dyn GCWork<VM>>> {
        const BLOCKS_PER_PACKET: usize = 256;
        debug_assert!(self.concurrent_marking.load(Ordering::Relaxed));
        self.line_unavail_state.store(self.line_mark_state.load(Ordering::Acquire), Ordering::Release);
        self.reusable_blocks.reset();
        self.defrag.mark_histograms.lock().clear();
        let blocks: Vec<Block> = self.chunk_map.all_chunks().flat_map(|chunk| chunk.iter_region::<Block>()).filter(|block| block.get_state() != BlockState::Unallocated).collect();
        self.blocks_allocated_since_gc.lock().unwrap().clear();
        // Nursery collections may sweep reused blocks before the last packet adds the histograms of the rest.
        self.mark_histograms_stale = true;
        self.lines_consumed.store(0, Ordering::Relaxed);
        let packets = blocks.chunks(BLOCKS_PER_PACKET).count();
        self.sweep_packets_left.store(packets, Ordering::SeqCst);
        // # Safety: ImmixSpace reference is always valid within this collection cycle.
        let space = unsafe { &*(self as *const Self) };
        blocks.chunks(BLOCKS_PER_PACKET).map(|blocks| Box::new(SweepBlocksAfterMark { space, blocks: blocks.to_vec() }) as Box<dyn GCWork<VM>>).collect()
    }

    /// Whether blocks are still waiting for the sweep that follows a concurrent full-heap mark.
    pub(crate) fn is_sweeping_after_mark(&self) -> bool {
        self.sweep_packets_left.load(Ordering::SeqCst) != 0
    }

    /// The sweep of a nursery collection, left to [`Self::sweep_nursery_chunks`] once the world resumes, so the pause does not wait for it. It covers the blocks allocated since the last collection, which no allocator holds and none can take until they are swept; the line mark state is the one this pause marked with.
    /// The line marks a sweep reads must not change before it is done, so every pause first finishes what an earlier one left ([`Self::finish_nursery_sweep`]); a pause after which a concurrent mark starts, or ends, sweeps in the pause instead.
    pub(crate) fn release_nursery_later(&mut self, clear_forwarding_bits: bool) {
        const BLOCKS_PER_CHUNK: usize = 32;
        debug_assert!(self.sweeps_allocated_blocks_only_in_nursery_gc() && !self.has_nursery_sweep_left());
        let blocks = std::mem::take(&mut *self.blocks_allocated_since_gc.lock().unwrap());
        let chunks: Vec<Vec<Block>> = blocks.chunks(BLOCKS_PER_CHUNK).map(<[Block]>::to_vec).collect();
        self.nursery_sweep_chunks_left.store(chunks.len(), Ordering::SeqCst);
        *self.nursery_sweep.lock().unwrap() = NurserySweep {
            chunks,
            line_mark_state: if super::BLOCK_ONLY { None } else { Some(self.line_mark_state.load(Ordering::Acquire)) },
            clear_forwarding_bits,
        };
        self.mark_histograms_stale = true;
        self.lines_consumed.store(0, Ordering::Relaxed);
    }

    /// Packets that sweep what [`Self::release_nursery_later`] left, as many as collector threads can share it.
    pub(crate) fn nursery_sweep_packets(&self, workers: usize) -> Vec<Box<dyn GCWork<VM>>> {
        // # Safety: the space lives as long as the MMTk instance.
        let space = unsafe { &*(self as *const Self) };
        let packets = self.nursery_sweep_chunks_left.load(Ordering::SeqCst).min(workers);
        (0..packets).map(|_| Box::new(SweepNurseryBlocks { space }) as Box<dyn GCWork<VM>>).collect()
    }

    pub(crate) fn has_nursery_sweep_left(&self) -> bool {
        self.nursery_sweep_chunks_left.load(Ordering::SeqCst) != 0
    }

    /// Pages that a nursery sweep left for after its pause may still free.
    pub(crate) fn nursery_sweep_pages_left(&self) -> usize {
        self.nursery_sweep_chunks_left.load(Ordering::SeqCst) * 32 * Block::PAGES
    }

    /// Sweep chunks of the nursery sweep until none is left; whichever thread sweeps the last chunk makes the freed blocks available. Runs on a collector thread.
    fn sweep_nursery_chunks(&self) {
        loop {
            let (chunk, line_mark_state, clear_forwarding_bits) = {
                let mut sweep = self.nursery_sweep.lock().unwrap();
                match sweep.chunks.pop() {
                    Some(chunk) => (chunk, sweep.line_mark_state, sweep.clear_forwarding_bits),
                    None => return,
                }
            };
            let mut histogram = self.defrag.new_histogram();
            for &block in &chunk {
                // Only these blocks hold young objects, so only they can hold forwarded ones.
                if clear_forwarding_bits {
                    if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC {
                        side.bzero_metadata(block.start(), Block::BYTES);
                    }
                }
                block.sweep(self, &mut histogram, line_mark_state, false);
            }
            if self.nursery_sweep_chunks_left.fetch_sub(1, Ordering::SeqCst) == 1 {
                self.flush_page_resource();
            }
        }
    }

    /// Finish the nursery sweep an earlier pause left, waiting for chunks other collector threads are sweeping. Called at the start of every pause, on a collector thread.
    pub(crate) fn finish_nursery_sweep(&self) {
        if !self.has_nursery_sweep_left() {
            return;
        }
        self.sweep_nursery_chunks();
        while self.has_nursery_sweep_left() {
            std::hint::spin_loop();
        }
    }

    /// A full-heap collection drops the sweep that follows a mark and sweeps everything itself; its prepare clears the mark bits the dropped packets would have.
    pub(crate) fn abandon_sweep_after_mark(&self) {
        self.sweep_packets_left.store(0, Ordering::SeqCst);
    }

    /// Let the space take part in concurrent full-heap marking. Called once by the plan, which maps `CONC_MARK`.
    pub(crate) fn use_concurrent_marking(&mut self) {
        self.uses_concurrent_marking = true;
    }

    /// Start a concurrent full-heap mark at the end of its initial pause. Lines the marker and promotions mark from now on carry a new epoch, while nursery sweeps and the allocator still treat lines of the last epoch as live, until the final sweep.
    pub(crate) fn start_concurrent_marking(&self) {
        debug_assert!(self.uses_concurrent_marking && !super::BLOCK_ONLY);
        let next = self.line_mark_state.load(Ordering::Acquire) + 1;
        self.line_mark_state.store(if next > Line::MAX_MARK_STATE { Line::RESET_MARK_STATE } else { next }, Ordering::Release);
        self.concurrent_marking.store(true, Ordering::SeqCst);
    }

    /// From the start of the final-mark pause, old objects are live only if the concurrent mark reached them.
    pub(crate) fn start_final_mark(&self) {
        self.final_mark.store(true, Ordering::SeqCst);
    }

    /// End a concurrent full-heap mark, after its final sweep, or because a full collection replaces it. A full collection's prepare clears `CONC_MARK`.
    pub(crate) fn end_concurrent_marking(&self, replaced_by_full_heap_collection: bool) {
        if replaced_by_full_heap_collection {
            self.concurrent_mark_replaced.store(true, Ordering::SeqCst);
        }
        self.concurrent_marking.store(false, Ordering::SeqCst);
        self.final_mark.store(false, Ordering::SeqCst);
    }

    /// Mark an old object for the concurrent full-heap mark, and the lines it spans. Returns whether this call marked it; a young object is never marked, since it was allocated after the mark's snapshot.
    pub fn attempt_concurrent_mark(&self, object: ObjectReference) -> bool {
        if !self.is_marked(object) || CONC_MARK.fetch_or_atomic::<u8>(object.to_raw_address(), 1, Ordering::SeqCst) != 0 {
            return false;
        }
        self.mark_lines(object);
        true
    }

    pub fn is_concurrent_marked(&self, object: ObjectReference) -> bool {
        CONC_MARK.load_atomic::<u8>(object.to_raw_address(), Ordering::SeqCst) != 0
    }

    // A nursery collection promotes `object` while a concurrent mark runs: it is live for the mark, which never traces it.
    fn mark_promoted_for_concurrent_marking(&self, object: ObjectReference) {
        if self.concurrent_marking.load(Ordering::Relaxed) {
            CONC_MARK.store_atomic::<u8>(object.to_raw_address(), 1, Ordering::Relaxed);
        }
    }

    pub(super) fn line_unavail_state(&self) -> u8 {
        self.line_unavail_state.load(Ordering::Acquire)
    }

    /// Whether nursery GCs sweep only the blocks allocated since the last GC.
    fn sweeps_allocated_blocks_only_in_nursery_gc(&self) -> bool {
        self.space_args.mixed_age && !self.rc_enabled
    }

    fn record_allocated_block(&self, block: Block) {
        if self.sweeps_allocated_blocks_only_in_nursery_gc() {
            self.blocks_allocated_since_gc.lock().unwrap().push(block);
        }
    }

    /// Generate sweep tasks for the blocks allocated since the last GC.
    fn generate_allocated_block_sweep_tasks(&self) -> Vec<Box<dyn GCWork<VM>>> {
        const BLOCKS_PER_PACKET: usize = 32;
        let blocks = std::mem::take(&mut *self.blocks_allocated_since_gc.lock().unwrap());
        // # Safety: ImmixSpace reference is always valid within this collection cycle.
        let space = unsafe { &*(self as *const Self) };
        let epilogue = Arc::new(FlushPageResource {
            space,
            counter: AtomicUsize::new(0),
        });
        let tasks: Vec<Box<dyn GCWork<VM>>> = blocks
            .chunks(BLOCKS_PER_PACKET)
            .map(|blocks| {
                Box::new(SweepBlocks {
                    space,
                    blocks: blocks.to_vec(),
                    epilogue: epilogue.clone(),
                }) as Box<dyn GCWork<VM>>
            })
            .collect();
        epilogue.counter.store(tasks.len(), Ordering::SeqCst);
        tasks
    }

    /// Count `defrag.mark_histograms` again from the line marks of every block. Line marks only
    /// change during GCs, so this gives the histograms a sweep of every block at the end of the
    /// last GC would have given.
    fn recount_mark_histograms(&self) {
        let line_mark_state = self.line_mark_state.load(Ordering::Acquire);
        let mut histogram = self.defrag.new_histogram();
        for chunk in self.chunk_map.all_chunks() {
            for block in chunk
                .iter_region::<Block>()
                .filter(|block| block.get_state() != BlockState::Unallocated)
            {
                let (marked_lines, holes) = block.count_marked_lines_and_holes(line_mark_state);
                if marked_lines != 0 {
                    histogram[holes] += marked_lines;
                }
            }
        }
        let mut histograms = self.defrag.mark_histograms.lock();
        histograms.clear();
        histograms.push(histogram);
    }

    /// This is called when a GC finished.
    /// Return whether this GC was a defrag GC, as a plan may want to know this.
    pub fn end_of_gc(&mut self) -> bool {
        let did_defrag = self.defrag.in_defrag();
        if self.is_defrag_enabled() {
            self.defrag.reset_in_defrag();
        }
        did_defrag
    }

    /// Generate chunk sweep tasks
    fn generate_sweep_tasks(&self, unlog_bits_op: UnlogBitsOperation, after_concurrent_mark: bool) -> Vec<Box<dyn GCWork<VM>>> {
        self.defrag.mark_histograms.lock().clear();
        // # Safety: ImmixSpace reference is always valid within this collection cycle.
        let space = unsafe { &*(self as *const Self) };
        let epilogue = Arc::new(FlushPageResource {
            space,
            counter: AtomicUsize::new(0),
        });
        let tasks = self.chunk_map.generate_tasks(|chunk| {
            Box::new(SweepChunk {
                space,
                chunk,
                unlog_bits_op,
                after_concurrent_mark,
                epilogue: epilogue.clone(),
            })
        });
        epilogue.counter.store(tasks.len(), Ordering::SeqCst);
        tasks
    }

    /// Release a block.
    pub fn release_block(&self, block: Block, zero_unlog_table: bool) {
        if zero_unlog_table {
            block.clear_field_unlog_table::<VM>();
        }
        block.deinit(self);
        self.pr.release_block(block);
    }

    /// Allocate a clean block.
    pub fn get_clean_block(
        &self,
        tls: VMThread,
        copy: bool,
        alloc_options: AllocationOptions,
    ) -> Option<Block> {
        // Copied objects overwrite what they are copied into, and a line a copy leaves free is zeroed when an allocator takes it as a hole.
        let block_address = self.acquire_pages(tls, Block::PAGES, alloc_options, self.common().zeroed && !copy);
        if block_address.is_zero() {
            return None;
        }
        let block = Block::from_aligned_address(block_address);
        if !self.rc_enabled || self.defrag.in_defrag() {
            self.defrag.notify_new_clean_block(copy);
        }
        if let Some(hooks) = self.hooks() {
            hooks.on_clean_block_acquired(block, copy);
        }
        block.init(copy, false, self);
        self.chunk_map.set_allocated(block.chunk(), true);
        self.record_allocated_block(block);
        if !self.rc_enabled {
            self.lines_consumed
                .fetch_add(Block::LINES, Ordering::SeqCst);
        }
        Some(block)
    }

    /// Pop a reusable block from the reusable block list.
    pub fn get_reusable_block(&self, copy: bool) -> Option<Block> {
        if super::BLOCK_ONLY {
            return None;
        }
        loop {
            let block = self.reusable_blocks.pop()?;
            // Skip blocks that should be evacuated.
            if copy && block.is_defrag_source() {
                continue;
            }
            if self.rc_enabled {
                if LXR_MATURE_EVACUATION && block.is_defrag_source() {
                    continue;
                }
                // Blocks in the `reusable_blocks` queue can be released after some RC collections.
                // These blocks can either have `Unallocated` state, or be reallocated again.
                // Skip these cases and only return the truly reusable blocks.
                if !block.get_state().is_reusable() {
                    continue;
                }
                if !block.attempt_mutator_reuse() {
                    continue;
                }
                if let Some(hooks) = self.hooks() {
                    hooks.on_reusable_block_acquired(block, copy);
                }
            } else {
                // Get available lines. Do this before block.init which will reset block state.
                let lines_delta = match block.get_state() {
                    BlockState::Reusable { unavailable_lines } => {
                        Block::LINES - unavailable_lines as usize
                    }
                    BlockState::Unmarked => Block::LINES,
                    _ => unreachable!("{:?} {:?}", block, block.get_state()),
                };
                self.lines_consumed.fetch_add(lines_delta, Ordering::SeqCst);
            }

            block.init(copy, true, self);
            self.record_allocated_block(block);
            return Some(block);
        }
    }

    pub fn trace_object_without_moving_rc(
        &self,
        queue: &mut impl ObjectQueue,
        object: ObjectReference,
    ) -> ObjectReference {
        if self.attempt_mark(object) {
            let addr = object.to_raw_address().as_usize();
            let straddle = if (addr & 0b11110000) == 0 {
                self.rc.object_is_in_straddle_line_no_rc_check(object)
            } else {
                false
            };
            if !straddle {
                queue.enqueue(object);
            }
        }
        object
    }

    /// Trace and mark objects without evacuation.
    pub fn trace_object_without_moving(
        &self,
        queue: &mut impl ObjectQueue,
        object: ObjectReference,
    ) -> ObjectReference {
        // This function should not be called during RC if mature evacuation is not enabled.
        if LXR_MATURE_EVACUATION {
            debug_assert!(!self.rc_enabled);
        }

        #[cfg(feature = "vo_bit")]
        if !self.rc_enabled {
            // The VO bit strategy is currently not applicable to RC.
            // RC clears VO bits during sweeping.
            vo_bit::helper::on_trace_object::<VM>(object);
        }

        if self.attempt_mark(object) {
            if self.rc_enabled {
                let straddle = self.rc.object_is_in_straddle_line_no_rc_check(object);
                if straddle {
                    return object;
                }
            } else {
                // Mark block and lines
                if !super::BLOCK_ONLY {
                    if !super::MARK_LINE_AT_SCAN_TIME {
                        self.mark_lines(object);
                    }
                } else {
                    let block = Block::containing(object);
                    let state = block.get_state();
                    if state != BlockState::Nursery && state != BlockState::Marked {
                        block.set_state(BlockState::Marked);
                    }
                }
            }

            #[cfg(feature = "vo_bit")]
            if !self.rc_enabled {
                // The VO bit strategy is currently not applicable to RC.
                // RC clears VO bits during sweeping.
                vo_bit::helper::on_object_marked::<VM>(object);
            }

            // Visit node
            queue.enqueue(object);
            if !self.rc_enabled {
                self.unlog_object_if_needed(object);
            }
            return object;
        }
        object
    }

    /// Trace object and do evacuation if required.
    #[allow(clippy::assertions_on_constants)]
    pub fn trace_object_with_opportunistic_copy(
        &self,
        queue: &mut impl ObjectQueue,
        object: ObjectReference,
        semantics: CopySemantics,
        worker: &mut GCWorker<VM>,
        nursery_collection: bool,
    ) -> ObjectReference {
        // This function should not be called when RC is enabled.
        // Otherwise the VO bit handling will be incorrect.
        debug_assert!(!self.rc_enabled);

        let copy_context = worker.get_copy_context_mut();
        debug_assert!(!super::BLOCK_ONLY);

        #[cfg(feature = "vo_bit")]
        vo_bit::helper::on_trace_object::<VM>(object);

        let forwarding_status = object_forwarding::attempt_to_forward::<VM>(object);
        if object_forwarding::state_is_forwarded_or_being_forwarded(forwarding_status) {
            // We lost the forwarding race as some other thread has set the forwarding word; wait
            // until the object has been forwarded by the winner. Note that the object may not
            // necessarily get forwarded since Immix opportunistically moves objects.
            #[allow(clippy::let_and_return)]
            let new_object =
                object_forwarding::spin_and_get_forwarded_object::<VM>(object, forwarding_status);
            #[cfg(debug_assertions)]
            {
                if new_object == object {
                    debug_assert!(
                        self.is_marked(object) || self.defrag.space_exhausted() || self.is_pinned(object),
                        "Forwarded object is the same as original object {} even though it should have been copied",
                        object,
                    );
                } else {
                    // new_object != object
                    debug_assert!(
                        !Block::containing(new_object).is_defrag_source(),
                        "Block {:?} containing forwarded object {} should not be a defragmentation source",
                        Block::containing(new_object),
                        new_object,
                    );
                }
            }
            new_object
        } else if self.is_marked(object) {
            // We won the forwarding race but the object is already marked so we clear the
            // forwarding status and return the unmoved object
            object_forwarding::clear_forwarding_bits::<VM>(object);
            object
        } else {
            // We won the forwarding race; actually forward and copy the object if it is not pinned
            // and we have sufficient space in our copy allocator
            debug_assert!(!nursery_collection || !self.rc_enabled);
            let new_object = if self.is_pinned(object)
                || (!nursery_collection && self.defrag.space_exhausted())
            {
                self.attempt_mark(object);
                object_forwarding::clear_forwarding_bits::<VM>(object);
                Block::containing(object).set_state(BlockState::Marked);

                #[cfg(feature = "vo_bit")]
                vo_bit::helper::on_object_marked::<VM>(object);

                if !super::MARK_LINE_AT_SCAN_TIME {
                    self.mark_lines(object);
                }

                self.unlog_object_if_needed(object);

                object
            } else {
                // We are forwarding objects. When the copy allocator allocates the block, it should
                // mark the block. So we do not need to explicitly mark it here.
                // Clippy complains if the "vo_bit" feature is not enabled.
                #[allow(clippy::let_and_return)]
                let new_object = object_forwarding::try_forward_object::<VM>(
                    object,
                    semantics,
                    copy_context,
                    |new_object| {
                        // post_copy should have set the unlog bit
                        // if `unlog_traced_object` is true.
                        debug_assert!(
                            !self.common.unlog_traced_object
                                || VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                                    .is_unlogged::<VM>(new_object, Ordering::Relaxed)
                        );
                        #[cfg(feature = "vo_bit")]
                        vo_bit::helper::on_object_forwarded::<VM>(new_object);
                    },
                )
                .expect("to-space overflow");

                new_object
            };
            debug_assert!({
                let state = Block::containing(new_object).get_state();
                state == BlockState::Marked || state == BlockState::Nursery
            });

            queue.enqueue(new_object);
            debug_assert!(new_object.is_live());
            new_object
        }
    }

    pub fn rc_trace_object<Q: ObjectQueue>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        semantics: CopySemantics,
        pause: Pause,
        mark: bool,
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        debug_assert!(self.rc_enabled);
        if LXR_MATURE_EVACUATION && Block::containing(object).is_defrag_source() {
            self.trace_forward_rc_mature_object(queue, object, semantics, pause, worker)
        } else if LXR_MATURE_EVACUATION {
            self.trace_mark_rc_mature_object(queue, object, pause, mark)
        } else {
            self.trace_object_without_moving(queue, object)
        }
    }

    pub fn trace_mark_rc_mature_object(
        &self,
        queue: &mut impl ObjectQueue,
        object: ObjectReference,
        _pause: Pause,
        mark: bool,
    ) -> ObjectReference {
        debug_assert!(
            !object_forwarding::is_forwarded::<VM>(object),
            "object {:?} is forwarded",
            object
        );
        if mark && self.attempt_mark(object) {
            queue.enqueue(object);
        }
        object
    }

    #[allow(clippy::assertions_on_constants)]
    pub fn trace_forward_rc_mature_object<Q: ObjectQueue>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        _semantics: CopySemantics,
        _pause: Pause,
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        let copy_context = worker.get_copy_context_mut();
        let forwarding_status = object_forwarding::attempt_to_forward::<VM>(object);
        if object_forwarding::state_is_forwarded_or_being_forwarded(forwarding_status) {
            object_forwarding::spin_and_get_forwarded_object::<VM>(object, forwarding_status)
        } else {
            // Evacuate the mature object
            let new = object_forwarding::try_forward_object::<VM>(
                object,
                CopySemantics::DefaultCopy,
                copy_context,
                |_new_object| {
                    // When using RC, we set the VO bit of the forwarded object.
                    #[cfg(feature = "vo_bit")]
                    vo_bit::set_vo_bit(_new_object);
                },
            )
            .expect("to-space overflow");
            // Transfer RC count
            if new.get_size::<VM>() > Line::BYTES {
                self.rc.mark_straddle_object(new);
            }
            self.rc.set(new, self.rc.count(object));
            self.attempt_mark(new);
            self.unmark(object);
            queue.enqueue(new);
            debug_assert_ne!(
                self.rc.count(new),
                0,
                "ERROR Invalid {:?} rc={}",
                new,
                self.rc.count(new)
            );
            new
        }
    }

    fn unlog_object_if_needed(&self, object: ObjectReference) {
        debug_assert!(!self.rc_enabled);
        if self.common.unlog_traced_object {
            // Make sure the side metadata for the line can fit into one byte. For smaller line size, we should
            // use `mark_as_unlogged` instead to mark the bit.
            const_assert!(
                Line::BYTES
                    >= (1
                        << (crate::util::constants::LOG_BITS_IN_BYTE
                            + crate::util::constants::LOG_MIN_OBJECT_SIZE))
            );
            const_assert_eq!(
                crate::vm::object_model::specs::VMGlobalLogBitSpec::LOG_NUM_BITS,
                0
            ); // We should put this to the addition, but type casting is not allowed in constant assertions.

            // Every immix line is 256 bytes, which is mapped to 4 bytes in the side metadata.
            // If we have one object in the line that is mature, we can assume all the objects in the line are mature objects.
            // So we can just mark the byte.
            VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                .mark_byte_as_unlogged::<VM>(object, Ordering::Relaxed);
        }
    }

    /// Mark all the lines that the given object spans.
    #[allow(clippy::assertions_on_constants)]
    pub fn mark_lines(&self, object: ObjectReference) {
        debug_assert!(!super::BLOCK_ONLY);
        if self.rc_enabled {
            return;
        }
        Line::mark_lines_for_object::<VM>(object, self.line_mark_state.load(Ordering::Acquire));
    }

    /// Atomically mark an object.
    pub fn attempt_mark(&self, object: ObjectReference) -> bool {
        loop {
            let old_value = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.load_atomic::<VM, u8>(
                object,
                None,
                Ordering::SeqCst,
            );
            if old_value == self.mark_state {
                return false;
            }

            if VM::VMObjectModel::LOCAL_MARK_BIT_SPEC
                .compare_exchange_metadata::<VM, u8>(
                    object,
                    old_value,
                    self.mark_state,
                    None,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                break;
            }
        }
        self.mark_promoted_for_concurrent_marking(object);
        true
    }

    /// Atomically unmark an object.  Return true if it changed the mark bit from 1 to 0.
    pub fn unmark(&self, object: ObjectReference) -> bool {
        let result = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.fetch_update_metadata::<VM, u8, _>(
            object,
            Ordering::Relaxed,
            Ordering::Relaxed,
            |v| {
                if v != 1 {
                    return None;
                }
                Some(0)
            },
        );
        result.is_ok()
    }

    fn is_marked_with(&self, object: ObjectReference, mark_state: u8) -> bool {
        let old_value = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.load_atomic::<VM, u8>(
            object,
            None,
            Ordering::SeqCst,
        );
        old_value == mark_state
    }

    pub(crate) fn is_marked(&self, object: ObjectReference) -> bool {
        self.is_marked_with(object, self.mark_state)
    }

    /// Check if an object is pinned.
    fn is_pinned(&self, _object: ObjectReference) -> bool {
        #[cfg(feature = "object_pinning")]
        return self.is_object_pinned(_object);

        #[cfg(not(feature = "object_pinning"))]
        false
    }

    /// Hole searching.
    ///
    /// Linearly scan lines in a block to search for the next
    /// hole, starting from the given line. If we find available lines,
    /// return a tuple of the start line and the end line (non-inclusive).
    ///
    /// Returns None if the search could not find any more holes.
    #[allow(clippy::assertions_on_constants)]
    pub fn get_next_available_lines(&self, copy: bool, search_start: Line) -> Option<(Line, Line)> {
        debug_assert!(!super::BLOCK_ONLY);
        if self.rc_enabled {
            self.rc_get_next_available_lines(copy, search_start)
        } else {
            self.normal_get_next_available_lines(search_start)
        }
    }

    /// Search holes by ref-counts instead of line marks
    #[allow(clippy::assertions_on_constants)]
    pub fn rc_get_next_available_lines(
        &self,
        copy: bool,
        search_start: Line,
    ) -> Option<(Line, Line)> {
        debug_assert!(!super::BLOCK_ONLY);
        debug_assert!(self.rc_enabled);
        let block = search_start.block();
        let rc_array = RCArray::of(block);
        let limit = Block::LINES;
        // Find start
        let first_free_cursor = {
            let start_cursor = search_start.get_index_within_block();
            let mut first_free_cursor = None;
            let mut find_free_line = false;
            for i in start_cursor..limit {
                if rc_array.is_dead(i) {
                    if i == 0 {
                        first_free_cursor = Some(i);
                        break;
                    } else if !find_free_line {
                        // This skips the first line of a hole
                        // because `mark_straddle_object_with_size` may or may not set the RC
                        // of the last line an object straddles.
                        find_free_line = true;
                    } else {
                        first_free_cursor = Some(i);
                        break;
                    }
                } else {
                    find_free_line = false;
                }
            }
            first_free_cursor
        };
        let start = match first_free_cursor {
            Some(c) => c,
            _ => return None,
        };
        // Find limit
        let end = {
            let mut cursor = start + 1;
            while cursor < limit {
                if !rc_array.is_dead(cursor) {
                    break;
                }
                cursor += 1;
            }
            cursor
        };
        // For bindings without `UNIFIED_OBJECT_REFERENCE_ADDRESS`, the object start
        // may land in the line immediately before its object reference address. Such a line
        // can be considered as empty. We conservatively consider the last line in a hole
        // may hold an object start. This is the only place in LXR we need to handle the object start vs object ref issue.
        // TODO: This solution is conservative and not ideal, but is the simplest fix.
        // An alternative is to shift RC_TABLE (and possibly the RC_STRADDLE_LINES) for the upper bound of (obj ref - obj start),
        // when we search for holes. See https://github.com/mmtk/mmtk-core/pull/1576 as a half-done prototype.
        let end = if !VM::VMObjectModel::UNIFIED_OBJECT_REFERENCE_ADDRESS && end < limit {
            end - 1
        } else {
            end
        };
        if end <= start {
            // The reservation consumed the entire hole (it was exactly one line). Retry from
            // just past it instead of handing out an empty range.
            let next_search_start = Line::from_aligned_address(block.start()).next_nth(start + 1);
            return self.rc_get_next_available_lines(copy, next_search_start);
        }
        let start = Line::from_aligned_address(block.start()).next_nth(start);
        let end = Line::from_aligned_address(block.start()).next_nth(end);
        if self.common.needs_log_bit {
            if !copy {
                Line::clear_field_unlog_table::<VM>(start..end);
            } else {
                Line::initialize_field_unlog_table_as_unlogged::<VM>(start..end);
            }
        }
        let num_lines = Line::steps_between(&start, &end).unwrap();
        if !copy {
            self.reused_lines_consumed
                .fetch_add(num_lines, Ordering::Relaxed);
        }
        if self
            .hooks()
            .is_some_and(|h| h.cm_in_progress_or_final_mark())
        {
            Line::initialize_mark_table_as_marked::<VM>(start..end);
            Line::inc_reuse_counts(start..end);
        }
        Some((start, end))
    }

    #[allow(clippy::assertions_on_constants)]
    pub fn normal_get_next_available_lines(&self, search_start: Line) -> Option<(Line, Line)> {
        debug_assert!(!self.rc_enabled);
        let unavail_state = self.line_unavail_state.load(Ordering::Acquire);
        let current_state = self.line_mark_state.load(Ordering::Acquire);
        let block = search_start.block();
        let mark_data = block.line_mark_table();
        let start_cursor = search_start.get_index_within_block();
        let mut cursor = start_cursor;
        // Find start
        while cursor < mark_data.len() {
            let mark = mark_data.get(cursor);
            if mark != unavail_state && mark != current_state {
                break;
            }
            cursor += 1;
        }
        if cursor == mark_data.len() {
            return None;
        }
        let start = search_start.next_nth(cursor - start_cursor);
        // Find limit
        while cursor < mark_data.len() {
            let mark = mark_data.get(cursor);
            if mark == unavail_state || mark == current_state {
                break;
            }
            cursor += 1;
        }
        let end = search_start.next_nth(cursor - start_cursor);
        debug_assert!(RegionIterator::<Line>::new(start, end)
            .all(|line| !line.is_marked(unavail_state) && !line.is_marked(current_state)));
        Some((start, end))
    }

    pub fn is_last_gc_exhaustive(&self, did_defrag_for_last_gc: bool) -> bool {
        if self.is_defrag_enabled() {
            did_defrag_for_last_gc
        } else {
            // If defrag is disabled, every GC is exhaustive.
            true
        }
    }

    pub(crate) fn get_mutator_recycled_lines_in_pages(&self) -> usize {
        debug_assert!(self.rc_enabled);
        self.reused_lines_consumed.load(Ordering::Relaxed)
            >> (LOG_BYTES_IN_PAGE - Line::LOG_BYTES as u8)
    }

    pub(crate) fn get_pages_allocated(&self) -> usize {
        debug_assert!(!self.rc_enabled);
        self.lines_consumed.load(Ordering::Relaxed) >> (LOG_BYTES_IN_PAGE - Line::LOG_BYTES as u8)
    }

    /// Post copy routine for Immix copy contexts
    fn post_copy(&self, object: ObjectReference, _bytes: usize) {
        if self.rc_enabled {
            return;
        }
        // Mark the object
        VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.store_atomic::<VM, u8>(
            object,
            self.mark_state,
            None,
            Ordering::SeqCst,
        );
        self.mark_promoted_for_concurrent_marking(object);
        // Mark the line
        if !super::MARK_LINE_AT_SCAN_TIME {
            self.mark_lines(object);
        }
        if self.common.unlog_traced_object {
            VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                .mark_byte_as_unlogged::<VM>(object, Ordering::Relaxed);
        }
    }

    pub(crate) fn prefer_copy_on_nursery_gc(&self) -> bool {
        self.is_nursery_copy_enabled()
    }

    pub(crate) fn is_nursery_copy_enabled(&self) -> bool {
        !self.space_args.never_move_objects && !cfg!(feature = "sticky_immix_non_moving_nursery")
    }

    pub(crate) fn is_defrag_enabled(&self) -> bool {
        !self.space_args.never_move_objects
    }
}

/// A work packet to prepare each block for a major GC.
/// Performs the action on a range of chunks.
pub struct PrepareBlockState<VM: VMBinding> {
    #[allow(dead_code)]
    pub space: &'static ImmixSpace<VM>,
    pub chunk: Chunk,
    pub defrag_threshold: Option<usize>,
    pub unlog_bits_op: UnlogBitsOperation,
    /// Line epochs `(from, to)`: lines marked `from` are marked `to` instead.
    pub relabel_lines: Option<(u8, u8)>,
}

impl<VM: VMBinding> PrepareBlockState<VM> {
    /// Clear object mark table
    fn reset_object_mark(&self) {
        // NOTE: We reset the mark bits because cyclic mark bit is currently not supported, yet.
        // See `ImmixSpace::prepare`.
        if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_MARK_BIT_SPEC {
            side.bzero_metadata(self.chunk.start(), Chunk::BYTES);
        }
        // A full collection may replace a concurrent mark that had not finished.
        if self.space.uses_concurrent_marking {
            CONC_MARK.bzero_metadata(self.chunk.start(), Chunk::BYTES);
            CONC_COVERED.bzero_metadata(self.chunk.start(), Chunk::BYTES);
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for PrepareBlockState<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        // Clear object mark table for this chunk
        self.reset_object_mark();
        // The number of defrag source blocks.  For debugging.
        let mut num_defrag_source_blocks = 0;
        let mut num_blocks_prepared = 0;
        // Iterate over all blocks in this chunk
        for block in self.chunk.iter_region::<Block>() {
            let state = block.get_state();
            // Skip unallocated blocks.
            if state == BlockState::Unallocated {
                continue;
            }
            num_blocks_prepared += 1;
            // Check if this block needs to be defragmented.
            let is_defrag_source = if !self.space.is_defrag_enabled() {
                // Do not set any block as defrag source if defrag is disabled.
                false
            } else if *mmtk.options.immix_defrag_every_block {
                // Set every block as defrag source if so desired.
                true
            } else if let Some(defrag_threshold) = self.defrag_threshold {
                // This GC is a defrag GC.
                block.get_holes() > defrag_threshold
            } else {
                // Not a defrag GC.
                false
            };
            if is_defrag_source {
                num_defrag_source_blocks += 1;
            }
            block.set_as_defrag_source(is_defrag_source);
            // Clear block mark data.
            block.set_state(BlockState::Unmarked);
            debug_assert!(!block.get_state().is_reusable());
            debug_assert_ne!(block.get_state(), BlockState::Marked);
            if let Some((from, to)) = self.relabel_lines {
                for line in block.lines().filter(|line| line.is_marked(from)) {
                    line.mark(to);
                }
            }
        }

        probe!(
            mmtk,
            immix_prepare_block_state,
            num_blocks_prepared,
            num_defrag_source_blocks
        );

        self.unlog_bits_op
            .execute::<VM>(self.chunk.start(), Chunk::BYTES);
    }
}

/// Sweeps blocks allocated since the last GC, in a nursery GC of a mixed-age space. It does not
/// update `defrag.mark_histograms`, which only a defrag GC reads and which it counts again, or
/// release empty chunks, which the next full sweep does.
struct SweepBlocks<VM: VMBinding> {
    space: &'static ImmixSpace<VM>,
    blocks: Vec<Block>,
    /// A destructor invoked when all `SweepBlocks` packets are finished.
    epilogue: Arc<FlushPageResource<VM>>,
}

impl<VM: VMBinding> GCWork<VM> for SweepBlocks<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let mut histogram = self.space.defrag.new_histogram();
        let line_mark_state = if super::BLOCK_ONLY {
            None
        } else {
            Some(self.space.line_mark_state.load(Ordering::Acquire))
        };
        let is_moving_gc = mmtk.get_plan().current_gc_may_move_object();
        for &block in &self.blocks {
            // Only these blocks hold young objects, so only they can hold forwarded ones.
            if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC {
                if is_moving_gc {
                    side.bzero_metadata(block.start(), Block::BYTES);
                }
            }
            block.sweep(self.space, &mut histogram, line_mark_state, false);
        }
        self.epilogue.finish_one_work_packet();
    }
}

/// Sweeps the blocks a nursery collection left to [`ImmixSpace::sweep_nursery_chunks`], once the world has resumed.
struct SweepNurseryBlocks<VM: VMBinding> {
    space: &'static ImmixSpace<VM>,
}

impl<VM: VMBinding> GCWork<VM> for SweepNurseryBlocks<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        self.space.sweep_nursery_chunks();
    }
}

/// What [`ImmixSpace::release_nursery_later`] leaves to sweep.
#[derive(Default)]
struct NurserySweep {
    chunks: Vec<Vec<Block>>,
    line_mark_state: Option<u8>,
    clear_forwarding_bits: bool,
}

/// Sweeps blocks after a concurrent full-heap mark, beside the program. The last packet makes the blocks it and the others freed or found reusable available to allocators; until then no allocator can take any of these blocks. A full-heap collection may drop the packets still waiting.
struct SweepBlocksAfterMark<VM: VMBinding> {
    space: &'static ImmixSpace<VM>,
    blocks: Vec<Block>,
}

impl<VM: VMBinding> GCWork<VM> for SweepBlocksAfterMark<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        let mut histogram = self.space.defrag.new_histogram();
        let line_mark_state = Some(self.space.line_mark_state.load(Ordering::Acquire));
        for &block in &self.blocks {
            if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC {
                side.bzero_metadata(block.start(), Block::BYTES);
            }
            block.sweep(self.space, &mut histogram, line_mark_state, true);
        }
        self.space.defrag.add_completed_mark_histogram(histogram);
        if self.space.sweep_packets_left.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.space.flush_page_resource();
        }
    }
}

/// Chunk sweeping work packet.
struct SweepChunk<VM: VMBinding> {
    space: &'static ImmixSpace<VM>,
    chunk: Chunk,
    unlog_bits_op: UnlogBitsOperation,
    /// Whether this is the final sweep of a concurrent full-heap mark.
    after_concurrent_mark: bool,
    /// A destructor invoked when all `SweepChunk` packets are finished.
    epilogue: Arc<FlushPageResource<VM>>,
}

impl<VM: VMBinding> GCWork<VM> for SweepChunk<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        assert!(self.space.chunk_map.get(self.chunk).unwrap().is_allocated());

        let mut histogram = self.space.defrag.new_histogram();
        let line_mark_state = if super::BLOCK_ONLY {
            None
        } else {
            Some(self.space.line_mark_state.load(Ordering::Acquire))
        };
        // Hints for clearing side forwarding bits.
        let is_moving_gc = mmtk.get_plan().current_gc_may_move_object();
        let is_defrag_gc = self.space.defrag.in_defrag();

        // number of swept (completely free) blocks.
        let mut swept_blocks = 0;
        // number of reused blocks.
        let mut reused_blocks = 0;
        // number of non-free blocks that cannot be reused (e.g. full, or non-empty when block-only).
        let mut unreused_blocks = 0;

        // Iterate over all allocated blocks in this chunk.
        for block in self
            .chunk
            .iter_region::<Block>()
            .filter(|block| block.get_state() != BlockState::Unallocated)
        {
            // Clear side forwarding bits.
            // In the beginning of the next GC, no side forwarding bits shall be set.
            // In this way, we can omit clearing forwarding bits when copying object.
            // See `GCWorkerCopyContext::post_copy`.
            // Note, `block.sweep()` overwrites `DEFRAG_STATE_TABLE` with the number of holes,
            // but we need it to know if a block is a defrag source.
            // We clear forwarding bits before `block.sweep()`.
            if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC {
                if is_moving_gc {
                    let objects_may_move = if is_defrag_gc {
                        // If it is a defrag GC, we only clear forwarding bits for defrag sources.
                        block.is_defrag_source()
                    } else {
                        // Otherwise, it must be a nursery GC of StickyImmix with copying nursery.
                        // We don't have information about which block contains moved objects,
                        // so we have to clear forwarding bits for all blocks.
                        true
                    };
                    if objects_may_move {
                        side.bzero_metadata(block.start(), Block::BYTES);
                    }
                }
            }

            match block.sweep(self.space, &mut histogram, line_mark_state, self.after_concurrent_mark) {
                BlockSweepResult::Swept => swept_blocks += 1,
                BlockSweepResult::Reused => reused_blocks += 1,
                BlockSweepResult::NoReuse => unreused_blocks += 1,
            }
        }

        probe!(
            mmtk,
            sweep_chunk_immix,
            swept_blocks,
            reused_blocks,
            unreused_blocks
        );

        // number of allocated blocks.
        let allocated_blocks = reused_blocks + unreused_blocks;

        // Set this chunk as free if there is not live blocks.
        if allocated_blocks == 0 {
            self.space.chunk_map.set_allocated(self.chunk, false)
        }
        self.space.defrag.add_completed_mark_histogram(histogram);

        self.unlog_bits_op
            .execute::<VM>(self.chunk.start(), Chunk::BYTES);

        self.epilogue.finish_one_work_packet();
    }
}

/// Count number of remaining work pacets, and flush page resource if all packets are finished.
struct FlushPageResource<VM: VMBinding> {
    space: &'static ImmixSpace<VM>,
    counter: AtomicUsize,
}

impl<VM: VMBinding> FlushPageResource<VM> {
    /// Called after a related work packet is finished.
    fn finish_one_work_packet(&self) {
        if 1 == self.counter.fetch_sub(1, Ordering::SeqCst) {
            // We've finished releasing all the dead blocks to the BlockPageResource's thread-local queues.
            // Now flush the BlockPageResource.
            self.space.flush_page_resource()
        }
    }
}

impl<VM: VMBinding> Drop for FlushPageResource<VM> {
    fn drop(&mut self) {
        epilogue::debug_assert_counter_zero(&self.counter, "FlushPageResource::counter");
    }
}

use crate::policy::copy_context::PolicyCopyContext;
use crate::util::alloc::Allocator;
use crate::util::alloc::ImmixAllocator;

/// Normal immix copy context. It has one copying Immix allocator.
/// Most immix plans use this copy context.
pub struct ImmixCopyContext<VM: VMBinding> {
    allocator: ImmixAllocator<VM>,
}

impl<VM: VMBinding> PolicyCopyContext for ImmixCopyContext<VM> {
    type VM = VM;

    fn prepare(&mut self) {
        self.allocator.reset();
    }
    fn release(&mut self) {
        self.allocator.reset();
    }
    fn alloc_copy(
        &mut self,
        _original: ObjectReference,
        bytes: usize,
        align: usize,
        offset: usize,
    ) -> Address {
        self.allocator.alloc(bytes, align, offset)
    }
    fn post_copy(&mut self, obj: ObjectReference, bytes: usize) {
        self.get_space().post_copy(obj, bytes)
    }
}

impl<VM: VMBinding> ImmixCopyContext<VM> {
    pub(crate) fn new(
        tls: VMWorkerThread,
        context: Arc<AllocatorContext<VM>>,
        space: &'static ImmixSpace<VM>,
    ) -> Self {
        ImmixCopyContext {
            allocator: ImmixAllocator::new(tls.0, Some(space), context, true),
        }
    }

    fn get_space(&self) -> &ImmixSpace<VM> {
        self.allocator.immix_space()
    }
}

/// Hybrid Immix copy context. It includes two different immix allocators. One with `copy = true`
/// is used for defrag GCs, and the other is used for other purposes (such as promoting objects from
/// nursery to Immix mature space). This is used by generational immix.
pub struct ImmixHybridCopyContext<VM: VMBinding> {
    copy_allocator: ImmixAllocator<VM>,
    defrag_allocator: ImmixAllocator<VM>,
}

impl<VM: VMBinding> PolicyCopyContext for ImmixHybridCopyContext<VM> {
    type VM = VM;

    fn prepare(&mut self) {
        self.copy_allocator.reset();
        self.defrag_allocator.reset();
    }
    fn release(&mut self) {
        self.copy_allocator.reset();
        self.defrag_allocator.reset();
    }
    fn alloc_copy(
        &mut self,
        _original: ObjectReference,
        bytes: usize,
        align: usize,
        offset: usize,
    ) -> Address {
        if self.get_space().in_defrag() {
            self.defrag_allocator.alloc(bytes, align, offset)
        } else {
            self.copy_allocator.alloc(bytes, align, offset)
        }
    }
    fn post_copy(&mut self, obj: ObjectReference, bytes: usize) {
        self.get_space().post_copy(obj, bytes)
    }
}

impl<VM: VMBinding> ImmixHybridCopyContext<VM> {
    pub(crate) fn new(
        tls: VMWorkerThread,
        context: Arc<AllocatorContext<VM>>,
        space: &'static ImmixSpace<VM>,
    ) -> Self {
        ImmixHybridCopyContext {
            copy_allocator: ImmixAllocator::new(tls.0, Some(space), context.clone(), true),
            defrag_allocator: ImmixAllocator::new(tls.0, Some(space), context, true),
        }
    }

    fn get_space(&self) -> &ImmixSpace<VM> {
        // Both copy allocators should point to the same space.
        debug_assert_eq!(
            self.defrag_allocator.immix_space().common().descriptor,
            self.copy_allocator.immix_space().common().descriptor
        );
        // Just get the space from either allocator
        self.defrag_allocator.immix_space()
    }
}

#[cfg(feature = "vo_bit")]
#[derive(Clone, Copy)]
enum VOBitsClearingScope {
    /// Clear all VO bits in all blocks.
    FullGC,
    /// Clear unmarked blocks, only.
    BlockOnly,
    /// Clear unmarked lines, only.  (i.e. lines with line mark state **not** equal to `state`).
    Line { state: u8 },
}

/// A work packet to clear VO bit metadata after Prepare.
#[cfg(feature = "vo_bit")]
struct ClearVOBitsAfterPrepare {
    chunk: Chunk,
    scope: VOBitsClearingScope,
}

#[cfg(feature = "vo_bit")]
impl<VM: VMBinding> GCWork<VM> for ClearVOBitsAfterPrepare {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        match self.scope {
            VOBitsClearingScope::FullGC => {
                vo_bit::bzero_vo_bit(self.chunk.start(), Chunk::BYTES);
            }
            VOBitsClearingScope::BlockOnly => {
                self.clear_blocks(None);
            }
            VOBitsClearingScope::Line { state } => {
                self.clear_blocks(Some(state));
            }
        }
    }
}

#[cfg(feature = "vo_bit")]
impl ClearVOBitsAfterPrepare {
    fn clear_blocks(&mut self, line_mark_state: Option<u8>) {
        for block in self
            .chunk
            .iter_region::<Block>()
            .filter(|block| block.get_state() != BlockState::Unallocated)
        {
            block.clear_vo_bits_for_unmarked_regions(line_mark_state);
        }
    }
}
