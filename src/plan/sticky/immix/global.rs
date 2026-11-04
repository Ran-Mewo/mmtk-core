use crate::plan::concurrent::ConcurrentPlan;
use crate::plan::generational::global::GenerationalPlan;
use crate::plan::global::CommonPlan;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::immix;
use crate::plan::Pause;
use crate::plan::PlanConstraints;
use crate::policy::gc_work::TraceKind;
use crate::policy::gc_work::TRACE_KIND_TRANSITIVE_PIN;
use crate::policy::immix::defrag::StatsForDefrag;
use crate::policy::immix::ImmixSpace;
use crate::policy::immix::TRACE_KIND_FAST;
use crate::policy::sft::SFT;
use crate::policy::space::Space;
use crate::scheduler::GCWork;
use crate::scheduler::WorkBucketStage;
use crate::util::copy::CopyConfig;
use crate::util::copy::CopySelector;
use crate::util::copy::CopySemantics;
use crate::util::heap::gc_trigger::SpaceStats;
use crate::util::metadata::log_bit::UnlogBitsOperation;
use crate::util::metadata::side_metadata::spec_defs::{CONC_COVERED, CONC_MARK};
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::statistics::counter::EventCounter;
use crate::util::ObjectReference;
use crate::vm::ObjectModel;
use crate::vm::VMBinding;
use crate::Plan;
use crate::MMTK;

use atomic::{Atomic, Ordering};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex};

use mmtk_macros::{HasSpaces, PlanTraceObject};

use super::gc_work::StickyImmixInitialMarkGCWorkContext;
use super::gc_work::StickyImmixMatureGCWorkContext;
use super::gc_work::StickyImmixNurseryGCWorkContext;

#[derive(HasSpaces, PlanTraceObject)]
pub struct StickyImmix<VM: VMBinding> {
    #[parent]
    immix: immix::Immix<VM>,
    gc_full_heap: AtomicBool,
    next_gc_full_heap: AtomicBool,
    full_heap_gc_count: Arc<Mutex<EventCounter>>,
    /// The current pause, or `None` between pauses. Nursery and full-heap pauses are both `Pause::Full`, told apart by `gc_full_heap`; the initial-mark and final-mark pauses are nursery pauses that also start or finish a concurrent full-heap mark.
    pause: Atomic<Option<Pause>>,
    /// Whether a concurrent full-heap mark is running, from the end of its initial-mark pause to the end of its final-mark pause.
    concurrent_marking: AtomicBool,
    /// Whether the marker has run out of work since the mark started, so the next pause can finish it.
    marking_drained: AtomicBool,
    /// Whether the next nursery pause starts a concurrent mark.
    start_marking: AtomicBool,
    /// Pages in use at the end of the last full-heap pause or final-mark pause, about what was live then.
    pages_after_full_heap_trace: AtomicUsize,
    /// Whether `pages_after_full_heap_trace` waits for the sweep after a final-mark pause.
    measure_live_pages: AtomicBool,
    /// Whether the current final-mark pause is for memory running short, so it sweeps before the world resumes.
    final_mark_for_memory: AtomicBool,
}

/// The plan constraints for the sticky immix plan.
pub const STICKY_IMMIX_CONSTRAINTS: PlanConstraints = PlanConstraints {
    // If we disable moving in Immix, this is a non-moving plan.
    moves_objects: !cfg!(feature = "immix_non_moving"),
    needs_log_bit: true,
    barrier: crate::plan::BarrierSelector::ObjectBarrier,
    // We may trace duplicate edges in sticky immix (or any plan that uses object remembering barrier). See https://github.com/mmtk/mmtk-core/issues/743.
    may_trace_duplicate_edges: true,
    generational: true,
    ..immix::IMMIX_CONSTRAINTS
};

impl<VM: VMBinding> Plan for StickyImmix<VM> {
    fn constraints(&self) -> &'static crate::plan::PlanConstraints {
        &STICKY_IMMIX_CONSTRAINTS
    }

    fn create_copy_config(&'static self) -> CopyConfig<Self::VM> {
        use enum_map::enum_map;
        CopyConfig {
            copy_mapping: enum_map! {
                CopySemantics::DefaultCopy => CopySelector::Immix(0),
                _ => CopySelector::Unused,
            },
            space_mapping: vec![(CopySelector::Immix(0), &self.immix.immix_space)],
            constraints: &STICKY_IMMIX_CONSTRAINTS,
        }
    }

    fn base(&self) -> &crate::plan::global::BasePlan<Self::VM> {
        self.immix.base()
    }

    fn base_mut(&mut self) -> &mut crate::plan::global::BasePlan<Self::VM> {
        self.immix.base_mut()
    }

    fn generational(
        &self,
    ) -> Option<&dyn crate::plan::generational::global::GenerationalPlan<VM = Self::VM>> {
        Some(self)
    }

    fn concurrent(&self) -> Option<&dyn ConcurrentPlan<VM = Self::VM>> {
        if self.uses_concurrent_marking() {
            Some(self)
        } else {
            None
        }
    }

    fn common(&self) -> &CommonPlan<Self::VM> {
        self.immix.common()
    }

    fn schedule_collection(&'static self, scheduler: &crate::scheduler::GCWorkScheduler<Self::VM>) {
        let marking = self.concurrent_marking.load(Ordering::SeqCst);
        let is_full_heap = self.requires_full_heap_collection(marking || self.immix.immix_space.is_sweeping_after_mark());
        self.gc_full_heap.store(is_full_heap, Ordering::SeqCst);
        probe!(mmtk, gen_full_heap, is_full_heap);

        // Running low on memory finishes a running mark, which frees the old space as a full-heap pause would, without tracing it all again.
        let pause = if is_full_heap {
            Pause::Full
        } else if marking {
            let for_memory = self.next_gc_full_heap.load(Ordering::SeqCst) || self.collection_attempts() > 1;
            self.final_mark_for_memory.store(for_memory, Ordering::SeqCst);
            if self.marking_drained.load(Ordering::SeqCst) || for_memory {
                Pause::FinalMark
            } else {
                Pause::Full
            }
        } else if self.start_marking.load(Ordering::SeqCst) {
            Pause::InitialMark
        } else {
            Pause::Full
        };
        self.pause.store(Some(pause), Ordering::SeqCst);

        if is_full_heap {
            info!("Full heap GC");
            use crate::plan::immix::Immix;
            use crate::policy::immix::TRACE_KIND_DEFRAG;
            Immix::schedule_immix_full_heap_collection::<
                StickyImmix<VM>,
                StickyImmixMatureGCWorkContext<VM, TRACE_KIND_FAST>,
                StickyImmixMatureGCWorkContext<VM, TRACE_KIND_DEFRAG>,
            >(self, &self.immix.immix_space, scheduler);
        } else if pause == Pause::InitialMark {
            info!("Nursery GC, starting a concurrent mark");
            scheduler.schedule_common_work::<StickyImmixInitialMarkGCWorkContext<VM>>(self);
        } else {
            info!("Nursery GC{}", if pause == Pause::FinalMark { ", finishing the concurrent mark" } else { "" });
            scheduler.schedule_common_work::<StickyImmixNurseryGCWorkContext<VM>>(self);
        }
    }

    fn get_allocator_mapping(
        &self,
    ) -> &'static enum_map::EnumMap<crate::AllocationSemantics, crate::util::alloc::AllocatorSelector>
    {
        &super::mutator::ALLOCATOR_MAPPING
    }

    fn prepare(&mut self, tls: crate::util::VMWorkerThread) {
        if self.is_current_gc_nursery() {
            // Prepare both large object space and immix space
            self.immix.immix_space.prepare(
                false,
                Some(StatsForDefrag::new(self)),
                // We don't do anything special to unlog bits during nursery GC
                // because ProcessModBuf will set the unlog bits back.
                UnlogBitsOperation::NoOp,
            );
            self.immix.common.los.prepare(false);
        } else {
            self.full_heap_gc_count.lock().unwrap().inc();
            self.immix.prepare_inner(
                tls,
                // We will reconstruct unlog bits during tracing.
                UnlogBitsOperation::BulkClear,
            );
        }
    }

    fn release(&mut self, tls: crate::util::VMWorkerThread) {
        if self.is_current_gc_nursery() {
            self.immix.common.los.release(false);
            if self.pause.load(Ordering::SeqCst) == Some(Pause::FinalMark) {
                self.immix.common.los.release_after_concurrent_marking();
                // A final-mark pause that memory ran short for sweeps now, since the allocation waiting on it needs what the sweep frees; otherwise the sweep follows the pause.
                if self.final_mark_for_memory.load(Ordering::SeqCst) {
                    self.immix.immix_space.release_after_concurrent_marking();
                } else {
                    let packets = self.immix.immix_space.release_after_concurrent_marking_later();
                    let bucket = &self.immix.common.base.scheduler.work_buckets[WorkBucketStage::ConcurrentResumable];
                    for packet in packets {
                        bucket.add_boxed_no_notify(packet);
                    }
                }
                // Weak references are processed by now, and the sanity trace that may follow expects the old bits alone to mean live.
                self.end_concurrent_marking(false);
            } else {
                self.immix.immix_space.release(
                    false,
                    // We don't do anything special to unlog bits during nursery GC
                    // because ProcessModBuf has set the unlog bits back.
                    UnlogBitsOperation::NoOp,
                );
            }
        } else {
            self.immix.release_inner(
                tls,
                // We reconstructred unlog bits during tracing.  Keep them.
                UnlogBitsOperation::NoOp,
            );
        }
    }

    fn on_pause_start(&self, mmtk: &'static MMTK<VM>) {
        let scheduler = &self.immix.common.base.scheduler;
        if self.pause.load(Ordering::SeqCst) == Some(Pause::FinalMark) {
            // What the marker left, and what the barriers recorded since, is traced first in this pause; the world is stopped, so nothing more can be recorded.
            let left = scheduler.work_buckets[WorkBucketStage::ConcurrentResumable].drain_all_packets();
            scheduler.work_buckets[WorkBucketStage::FinishConcurrentWork].bulk_add(left);
            self.immix.immix_space.start_final_mark();
            self.immix.common.los.start_final_mark();
        } else if self.gc_full_heap.load(Ordering::SeqCst) && (self.concurrent_marking.load(Ordering::SeqCst) || self.immix.immix_space.is_sweeping_after_mark()) {
            // A full-heap pause replaces the mark or the sweep after it; its prepare clears the concurrent mark bits, and its own sweep covers every block.
            drop(scheduler.work_buckets[WorkBucketStage::ConcurrentResumable].drain_all_packets());
            self.immix.immix_space.abandon_sweep_after_mark();
            if self.concurrent_marking.load(Ordering::SeqCst) {
                self.end_concurrent_marking(true);
            }
        } else if self.immix.immix_space.is_sweeping_after_mark() && (self.next_gc_full_heap.load(Ordering::SeqCst) || self.collection_attempts() > 1) {
            // Memory runs short before the sweep after a mark is done, so this pause finishes it first.
            let left = scheduler.work_buckets[WorkBucketStage::ConcurrentResumable].drain_all_packets();
            scheduler.work_buckets[WorkBucketStage::FinishConcurrentWork].bulk_add(left);
        }
        mmtk.gc_trigger.policy.on_gc_start(mmtk);
    }

    fn on_pause_end(
        &mut self,
        mmtk: &'static MMTK<VM>,
        tls: crate::util::opaque_pointer::VMWorkerThread,
    ) {
        let next_gc_full_heap =
            crate::plan::generational::global::CommonGenPlan::should_next_gc_be_full_heap(self);
        self.next_gc_full_heap
            .store(next_gc_full_heap, Ordering::Relaxed);

        let was_defrag = self.immix.immix_space.end_of_gc();
        self.immix
            .set_last_gc_was_defrag(was_defrag, Ordering::Relaxed);

        self.immix.common.on_pause_end(tls);

        let pause = self.pause.swap(None, Ordering::SeqCst);
        // What a full-heap trace left is known once its sweep is done, which may be pauses later.
        if self.gc_full_heap.load(Ordering::SeqCst) || pause == Some(Pause::FinalMark) {
            self.measure_live_pages.store(true, Ordering::Relaxed);
        }
        if self.measure_live_pages.load(Ordering::Relaxed) && !self.immix.immix_space.is_sweeping_after_mark() {
            self.pages_after_full_heap_trace.store(self.get_reserved_pages(), Ordering::Relaxed);
            self.measure_live_pages.store(false, Ordering::Relaxed);
        }
        if pause == Some(Pause::InitialMark) {
            self.immix.immix_space.start_concurrent_marking();
            self.immix.common.los.start_concurrent_marking();
            self.marking_drained.store(false, Ordering::SeqCst);
            self.concurrent_marking.store(true, Ordering::SeqCst);
        } else if self.concurrent_marking.load(Ordering::SeqCst) && mmtk.scheduler.work_buckets[WorkBucketStage::ConcurrentResumable].is_empty() {
            self.marking_drained.store(true, Ordering::SeqCst);
        }
        self.start_marking.store(!self.concurrent_marking.load(Ordering::SeqCst) && self.should_start_marking(), Ordering::SeqCst);

        mmtk.gc_trigger.policy.on_gc_end(mmtk);
    }

    fn collection_required(&self, space_full: bool, space: Option<SpaceStats<Self::VM>>) -> bool {
        let nursery_full = self.immix.immix_space.get_pages_allocated()
            > self.base().gc_trigger.get_max_nursery_pages();
        if space_full
            && space.is_some()
            && space.as_ref().unwrap().0.name() != self.immix.immix_space.name()
        {
            self.next_gc_full_heap.store(true, Ordering::SeqCst);
        }
        self.immix.collection_required(space_full, space) || nursery_full
    }

    fn last_collection_was_exhaustive(&self) -> bool {
        self.gc_full_heap.load(Ordering::Relaxed) && self.immix.last_collection_was_exhaustive()
    }

    fn current_gc_may_move_object(&self) -> bool {
        if self.is_current_gc_nursery() {
            self.get_immix_space().prefer_copy_on_nursery_gc()
        } else {
            self.get_immix_space().in_defrag()
        }
    }

    fn get_collection_reserved_pages(&self) -> usize {
        self.immix.get_collection_reserved_pages()
    }

    fn get_used_pages(&self) -> usize {
        self.immix.get_used_pages()
    }

    fn sanity_check_object(&self, object: crate::util::ObjectReference) -> bool {
        if self.is_current_gc_nursery() {
            // Every reachable object should be logged
            if !VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.is_unlogged::<VM>(object, Ordering::SeqCst) {
                error!("Object {} is not unlogged (all objects that have been traced should be unlogged/mature)", object);
                return false;
            }

            // Every reachable object should be marked
            if self.immix.immix_space.in_space(object) && !self.immix.immix_space.is_marked(object)
            {
                error!(
                    "Object {} is not marked (all objects that have been traced should be marked)",
                    object
                );
                return false;
            } else if self.immix.common.los.in_space(object)
                && !self.immix.common.los.is_live(object)
            {
                error!("LOS Object {} is not marked", object);
                return false;
            }
        }
        true
    }
}

impl<VM: VMBinding> GenerationalPlan for StickyImmix<VM> {
    fn is_current_gc_nursery(&self) -> bool {
        !self.gc_full_heap.load(Ordering::SeqCst)
    }

    fn is_object_in_nursery(&self, object: crate::util::ObjectReference) -> bool {
        self.immix.immix_space.in_space(object) && !self.immix.immix_space.is_marked(object)
    }

    // This check is used for memory slice copying barrier, where we only know addresses instead of objects.
    // As sticky immix needs object metadata to know if an object is an nursery object or not, we cannot really tell
    // whether an address is in nursery or not. In this case, we just return false -- this is a conservative return value
    // for the memory slice copying barrier. It means we will treat the object as if it is in mature space, and will
    // push it to the remembered set.
    // FIXME: this will remember excessive objects, and can cause serious slowdown in some cases.
    fn is_address_in_nursery(&self, _addr: crate::util::Address) -> bool {
        false
    }

    fn get_mature_physical_pages_available(&self) -> usize {
        self.immix.immix_space.available_physical_pages()
    }

    fn get_mature_reserved_pages(&self) -> usize {
        self.immix.immix_space.reserved_pages()
    }

    fn force_full_heap_collection(&self) {
        self.next_gc_full_heap.store(true, Ordering::SeqCst);
    }

    fn last_collection_full_heap(&self) -> bool {
        self.gc_full_heap.load(Ordering::SeqCst)
    }
}

impl<VM: VMBinding> crate::plan::generational::global::GenerationalPlanExt<VM> for StickyImmix<VM> {
    fn trace_object_nursery<Q: crate::ObjectQueue, const KIND: TraceKind>(
        &self,
        queue: &mut Q,
        object: crate::util::ObjectReference,
        worker: &mut crate::scheduler::GCWorker<VM>,
    ) -> crate::util::ObjectReference {
        if self.immix.immix_space.in_space(object) {
            if !self.is_object_in_nursery(object) {
                // Mature object
                trace!("Immix mature object {}, skip", object);
                return object;
            } else {
                // Nursery object
                let object = if KIND == TRACE_KIND_TRANSITIVE_PIN || KIND == TRACE_KIND_FAST {
                    trace!(
                        "Immix nursery object {} is being traced without moving",
                        object
                    );
                    self.immix
                        .immix_space
                        .trace_object_without_moving(queue, object)
                } else if self.immix.immix_space.prefer_copy_on_nursery_gc() {
                    let ret = self.immix.immix_space.trace_object_with_opportunistic_copy(
                        queue,
                        object,
                        // We just use default copy here. We have set args for ImmixSpace to deal with unlogged bit,
                        // and we do not need to use CopySemantics::PromoteToMature.
                        CopySemantics::DefaultCopy,
                        worker,
                        true,
                    );
                    trace!(
                        "Immix nursery object {} is being traced with opportunistic copy {}",
                        object,
                        if ret == object {
                            "".to_string()
                        } else {
                            format!("-> new object {}", ret)
                        }
                    );
                    ret
                } else {
                    trace!(
                        "Immix nursery object {} is being traced without moving",
                        object
                    );
                    self.immix
                        .immix_space
                        .trace_object_without_moving(queue, object)
                };

                return object;
            }
        }

        if self.immix.common().get_los().in_space(object) {
            return self
                .immix
                .common()
                .get_los()
                .trace_object::<Q>(queue, object);
        }

        object
    }
}

impl<VM: VMBinding> StickyImmix<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let full_heap_gc_count = args.stats.new_event_counter("majorGC", true, true);
        let concurrent_marking = *args.options.sticky_immix_concurrent_marking;
        let mut global_specs = crate::plan::generational::new_generational_global_metadata_specs::<VM>();
        if concurrent_marking {
            global_specs.push(CONC_MARK);
            global_specs.push(CONC_COVERED);
        }
        let plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &STICKY_IMMIX_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&global_specs),
        };

        let mut immix = immix::Immix::new_with_args(
            plan_args,
            crate::policy::immix::ImmixSpaceArgs {
                // In StickyImmix, both young and old objects are allocated in the ImmixSpace.
                mixed_age: true,
                never_move_objects: false,
            },
        );
        if concurrent_marking {
            immix.immix_space.use_concurrent_marking();
            immix.common.los.use_concurrent_marking();
        }
        Self {
            immix,
            gc_full_heap: AtomicBool::new(false),
            next_gc_full_heap: AtomicBool::new(false),
            full_heap_gc_count,
            pause: Atomic::new(None),
            concurrent_marking: AtomicBool::new(false),
            marking_drained: AtomicBool::new(false),
            start_marking: AtomicBool::new(false),
            pages_after_full_heap_trace: AtomicUsize::new(0),
            measure_live_pages: AtomicBool::new(false),
            final_mark_for_memory: AtomicBool::new(false),
        }
    }

    // While a mark or the sweep after it runs, the first nursery pause that leaves too little memory is followed by one that finishes it, and only if that frees too little too, by a full-heap pause.
    fn requires_full_heap_collection(&self, old_space_work_pending: bool) -> bool {
        // Separate each condition so the code is clear
        #[allow(clippy::if_same_then_else, clippy::needless_bool)]
        if crate::plan::generational::FULL_NURSERY_GC {
            trace!("full heap: forced full heap");
            // For barrier overhead measurements, we always do full gc in nursery collections.
            true
        } else if self
            .immix
            .common
            .base
            .global_state
            .user_triggered_collection
            .load(Ordering::SeqCst)
            && *self.immix.common.base.options.full_heap_system_gc
        {
            // User triggered collection, and we force full heap for user triggered collection
            true
        } else if (!old_space_work_pending && self.next_gc_full_heap.load(Ordering::SeqCst)) || self.collection_attempts() > 1 + old_space_work_pending as usize {
            // Forces full heap collection
            true
        } else {
            false
        }
    }

    fn collection_attempts(&self) -> usize {
        self.immix.common.base.global_state.cur_collection_attempts.load(Ordering::SeqCst)
    }

    pub fn get_immix_space(&self) -> &ImmixSpace<VM> {
        &self.immix.immix_space
    }

    fn uses_concurrent_marking(&self) -> bool {
        *self.immix.common.base.options.sticky_immix_concurrent_marking
    }

    // A mark starts once the given share of the room the last full-heap trace left free is in use, and not before the sweep after the last mark is done.
    fn should_start_marking(&self) -> bool {
        if !self.uses_concurrent_marking() || self.immix.immix_space.is_sweeping_after_mark() {
            return false;
        }
        let live = self.pages_after_full_heap_trace.load(Ordering::Relaxed);
        let room = self.get_total_pages().saturating_sub(live);
        let trigger = live + room * *self.immix.common.base.options.sticky_immix_concurrent_mark_trigger / 100;
        self.get_reserved_pages() > trigger
    }

    fn end_concurrent_marking(&self, replaced_by_full_heap_collection: bool) {
        self.concurrent_marking.store(false, Ordering::SeqCst);
        self.immix.immix_space.end_concurrent_marking(replaced_by_full_heap_collection);
        self.immix.common.los.end_concurrent_marking();
        // The spaces clear their objects' covered bits as they sweep; the VM space is never swept.
        #[cfg(feature = "vm_space")]
        self.immix.common.base.vm_space.bzero_side_metadata(&CONC_COVERED);
    }

    /// Whether a concurrent full-heap mark is running, between its initial-mark and final-mark pauses.
    pub(super) fn is_concurrent_marking(&self) -> bool {
        self.concurrent_marking.load(Ordering::Acquire)
    }

    /// Mark an old object for the concurrent mark. Returns whether this call marked it, so it is to be scanned.
    pub(super) fn attempt_concurrent_mark(&self, object: ObjectReference) -> bool {
        if self.immix.immix_space.in_space(object) {
            self.immix.immix_space.attempt_concurrent_mark(object)
        } else if self.immix.common.los.in_space(object) {
            self.immix.common.los.attempt_concurrent_mark(object)
        } else {
            false
        }
    }

    /// Whether `object` is in a space this plan collects, rather than in the VM space or another space it never frees.
    pub(super) fn is_in_collected_space(&self, object: ObjectReference) -> bool {
        self.immix.immix_space.in_space(object) || self.immix.common.los.in_space(object)
    }

    /// Whether the concurrent mark has yet to reach `object`: an old object it has not marked.
    pub(super) fn is_unmarked_old_object(&self, object: ObjectReference) -> bool {
        let immix = &self.immix.immix_space;
        let los = &self.immix.common.los;
        if immix.in_space(object) {
            immix.is_marked(object) && !immix.is_concurrent_marked(object)
        } else if los.in_space(object) {
            los.is_marked(object) && !los.is_concurrent_marked(object)
        } else {
            false
        }
    }

    /// Queue work for the concurrent mark. Outside the final-mark pause it waits in `ConcurrentResumable`, which nursery pauses leave alone; inside it, it is done before the rest of the pause.
    pub(super) fn add_marking_work(&self, work: impl GCWork<VM>) {
        let buckets = &self.immix.common.base.scheduler.work_buckets;
        match self.pause.load(Ordering::SeqCst) {
            Some(Pause::FinalMark) => buckets[WorkBucketStage::FinishConcurrentWork].add(work),
            Some(_) => buckets[WorkBucketStage::ConcurrentResumable].add_no_notify(work),
            None => buckets[WorkBucketStage::ConcurrentResumable].add(work),
        }
    }
}

impl<VM: VMBinding> ConcurrentPlan for StickyImmix<VM> {
    fn concurrent_work_in_progress(&self) -> bool {
        self.is_concurrent_marking()
    }

    fn current_pause(&self) -> Option<Pause> {
        self.pause.load(Ordering::SeqCst)
    }

    // The marking packets yield on their own once they see the request; disabling their bucket stops new ones from starting, so the workers park and the pause can begin.
    fn on_concurrent_work_interrupted(&self) {
        self.immix.common.base.scheduler.work_buckets[WorkBucketStage::ConcurrentResumable].set_enabled(false);
    }

    // The status stays `InConcurrentGC` until the final-mark pause, so a pause requested meanwhile still disables the marker's bucket: barriers keep adding work to it, and workers that kept polling it would never park.
    fn on_concurrent_work_drained(&self) {
        self.marking_drained.store(true, Ordering::SeqCst);
    }
}
