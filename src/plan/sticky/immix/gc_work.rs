use crate::plan::generational::gc_work::GenNurseryTrace;
use crate::plan::tracing::PlanTrace;
use crate::policy::gc_work::TraceKind;
use crate::policy::gc_work::DEFAULT_TRACE;
use crate::policy::gc_work::TRACE_KIND_TRANSITIVE_PIN;
use crate::vm::{RootsWorkFactory, VMBinding};
use crate::MMTK;

use super::concurrent::InitialMarkRootsWorkFactory;
use super::global::StickyImmix;

pub struct StickyImmixNurseryGCWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);

impl<VM: VMBinding> crate::scheduler::GCWorkContext for StickyImmixNurseryGCWorkContext<VM> {
    type VM = VM;
    type PlanType = StickyImmix<VM>;
    type DefaultTrace = GenNurseryTrace<VM, Self::PlanType, DEFAULT_TRACE>;
    type PinningTrace = GenNurseryTrace<VM, Self::PlanType, TRACE_KIND_TRANSITIVE_PIN>;
}

/// The nursery collection that starts a concurrent full-heap mark: the same work, with roots also handed to the mark.
pub struct StickyImmixInitialMarkGCWorkContext<VM: VMBinding>(std::marker::PhantomData<VM>);

impl<VM: VMBinding> crate::scheduler::GCWorkContext for StickyImmixInitialMarkGCWorkContext<VM> {
    type VM = VM;
    type PlanType = StickyImmix<VM>;
    type DefaultTrace = GenNurseryTrace<VM, Self::PlanType, DEFAULT_TRACE>;
    type PinningTrace = GenNurseryTrace<VM, Self::PlanType, TRACE_KIND_TRANSITIVE_PIN>;

    fn make_roots_work_factory(mmtk: &'static MMTK<VM>) -> impl RootsWorkFactory<VM::VMSlot> {
        InitialMarkRootsWorkFactory::<VM>::new(mmtk)
    }
}

pub struct StickyImmixMatureGCWorkContext<VM: VMBinding, const KIND: TraceKind>(
    std::marker::PhantomData<VM>,
);
impl<VM: VMBinding, const KIND: TraceKind> crate::scheduler::GCWorkContext
    for StickyImmixMatureGCWorkContext<VM, KIND>
{
    type VM = VM;
    type PlanType = StickyImmix<VM>;
    type DefaultTrace = PlanTrace<Self::PlanType, KIND>;
    type PinningTrace = PlanTrace<Self::PlanType, TRACE_KIND_TRANSITIVE_PIN>;
}
