"""
Feature mixins for the Epic PyTorch Debugger.

Each feature is implemented as a mixin class that can be composed
into the main debugger class.
"""

try:
    from .anomaly import AnomalyDetectionMixin
    from .profiling import ProfilingMixin
    from .gradients import GradientTrackingMixin
    from .modules import ModuleTrackingMixin
    from .shapes import ShapeTrackingMixin
    from .backward import BackwardTrackingMixin
    from .snapshots import SnapshotMixin
    from .watch import WatchMixin
    from .replay import ReplayMixin
    from .stepping import SteppingMixin
    from .precision import PrecisionTrackingMixin
    from .memory_leak import MemoryLeakDetectionMixin
    from .distributed import DistributedTrackingMixin
except ImportError:
    from features.anomaly import AnomalyDetectionMixin
    from features.profiling import ProfilingMixin
    from features.gradients import GradientTrackingMixin
    from features.modules import ModuleTrackingMixin
    from features.shapes import ShapeTrackingMixin
    from features.backward import BackwardTrackingMixin
    from features.snapshots import SnapshotMixin
    from features.watch import WatchMixin
    from features.replay import ReplayMixin
    from features.stepping import SteppingMixin
    from features.precision import PrecisionTrackingMixin
    from features.memory_leak import MemoryLeakDetectionMixin
    from features.distributed import DistributedTrackingMixin

__all__ = [
    'AnomalyDetectionMixin',
    'ProfilingMixin',
    'GradientTrackingMixin',
    'ModuleTrackingMixin',
    'ShapeTrackingMixin',
    'BackwardTrackingMixin',
    'SnapshotMixin',
    'WatchMixin',
    'ReplayMixin',
    'SteppingMixin',
    'PrecisionTrackingMixin',
    'MemoryLeakDetectionMixin',
    'DistributedTrackingMixin',
]
