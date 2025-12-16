"""
Epic PyTorch Debugger - A comprehensive debugging toolkit for PyTorch.

Features:
- Computation graph tracking and visualization
- NaN/Inf anomaly detection
- Operation timing and profiling
- CUDA memory tracking
- Gradient monitoring and flow visualization
- Module-aware tracing (per-layer stats)
- Shape transformation logging
- Backward pass tracking
- Tensor snapshots and comparison
- Watch mode for tensor monitoring
- Operation replay for debugging
- Interactive step-through debugging
- Mixed precision / dtype tracking
- Memory leak detection
- Distributed training support

Basic Usage:
    from epic_pytorch_debugger import EpicPytorchDebugger
    
    with EpicPytorchDebugger(profile_ops=True, detect_anomaly=True) as dbg:
        output = model(input)
        loss.backward()
        dbg.print_op_timings()

With module tracking:
    with EpicPytorchDebugger(track_modules=True, track_memory=True) as dbg:
        dbg.register_modules(model)
        output = model(input)
        dbg.print_module_summary()

As decorator:
    @epic_debugger(profile_ops=True)
    def train_step(model, x, y):
        ...
"""

from .debugger import EpicPytorchDebugger, epic_debugger, epic_pytorch_debugger_decorator
from .core import TreeNode, TensorMetadataStore, OpInfo, AnomalyDetectedError
from .utils import (
    break_on_nan,
    break_on_inf,
    break_on_nan_or_inf,
    break_on_large_values,
    break_on_small_values,
    break_on_small_grad_norm,
    break_on_large_grad_norm,
    break_on_shape,
    break_on_dtype,
    break_on_device,
    combine_conditions,
    default_trace_filter,
    exclude_modules_filter,
    include_only_filter,
    tensor_summary,
    tensor_stats,
)
from .functions import (
    format_tensor_details,
    print_tensor_details,
    format_vars,
    print_vars,
    format_hierarchy,
    analyze_hierarchy,
)

__version__ = "2.0.0"

__all__ = [
    # Main debugger
    "EpicPytorchDebugger",
    "epic_debugger",
    "epic_pytorch_debugger_decorator",
    # Core classes
    "TreeNode",
    "TensorMetadataStore", 
    "OpInfo",
    "AnomalyDetectedError",
    # Break conditions
    "break_on_nan",
    "break_on_inf",
    "break_on_nan_or_inf",
    "break_on_large_values",
    "break_on_small_values",
    "break_on_small_grad_norm",
    "break_on_large_grad_norm",
    "break_on_shape",
    "break_on_dtype",
    "break_on_device",
    "combine_conditions",
    # Trace filters
    "default_trace_filter",
    "exclude_modules_filter",
    "include_only_filter",
    # Utilities
    "tensor_summary",
    "tensor_stats",
    # Formatting functions
    "format_tensor_details",
    "print_tensor_details",
    "format_vars",
    "print_vars",
    "format_hierarchy",
    "analyze_hierarchy",
]
