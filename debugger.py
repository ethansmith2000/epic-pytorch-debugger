"""
Epic PyTorch Debugger - Main debugger class.

A comprehensive PyTorch debugger that intercepts operations and provides
extensive debugging capabilities through a modular mixin architecture.
"""
import functools
import pdb
import re
import sys
import time
import traceback
import weakref
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

try:
    from .core import TreeNode, TensorMetadataStore, OpInfo
    from .features import (
        AnomalyDetectionMixin,
        ProfilingMixin,
        GradientTrackingMixin,
        ModuleTrackingMixin,
        ShapeTrackingMixin,
        BackwardTrackingMixin,
        SnapshotMixin,
        WatchMixin,
        ReplayMixin,
        SteppingMixin,
        PrecisionTrackingMixin,
        MemoryLeakDetectionMixin,
        DistributedTrackingMixin,
    )
except ImportError:
    from core import TreeNode, TensorMetadataStore, OpInfo
    from features import (
        AnomalyDetectionMixin,
        ProfilingMixin,
        GradientTrackingMixin,
        ModuleTrackingMixin,
        ShapeTrackingMixin,
        BackwardTrackingMixin,
        SnapshotMixin,
        WatchMixin,
        ReplayMixin,
        SteppingMixin,
        PrecisionTrackingMixin,
        MemoryLeakDetectionMixin,
        DistributedTrackingMixin,
    )


class EpicPytorchDebugger(
    TorchDispatchMode,
    AnomalyDetectionMixin,
    ProfilingMixin,
    GradientTrackingMixin,
    ModuleTrackingMixin,
    ShapeTrackingMixin,
    BackwardTrackingMixin,
    SnapshotMixin,
    WatchMixin,
    ReplayMixin,
    SteppingMixin,
    PrecisionTrackingMixin,
    MemoryLeakDetectionMixin,
    DistributedTrackingMixin,
):
    """
    A comprehensive PyTorch debugger that intercepts tensor operations.
    
    Features:
    - Computation graph tracking with visualization
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
    
    Usage:
        with EpicPytorchDebugger(profile_ops=True, detect_anomaly=True) as dbg:
            output = model(input)
            loss.backward()
            dbg.print_op_timings()
    """
    
    _DEFAULT_SKIP_NAMES: frozenset = frozenset([
        "_ih", "_oh", "_dh", "In", "Out", "get_ipython",
        "exit", "quit", "open", "_", "__", "___",
        "_i", "_ii", "_iii", "_i1", "_i2"
    ])

    def __init__(
        self,
        # Basic options
        enabled: bool = True,
        debug_always: bool = False,
        do_pdb: bool = True,
        exception_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        normal_debug_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        run_trace: bool = True,
        capture_non_tensors: bool = False,
        trace_filter: Optional[Callable[[FrameType], bool]] = None,
        exclude_names: Optional[Set[str]] = None,
        # Computation graph options
        only_require_grad: bool = False,
        only_leaf_tensors: bool = False,
        attach_leaf_graphs: bool = True,
        comp_graph_max_depth: Optional[int] = None,
        include_shapes: bool = True,
        # Anomaly detection
        detect_anomaly: bool = False,
        anomaly_pdb: bool = True,
        break_condition: Optional[Callable[[torch.Tensor], bool]] = None,
        # Profiling
        profile_ops: bool = False,
        track_memory: bool = False,
        # Gradient tracking
        track_gradients: bool = False,
        track_grad_flow: bool = False,
        # Module tracking
        track_modules: bool = False,
        # Shape tracking
        track_shapes: bool = False,
        # Backward tracking
        track_backward: bool = False,
        # Watch mode
        watch_tensors: Optional[List[str]] = None,
        detect_inplace: bool = False,
        track_device_transfers: bool = False,
        # Replay
        enable_replay: bool = False,
        replay_capacity: int = 1000,
        # Stepping
        enable_stepping: bool = False,
        # Precision tracking
        track_precision: bool = False,
        # Memory leak detection
        detect_leaks: bool = False,
        leak_threshold_seconds: float = 60.0,
        # Distributed
        track_distributed: bool = False,
        # Performance options
        lightweight: bool = False,
        trace_sample_rate: int = 1,
        **debug_kwargs: Any
    ) -> None:
        super().__init__()
        
        # Apply lightweight mode overrides
        if lightweight:
            run_trace = False  # Disable sys.settrace (biggest performance impact)
            track_backward = False  # No backward tracking needed
            include_shapes = False  # Reduce overhead
            trace_sample_rate = max(trace_sample_rate, 10)
        
        # Basic options
        self.enabled = enabled
        self.debug_always = debug_always
        self.do_pdb = do_pdb
        self.exception_fn = exception_fn
        self.normal_debug_fn = normal_debug_fn
        self.run_trace = run_trace
        self.capture_non_tensors = capture_non_tensors
        self.trace_filter = trace_filter
        self.debug_kwargs = debug_kwargs
        self.oldtrace: Optional[Callable] = None
        self._mode_pushed: bool = False
        
        # Computation graph options
        self.only_require_grad = only_require_grad
        self.only_leaf_tensors = only_leaf_tensors
        self.attach_leaf_graphs = attach_leaf_graphs
        self.comp_graph_max_depth = comp_graph_max_depth
        self.include_shapes = include_shapes
        
        # Anomaly detection
        self.detect_anomaly = detect_anomaly
        self.anomaly_pdb = anomaly_pdb
        self.break_condition = break_condition
        
        # Profiling
        self.profile_ops = profile_ops
        self.track_memory = track_memory
        
        # Gradient tracking
        self.track_gradients = track_gradients
        self.track_grad_flow = track_grad_flow
        
        # Module tracking
        self.track_modules = track_modules
        
        # Shape tracking
        self.track_shapes = track_shapes
        
        # Backward tracking
        self.track_backward = track_backward
        
        # Watch mode
        self.detect_inplace = detect_inplace
        self.track_device_transfers = track_device_transfers
        
        # Precision tracking
        self.track_precision = track_precision
        
        # Distributed
        self.track_distributed = track_distributed
        
        # Cache skip names
        self._skip_names: frozenset = self._DEFAULT_SKIP_NAMES | set(exclude_names or [])
        
        # Performance: trace sampling
        self._trace_sample_rate: int = trace_sample_rate
        self._trace_call_count: int = 0

        # Initialize all mixin state
        self._reset_session_state()
        
        # Initialize feature-specific state
        self._init_anomaly_detection()
        self._init_profiling()
        self._init_gradient_tracking()
        self._init_module_tracking()
        self._init_shape_tracking()
        self._init_backward_tracking()
        self._init_snapshots()
        self._init_watch(watch_tensors)
        self._init_replay(enable_replay, replay_capacity)
        self._init_stepping(enable_stepping)
        self._init_precision_tracking(track_precision)
        self._init_leak_detection(detect_leaks, leak_threshold_seconds)
        self._init_distributed_tracking(track_distributed)

    def _reset_session_state(self) -> None:
        """Reset per-session state."""
        # Tensor tracking
        self.refs_dict: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._strong_refs: Dict[str, Any] = {}
        self._metadata_store = TensorMetadataStore()
        # Track if we have any computation graphs (for optimization)
        self._has_comp_graphs: bool = False

    # =========================================================================
    # Tensor Metadata Helpers
    # =========================================================================
    
    def _set_tensor_metadata(self, tensor: torch.Tensor, attr: str, value: Any) -> None:
        """Store metadata on a tensor."""
        self._metadata_store.set(tensor, attr, value)

    def _get_tensor_metadata(self, tensor: torch.Tensor, attr: str, default: Any = None) -> Any:
        """Retrieve metadata from a tensor."""
        return self._metadata_store.get(tensor, attr, default)

    def _store_ref(self, name: str, obj: Any) -> None:
        """Store a reference to an object."""
        if not self.capture_non_tensors and not isinstance(obj, torch.Tensor):
            return
        try:
            self.refs_dict[name] = obj
        except TypeError:
            self._strong_refs[name] = obj
    
    def track(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Manually track a tensor for snapshots and debugging.
        
        Use this when run_trace=False to still get tensor tracking:
            x = dbg.track("input", torch.randn(32, 10))
            output = dbg.track("output", model(x))
        
        Returns the tensor unchanged (for chaining).
        """
        self._store_ref(name, tensor)
        self._set_tensor_metadata(tensor, 'tensor_name', name)
        return tensor

    def _merged_refs(self) -> Dict[str, Any]:
        """Return a merged dict of all stored references."""
        merged = dict(self._strong_refs)
        try:
            keys = list(self.refs_dict.keys())
            for k in keys:
                try:
                    v = self.refs_dict.get(k)
                    if v is not None:
                        merged[k] = v
                except (KeyError, TypeError):
                    pass
        except RuntimeError:
            pass
        return merged

    # =========================================================================
    # Trace Control
    # =========================================================================
    
    def _should_trace(self, frame: FrameType) -> bool:
        """Determine if a frame should be traced."""
        if not self.enabled:
            return False
        if self.trace_filter is not None:
            try:
                return bool(self.trace_filter(frame))
            except Exception:
                return True
        filename = frame.f_code.co_filename or ""
        return not any(marker in filename for marker in ("site-packages", "torch/", "torch\\", "python"))

    def _get_op_name(self, func: Any) -> str:
        """Extract a readable operation name from a torch dispatch function."""
        if hasattr(func, '__name__'):
            return func.__name__
        if hasattr(func, 'name'):
            return func.name()
        func_str = str(func)
        if 'aten.' in func_str:
            parts = func_str.split('.')
            for i, part in enumerate(parts):
                if part == 'aten' and i + 1 < len(parts):
                    return parts[i + 1]
        return func_str

    def _get_tensor_shape(self, tensor: torch.Tensor) -> Optional[Tuple[int, ...]]:
        """Safely get tensor shape."""
        if not self.include_shapes:
            return None
        try:
            return tuple(tensor.shape)
        except Exception:
            return None
    
    def _extract_shapes_fast(self, obj: Any) -> List[Tuple[int, ...]]:
        """
        OPTIMIZED: Extract shapes from tensors without tree_map overhead.
        Fast path for common cases (single tensor, tuple/list of tensors).
        """
        if isinstance(obj, torch.Tensor):
            try:
                return [tuple(obj.shape)]
            except Exception:
                return []
        
        shapes = []
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if isinstance(item, torch.Tensor):
                    try:
                        shapes.append(tuple(item.shape))
                    except Exception:
                        pass
                elif isinstance(item, (list, tuple, dict)):
                    shapes.extend(self._extract_shapes_fast(item))
        elif isinstance(obj, dict):
            for item in obj.values():
                if isinstance(item, torch.Tensor):
                    try:
                        shapes.append(tuple(item.shape))
                    except Exception:
                        pass
                elif isinstance(item, (list, tuple, dict)):
                    shapes.extend(self._extract_shapes_fast(item))
        return shapes
    
    def _gather_comp_graphs_fast(self, args: Tuple, kwargs: Dict) -> List["TreeNode"]:
        """
        OPTIMIZED: Gather computation graphs from input tensors without tree_map.
        """
        comp_graphs = []
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                node = self._get_tensor_metadata(arg, "comp_graph")
                if node is not None:
                    comp_graphs.append(node)
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        node = self._get_tensor_metadata(item, "comp_graph")
                        if node is not None:
                            comp_graphs.append(node)
        
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                node = self._get_tensor_metadata(v, "comp_graph")
                if node is not None:
                    comp_graphs.append(node)
        
        return comp_graphs

    # =========================================================================
    # Graph Export
    # =========================================================================
    
    def export_graph(self, tensor_or_name: Union[torch.Tensor, str], filename: str) -> None:
        """Export computation graph for a tensor to DOT file."""
        if isinstance(tensor_or_name, str):
            refs = self._merged_refs()
            if tensor_or_name not in refs:
                print(f"Tensor '{tensor_or_name}' not found in tracked tensors.")
                return
            tensor = refs[tensor_or_name]
        else:
            tensor = tensor_or_name
        
        comp_graph = self._get_tensor_metadata(tensor, "comp_graph")
        if comp_graph is None:
            print("No computation graph found for this tensor.")
            return
        
        comp_graph.save_dot(filename)
        print(f"Computation graph saved to {filename}")

    # =========================================================================
    # Summary / Logging
    # =========================================================================
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get a summary dictionary suitable for logging to wandb/tensorboard."""
        summary = {}
        
        # Op timing stats
        if hasattr(self, '_op_timings') and self._op_timings:
            op_stats = self.get_op_timings()
            total_time = sum(s['total_ms'] for s in op_stats.values())
            summary['total_op_time_ms'] = total_time
            summary['num_ops'] = sum(s['count'] for s in op_stats.values())
        
        # Memory stats
        if hasattr(self, '_memory_log') and self._memory_log:
            peak_mem = max(e['max_allocated_mb'] for e in self._memory_log)
            summary['peak_memory_mb'] = peak_mem
        
        # Gradient stats
        if hasattr(self, '_grad_history') and self._grad_history:
            all_norms = [g['norm'] for grads in self._grad_history.values() for g in grads]
            if all_norms:
                summary['grad_norm_mean'] = sum(all_norms) / len(all_norms)
                summary['grad_norm_max'] = max(all_norms)
        
        # Module stats
        if hasattr(self, '_module_timings') and self._module_timings:
            module_stats = self.get_module_timings()
            if module_stats:
                summary['num_modules_tracked'] = len(module_stats)
        
        # Shape changes
        if hasattr(self, '_shape_log'):
            summary['shape_changes'] = len(self._shape_log)
        
        # Backward stats
        if hasattr(self, '_backward_ops') and self._backward_ops:
            summary['backward_ops'] = len(self._backward_ops)
            summary['backward_time_ms'] = sum(op['time_ms'] for op in self._backward_ops)
        
        # Grad flow stats
        if hasattr(self, '_grad_flow_data') and self._grad_flow_data:
            all_flow_norms = [e['grad_norm'] for e in self._grad_flow_data]
            if all_flow_norms:
                summary['grad_flow_mean'] = sum(all_flow_norms) / len(all_flow_norms)
                summary['grad_flow_max'] = max(all_flow_norms)
        
        # Anomaly stats
        if hasattr(self, '_anomaly_count'):
            summary['anomaly_count'] = self._anomaly_count
        
        # Replay stats
        if hasattr(self, '_saved_ops'):
            summary['saved_ops'] = len(self._saved_ops)
        
        # Precision stats
        if hasattr(self, '_precision_loss_count'):
            summary['precision_losses'] = self._precision_loss_count
        
        # Leak stats
        if hasattr(self, '_leaked_tensors'):
            summary['potential_leaks'] = len(self._leaked_tensors)
        
        # Distributed stats
        if hasattr(self, '_collective_ops') and self._collective_ops:
            summary['collective_ops'] = len(self._collective_ops)
        
        return summary

    # =========================================================================
    # Main Dispatch
    # =========================================================================

    def __torch_dispatch__(
        self,
        func: Callable,
        types: Tuple[type, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        kwargs = kwargs or {}

        if not self.enabled:
            return func(*args, **kwargs)

        op_name = self._get_op_name(func)
        
        # Check if we're in backward pass (now just a flag check, very fast)
        in_backward = self.track_backward and self._is_in_backward()
        
        # Only check inplace if enabled
        if self.detect_inplace:
            self._check_inplace(func, op_name)
        
        # Memory tracking: before (only if enabled)
        if self.track_memory:
            self._log_memory(op_name, 'before')
        
        # Record module operation (only if enabled)
        if self.track_modules:
            self._record_module_op(op_name)
        
        # Extract input shapes/dtypes only if needed
        input_shapes = []
        if self.track_shapes or in_backward:
            input_shapes = self._extract_shapes_fast((args, kwargs))
        
        input_dtypes_info = []
        if self.track_precision:
            input_dtypes_info = self._extract_dtypes((args, kwargs))
        
        # OPTIMIZATION: Only gather comp graphs if we're building computation graphs
        # (i.e., if run_trace is enabled or we have existing graphs)
        comp_graphs: List[TreeNode] = []
        build_comp_graph = self.run_trace or self._has_comp_graphs
        if build_comp_graph:
            comp_graphs = self._gather_comp_graphs_fast(args, kwargs)

        # Handle interactive stepping
        if self.enable_stepping:
            module_name = self._get_current_module() if self.track_modules else None
            self._handle_step_pause(op_name, func, args, kwargs, module_name)
            self._step_count += 1

        # Execute with timing (only time if profiling)
        if self.profile_ops or in_backward:
            start_time = time.perf_counter()
            out = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            if self.profile_ops:
                self._record_op_time(op_name, elapsed)
        else:
            out = func(*args, **kwargs)
            elapsed = 0.0
        
        # Memory tracking: after
        if self.track_memory:
            self._log_memory(op_name, 'after')
        
        # Extract output shapes only if needed
        output_shapes = []
        if self.track_shapes or in_backward:
            output_shapes = self._extract_shapes_fast(out)
        
        # Precision tracking
        if self.track_precision and input_dtypes_info:
            output_dtypes_info = self._extract_dtypes(out)
            self._check_dtype_conversion(op_name, 
                [(d, s, i) for d, s, i in zip(input_dtypes_info, input_shapes, range(len(input_shapes)))],
                [(d, s, i) for d, s, i in zip(output_dtypes_info, output_shapes, range(len(output_shapes)))]
            )
        
        # Log shape changes
        if self.track_shapes:
            self._log_shape_change(op_name, input_shapes, output_shapes)
        
        # Log backward operation
        if in_backward:
            self._log_backward_op(op_name, elapsed, input_shapes, output_shapes)
        
        # Check distributed ops
        if self.track_distributed:
            self._check_distributed_op(op_name, args, elapsed)
        
        # Save for replay
        if self.enable_replay:
            self._save_op_for_replay(op_name, func, args, kwargs, out, input_shapes, output_shapes)
        
        # OPTIMIZATION: Skip output processing if no features need it
        need_output_processing = (
            self.detect_anomaly or 
            self.break_condition is not None or 
            self.detect_leaks or 
            self.track_device_transfers or 
            build_comp_graph
        )
        
        if not need_output_processing:
            return out
        
        # Process outputs
        anomaly_detected = False
        break_triggered = False
        
        def process_single_tensor(x: torch.Tensor) -> torch.Tensor:
            nonlocal anomaly_detected, break_triggered
            
            # Track for memory leaks
            if self.detect_leaks:
                self._track_tensor_creation(x, op_name)
            
            # Check for NaN/Inf
            if self.detect_anomaly and self._check_anomaly(x, op_name):
                anomaly_detected = True
            
            # Check break condition
            if self.break_condition is not None and self._check_break_condition(x, op_name):
                break_triggered = True
            
            # Device transfer tracking
            if self.track_device_transfers:
                self._check_device_transfer(x, op_name)
            
            # Attach computation graph (only if we have parent graphs)
            if comp_graphs:
                if self.attach_leaf_graphs or not x.is_leaf:
                    existing_name = self._get_tensor_metadata(x, "tensor_name")
                    shape = self._get_tensor_shape(x)
                    
                    new_node = TreeNode(
                        data=existing_name,
                        op_name=op_name,
                        shape=shape,
                        dtype=x.dtype,
                        device=str(x.device),
                    )
                    new_node._max_depth = self.comp_graph_max_depth
                    
                    if existing_name:
                        new_node.add_alias(existing_name)
                    
                    for child in comp_graphs:
                        new_node.add_child(child)
                    self._set_tensor_metadata(x, "comp_graph", new_node)
                    self._has_comp_graphs = True
            
            return x
        
        # OPTIMIZATION: Fast path for single tensor (most common)
        if isinstance(out, torch.Tensor):
            out = process_single_tensor(out)
        elif isinstance(out, (list, tuple)):
            processed = []
            for item in out:
                if isinstance(item, torch.Tensor):
                    processed.append(process_single_tensor(item))
                else:
                    processed.append(item)
            out = type(out)(processed)
        else:
            # Fallback for complex structures
            def process_output(x: Any) -> Any:
                if isinstance(x, torch.Tensor):
                    return process_single_tensor(x)
                return x
            out = tree_map(process_output, out)
        
        # Handle anomaly detection
        if anomaly_detected:
            self._handle_anomaly()
        
        # Handle break condition
        if break_triggered and self.do_pdb:
            print("Dropping into pdb due to break condition...")
            pdb.set_trace()
        
        return out

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "EpicPytorchDebugger":
        self._reset_session_state()

        if self.enabled:
            super().__enter__()
            self._mode_pushed = True
        
        if self.run_trace and self.enabled:
            self.oldtrace = sys.gettrace()
            self_ref = self

            def local_trace(frame: FrameType, event: str, arg: Any) -> Callable:
                if event == 'line' and self_ref._should_trace(frame):
                    self_ref.trace_assignments(frame)
                return local_trace

            sys.settrace(local_trace)
        
        return self

    def trace_assignments(self, frame: Optional[FrameType]) -> None:
        """Track variable assignments in the given frame."""
        if not self.enabled or frame is None:
            return
        
        # OPTIMIZATION: Sample traces to reduce overhead
        self._trace_call_count += 1
        if self._trace_sample_rate > 1 and (self._trace_call_count % self._trace_sample_rate) != 0:
            return

        local_vars = list(frame.f_locals.items())
        
        for name, obj in local_vars:
            if re.match(r"__.*__", name) or name in self._skip_names:
                continue
            self._store_ref(name, obj)

        tensors: Dict[str, torch.Tensor] = {}
        for name, obj in local_vars:
            if not isinstance(obj, torch.Tensor):
                continue
            if self.only_require_grad and not obj.requires_grad:
                continue
            if self.only_leaf_tensors and not obj.is_leaf:
                continue
            tensors[name] = obj

        for name, tensor in tensors.items():
            self._set_tensor_metadata(tensor, 'tensor_name', name)
            
            # Register gradient hooks
            self._register_grad_hook(tensor, name)
            self._register_grad_flow_hook(tensor, name)
            
            # Track device
            if self.track_device_transfers:
                self._set_tensor_metadata(tensor, "_prev_device", str(tensor.device))

            node = self._get_tensor_metadata(tensor, 'comp_graph')
            if node is None:
                shape = self._get_tensor_shape(tensor)
                node = TreeNode(data=name, op_name=None, shape=shape, dtype=tensor.dtype)
                node.add_alias(name)
            else:
                node.add_alias(name)
            
            node._max_depth = self.comp_graph_max_depth
            self._set_tensor_metadata(tensor, 'comp_graph', node)

        # Check watched tensors
        self._check_watched_tensors()
        
        # Check for leaks periodically
        if self.detect_leaks:
            self._check_for_leaks()

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        ref_dict = self._merged_refs()
        
        # Clean up hooks
        self._cleanup_grad_hooks()
        self._cleanup_module_hooks()
        
        if exc_type is not None and self.enabled:
            traceback.print_exc()
            if self.exception_fn is not None:
                print("*" * 10 + " BEGIN EXCEPTION_FN " + "*" * 10)
                self.exception_fn(ref_dict, **self.debug_kwargs)
                print("*" * 10 + " END EXCEPTION_FN " + "*" * 10)
            if self.do_pdb:
                pdb.set_trace()

        if self.debug_always and self.enabled:
            if self.normal_debug_fn is not None:
                print("*" * 10 + " BEGIN DEBUG_FN " + "*" * 10)
                self.normal_debug_fn(ref_dict, **self.debug_kwargs)
                print("*" * 10 + " END DEBUG_FN " + "*" * 10)

        if self.run_trace and self.enabled and self.oldtrace is not None:
            sys.settrace(self.oldtrace)
            self.oldtrace = None

        if self._mode_pushed:
            super().__exit__(exc_type, exc_val, exc_tb)
            self._mode_pushed = False


# =============================================================================
# Decorator
# =============================================================================

def epic_debugger(
    enabled: bool = True,
    **kwargs
) -> Callable:
    """
    Decorator to wrap a function with EpicPytorchDebugger context.
    
    Example:
        @epic_debugger(profile_ops=True, detect_anomaly=True)
        def train_step(model, x, y):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **fn_kwargs: Any) -> Any:
            with EpicPytorchDebugger(enabled=enabled, **kwargs) as dbg:
                result = func(*args, **fn_kwargs)
                wrapper._last_debugger = dbg
                return result
        wrapper._last_debugger = None
        return wrapper
    return decorator


# Backwards compatibility alias
epic_pytorch_debugger_decorator = epic_debugger
