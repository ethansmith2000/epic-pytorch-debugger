import functools
import pdb
import re
import sys
import time
import traceback
import weakref
from collections import defaultdict
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


# =============================================================================
# Anomaly Detection Exception
# =============================================================================

class AnomalyDetectedError(Exception):
    """Raised when NaN/Inf is detected in tensor outputs."""
    pass


# =============================================================================
# TreeNode for Computation Graph
# =============================================================================

class TreeNode:
    """
    A node in the computation graph tree.
    
    Attributes:
        data: The current/primary name for this tensor (or operation name if unnamed)
        op_name: The PyTorch operation that created this tensor (e.g., "aten.mm")
        shape: The tensor shape at creation time
        aliases: All variable names this tensor has been assigned to
        children: Child nodes (input tensors to the operation)
    """
    
    def __init__(
        self,
        data: Any,
        op_name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        self.data = data
        self.op_name = op_name
        self.shape = shape
        self.aliases: List[str] = []
        self.children: List["TreeNode"] = []
        self._max_depth: Optional[int] = None

    def add_alias(self, name: str) -> None:
        """Add a variable name alias for this tensor."""
        if name not in self.aliases:
            self.aliases.append(name)
        self.data = name

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

    def remove_child(self, child: "TreeNode") -> None:
        self.children = [c for c in self.children if c is not child]
    
    def _format_node_label(self) -> str:
        """Format the display label for this node."""
        parts = []
        
        if len(self.aliases) > 1:
            primary = self.aliases[-1]
            others = self.aliases[:-1]
            parts.append(f"{primary} (aka {', '.join(others)})")
        elif self.aliases:
            parts.append(self.aliases[0])
        elif self.data:
            parts.append(str(self.data))
        
        if self.op_name:
            if parts:
                parts.append(f"[{self.op_name}]")
            else:
                parts.append(f"<{self.op_name}>")
        
        if self.shape is not None:
            parts.append(f"{self.shape}")
        
        return " ".join(parts) if parts else "<unknown>"

    def __repr__(self, level: int = 0, prefix: str = '', is_last: bool = True, max_depth: Optional[int] = None) -> str:
        if max_depth is None:
            max_depth = self._max_depth
        if max_depth is not None and level >= max_depth:
            connector = "└── " if is_last else "├── "
            return f"{prefix}{connector}... (truncated)\n"
        connector = "└── " if is_last else "├── "
        label = self._format_node_label()
        result = f"{prefix}{connector}{label}\n"
        prefix += "    " if is_last else "│   "
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            result += child.__repr__(level + 1, prefix, is_last_child, max_depth=max_depth)
        return result

    def to_dot(self, graph_name: str = "computation_graph") -> str:
        """Export the tree to DOT format for Graphviz visualization."""
        lines = [f"digraph {graph_name} {{"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=rounded];")
        
        node_id = [0]
        
        def _traverse(node: "TreeNode", parent_id: Optional[int] = None) -> int:
            current_id = node_id[0]
            node_id[0] += 1
            
            label = node._format_node_label().replace('"', '\\"')
            lines.append(f'    node{current_id} [label="{label}"];')
            
            if parent_id is not None:
                lines.append(f"    node{current_id} -> node{parent_id};")
            
            for child in node.children:
                _traverse(child, current_id)
            
            return current_id
        
        _traverse(self)
        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, filename: str, graph_name: str = "computation_graph") -> None:
        """Save the tree to a DOT file."""
        dot_content = self.to_dot(graph_name)
        with open(filename, 'w') as f:
            f.write(dot_content)


# =============================================================================
# Main Debugger Class
# =============================================================================

class EpicPytorchDebugger(TorchDispatchMode):
    """
    A comprehensive PyTorch debugger that intercepts operations and provides
    debugging capabilities including:
    - Computation graph tracking
    - NaN/Inf detection
    - Operation timing
    - Memory tracking
    - Gradient monitoring
    - Conditional breakpoints
    - Tensor snapshots
    - Watch mode
    - Device transfer tracking
    """
    
    _DEFAULT_SKIP_NAMES: frozenset = frozenset([
        "_ih", "_oh", "_dh", "In", "Out", "get_ipython",
        "exit", "quit", "open", "_", "__", "___",
        "_i", "_ii", "_iii", "_i1", "_i2"
    ])

    def __init__(
        self,
        # Basic options
        debug_always: bool = False,
        enabled: bool = True,
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
        # NaN/Inf detection (LOW EFFORT)
        detect_anomaly: bool = False,
        anomaly_pdb: bool = True,
        # Operation timing (LOW EFFORT)
        profile_ops: bool = False,
        # Memory tracking (MEDIUM EFFORT)
        track_memory: bool = False,
        # Gradient hooks (MEDIUM EFFORT)
        track_gradients: bool = False,
        # Conditional breakpoints (MEDIUM EFFORT)
        break_condition: Optional[Callable[[torch.Tensor], bool]] = None,
        # Watch mode (NICE TO HAVE)
        watch_tensors: Optional[List[str]] = None,
        # In-place detection (NICE TO HAVE)
        detect_inplace: bool = False,
        # Device tracking (NICE TO HAVE)
        track_device_transfers: bool = False,
        **debug_kwargs: Any
    ) -> None:
        """
        Initialize the debugger with various options.
        
        Basic Options:
        debug_always: if True, will always run the normal_debug_fn
            enabled: completely enable/disable the debugger
        do_pdb: if True, will run pdb on exception
        exception_fn: function to run on exception
            normal_debug_fn: function that will always run if debug_always is True
            run_trace: if True, will run trace_assignments for variable tracking
            capture_non_tensors: if True, keep references to non-tensor locals
            trace_filter: optional callable(frame) -> bool to decide whether to trace
            exclude_names: locals to ignore during capture
        
        Computation Graph Options:
        only_require_grad: if True, capture tensors that require grad only
        only_leaf_tensors: if True, capture leaf tensors only
        attach_leaf_graphs: if False, skip comp graph attachment for leaf tensors
        comp_graph_max_depth: optional int to truncate tree printing depth
            include_shapes: if True, include tensor shapes in graph nodes
        
        Detection & Profiling:
            detect_anomaly: if True, detect NaN/Inf in outputs and raise/break
            anomaly_pdb: if True, drop into pdb when anomaly detected
            profile_ops: if True, track timing for each operation
            track_memory: if True, track CUDA memory usage per operation
        
        Gradient & Breakpoints:
            track_gradients: if True, register hooks to track gradients
            break_condition: callable(tensor) -> bool, break if returns True
        
        Watch & Tracking:
            watch_tensors: list of tensor names to watch and print on change
            detect_inplace: if True, warn on in-place operations
            track_device_transfers: if True, log device transfers
        """
        super().__init__()
        
        # Basic options
        self.debug_always = debug_always
        self.enabled = enabled
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
        
        # NaN/Inf detection
        self.detect_anomaly = detect_anomaly
        self.anomaly_pdb = anomaly_pdb
        
        # Operation timing
        self.profile_ops = profile_ops
        
        # Memory tracking
        self.track_memory = track_memory
        
        # Gradient hooks
        self.track_gradients = track_gradients
        
        # Conditional breakpoints
        self.break_condition = break_condition
        
        # Watch mode
        self.watch_tensors = set(watch_tensors or [])
        self._watched_tensor_states: Dict[str, Dict[str, Any]] = {}
        
        # In-place detection
        self.detect_inplace = detect_inplace
        
        # Device tracking
        self.track_device_transfers = track_device_transfers
        
        # Cache combined skip names
        self._skip_names: frozenset = self._DEFAULT_SKIP_NAMES | set(exclude_names or [])

        self._reset_session_state()

    def _reset_session_state(self) -> None:
        """Reset per-session state."""
        # Tensor tracking
        self.refs_dict: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._strong_refs: Dict[str, Any] = {}
        # Use id-based dict instead of WeakKeyDictionary to avoid tensor comparison issues
        self._tensor_metadata: Dict[int, Dict[str, Any]] = {}
        
        # Operation timing
        self._op_timings: Dict[str, List[float]] = defaultdict(list)
        
        # Memory tracking
        self._memory_log: List[Dict[str, Any]] = []
        
        # Gradient tracking
        self._grad_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._grad_hooks: List[Any] = []
        
        # Snapshots
        self._snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Device transfer log
        self._device_transfers: List[Dict[str, Any]] = []
        
        # In-place operation log
        self._inplace_ops: List[Dict[str, Any]] = []
        
        # Watch state
        self._watched_tensor_states = {}

    # =========================================================================
    # Metadata helpers
    # =========================================================================
    
    def _set_tensor_metadata(self, tensor: torch.Tensor, attr: str, value: Any) -> None:
        """Store metadata on a tensor."""
        try:
            setattr(tensor, attr, value)
        except AttributeError:
            pass
        tensor_id = id(tensor)
        if tensor_id not in self._tensor_metadata:
            self._tensor_metadata[tensor_id] = {}
        self._tensor_metadata[tensor_id][attr] = value

    def _get_tensor_metadata(self, tensor: torch.Tensor, attr: str, default: Any = None) -> Any:
        """Retrieve metadata from a tensor."""
        if hasattr(tensor, attr):
            return getattr(tensor, attr)
        tensor_id = id(tensor)
        meta = self._tensor_metadata.get(tensor_id)
        if meta is not None and attr in meta:
            return meta[attr]
        return default

    def _store_ref(self, name: str, obj: Any) -> None:
        """Store a reference to an object."""
        if not self.capture_non_tensors and not isinstance(obj, torch.Tensor):
            return
        try:
            self.refs_dict[name] = obj
        except TypeError:
            self._strong_refs[name] = obj

    def _merged_refs(self) -> Dict[str, Any]:
        """Return a merged dict of all stored references."""
        merged = dict(self._strong_refs)
        # Copy keys first to avoid "dictionary changed size during iteration"
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
            # If iteration fails, just use strong refs
            pass
        return merged

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

    # =========================================================================
    # LOW EFFORT: NaN/Inf Detection
    # =========================================================================
    
    def _check_anomaly(self, tensor: torch.Tensor, op_name: str) -> bool:
        """Check tensor for NaN/Inf values. Returns True if anomaly found."""
        if not self.detect_anomaly:
            return False
        
        try:
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            
            if has_nan or has_inf:
                tensor_name = self._get_tensor_metadata(tensor, "tensor_name", "<unnamed>")
                anomaly_type = []
                if has_nan:
                    anomaly_type.append("NaN")
                if has_inf:
                    anomaly_type.append("Inf")
                
                print("\n" + "!" * 60)
                print(f"ANOMALY DETECTED: {', '.join(anomaly_type)}")
                print(f"  Operation: {op_name}")
                print(f"  Tensor: {tensor_name}")
                print(f"  Shape: {tuple(tensor.shape)}")
                print(f"  Device: {tensor.device}")
                
                # Show computation graph if available
                comp_graph = self._get_tensor_metadata(tensor, "comp_graph")
                if comp_graph:
                    print(f"  Computation graph:\n{comp_graph}")
                
                print("!" * 60 + "\n")
                
                return True
        except Exception:
            pass
        
        return False

    # =========================================================================
    # LOW EFFORT: Operation Timing
    # =========================================================================
    
    def get_op_timings(self) -> Dict[str, Dict[str, float]]:
        """Get operation timing statistics."""
        stats = {}
        for op_name, times in self._op_timings.items():
            if times:
                stats[op_name] = {
                    'count': len(times),
                    'total_ms': sum(times) * 1000,
                    'mean_ms': (sum(times) / len(times)) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                }
        return stats

    def print_op_timings(self, top_n: int = 20) -> None:
        """Print operation timing statistics."""
        stats = self.get_op_timings()
        if not stats:
            print("No operation timings recorded.")
            return
        
        # Sort by total time
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        
        print("\n" + "=" * 70)
        print("OPERATION TIMING STATISTICS")
        print("=" * 70)
        print(f"{'Operation':<30} {'Count':>8} {'Total(ms)':>12} {'Mean(ms)':>12}")
        print("-" * 70)
        
        for op_name, s in sorted_ops[:top_n]:
            print(f"{op_name:<30} {s['count']:>8} {s['total_ms']:>12.3f} {s['mean_ms']:>12.3f}")
        
        print("=" * 70 + "\n")

    # =========================================================================
    # LOW EFFORT: Export Graph to DOT
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
    # MEDIUM EFFORT: Memory Tracking
    # =========================================================================
    
    def _log_memory(self, op_name: str, phase: str) -> None:
        """Log current memory usage."""
        if not self.track_memory:
            return
        
        try:
            if torch.cuda.is_available():
                self._memory_log.append({
                    'op': op_name,
                    'phase': phase,
                    'timestamp': time.time(),
                    'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                    'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                })
        except Exception:
            pass

    def get_memory_log(self) -> List[Dict[str, Any]]:
        """Get the memory usage log."""
        return self._memory_log

    def print_memory_summary(self) -> None:
        """Print memory usage summary."""
        if not self._memory_log:
            print("No memory data recorded. Enable with track_memory=True")
            return
        
        print("\n" + "=" * 70)
        print("MEMORY USAGE SUMMARY")
        print("=" * 70)
        
        # Group by operation
        op_memory: Dict[str, List[float]] = defaultdict(list)
        for entry in self._memory_log:
            if entry['phase'] == 'after':
                op_memory[entry['op']].append(entry['allocated_mb'])
        
        if op_memory:
            print(f"{'Operation':<30} {'Count':>8} {'Avg MB':>12} {'Max MB':>12}")
            print("-" * 70)
            for op, mems in sorted(op_memory.items(), key=lambda x: max(x[1]), reverse=True)[:20]:
                print(f"{op:<30} {len(mems):>8} {sum(mems)/len(mems):>12.2f} {max(mems):>12.2f}")
        
        if self._memory_log:
            peak = max(e['max_allocated_mb'] for e in self._memory_log)
            print(f"\nPeak memory allocated: {peak:.2f} MB")
        
        print("=" * 70 + "\n")

    # =========================================================================
    # MEDIUM EFFORT: Gradient Hooks
    # =========================================================================
    
    def _register_grad_hook(self, tensor: torch.Tensor, name: str) -> None:
        """Register a gradient hook on a tensor."""
        if not self.track_gradients or not tensor.requires_grad:
            return
        
        def grad_hook(grad: torch.Tensor) -> None:
            try:
                grad_info = {
                    'timestamp': time.time(),
                    'shape': tuple(grad.shape),
                    'dtype': str(grad.dtype),
                    'device': str(grad.device),
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item() if grad.numel() > 1 else 0.0,
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                }
                self._grad_history[name].append(grad_info)
                
                # Check for anomalies in gradients
                if grad_info['has_nan'] or grad_info['has_inf']:
                    print(f"\n!!! GRADIENT ANOMALY for '{name}': "
                          f"{'NaN' if grad_info['has_nan'] else ''}"
                          f"{'Inf' if grad_info['has_inf'] else ''}")
            except Exception:
                pass
        
        handle = tensor.register_hook(grad_hook)
        self._grad_hooks.append(handle)

    def get_grad_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get gradient history for all tracked tensors."""
        return dict(self._grad_history)

    def print_grad_summary(self) -> None:
        """Print gradient statistics summary."""
        if not self._grad_history:
            print("No gradient data recorded. Enable with track_gradients=True")
            return
        
        print("\n" + "=" * 70)
        print("GRADIENT SUMMARY")
        print("=" * 70)
        print(f"{'Tensor':<25} {'Count':>6} {'Avg Norm':>12} {'Max Norm':>12} {'Anomaly':>8}")
        print("-" * 70)
        
        for name, grads in sorted(self._grad_history.items()):
            norms = [g['norm'] for g in grads]
            has_anomaly = any(g['has_nan'] or g['has_inf'] for g in grads)
            print(f"{name:<25} {len(grads):>6} {sum(norms)/len(norms):>12.4f} "
                  f"{max(norms):>12.4f} {'YES' if has_anomaly else 'no':>8}")
        
        print("=" * 70 + "\n")

    # =========================================================================
    # MEDIUM EFFORT: Conditional Breakpoints
    # =========================================================================
    
    def _check_break_condition(self, tensor: torch.Tensor, op_name: str) -> bool:
        """Check if break condition is met for a tensor."""
        if self.break_condition is None:
            return False
        
        try:
            if self.break_condition(tensor):
                tensor_name = self._get_tensor_metadata(tensor, "tensor_name", "<unnamed>")
                print("\n" + "=" * 60)
                print("BREAKPOINT CONDITION MET")
                print(f"  Operation: {op_name}")
                print(f"  Tensor: {tensor_name}")
                print(f"  Shape: {tuple(tensor.shape)}")
                print(f"  Device: {tensor.device}")
                print("=" * 60 + "\n")
                return True
        except Exception as e:
            print(f"Warning: Break condition raised exception: {e}")
        
        return False

    # =========================================================================
    # HIGH EFFORT: Tensor Snapshots
    # =========================================================================
    
    def snapshot(self, name: str) -> None:
        """Save a snapshot of all tracked tensors."""
        refs = self._merged_refs()
        snapshot_data = {}
        
        for var_name, obj in refs.items():
            if isinstance(obj, torch.Tensor):
                try:
                    # Store a detached clone
                    snapshot_data[var_name] = obj.detach().clone()
                except Exception:
                    pass
        
        self._snapshots[name] = snapshot_data
        print(f"Snapshot '{name}' saved with {len(snapshot_data)} tensors.")

    def list_snapshots(self) -> List[str]:
        """List all saved snapshots."""
        return list(self._snapshots.keys())

    def diff_snapshots(
        self,
        name1: str,
        name2: str,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> Dict[str, Dict[str, Any]]:
        """Compare two snapshots and return differences."""
        if name1 not in self._snapshots:
            raise ValueError(f"Snapshot '{name1}' not found")
        if name2 not in self._snapshots:
            raise ValueError(f"Snapshot '{name2}' not found")
        
        snap1 = self._snapshots[name1]
        snap2 = self._snapshots[name2]
        
        all_keys = set(snap1.keys()) | set(snap2.keys())
        diffs = {}
        
        for key in all_keys:
            if key not in snap1:
                diffs[key] = {'status': 'added_in_second'}
            elif key not in snap2:
                diffs[key] = {'status': 'removed_in_second'}
            else:
                t1, t2 = snap1[key], snap2[key]
                if t1.shape != t2.shape:
                    diffs[key] = {
                        'status': 'shape_changed',
                        'shape1': tuple(t1.shape),
                        'shape2': tuple(t2.shape),
                    }
                elif not torch.allclose(t1, t2, rtol=rtol, atol=atol):
                    diff_tensor = (t2 - t1).abs()
                    diffs[key] = {
                        'status': 'values_changed',
                        'max_diff': diff_tensor.max().item(),
                        'mean_diff': diff_tensor.mean().item(),
                        'num_changed': (diff_tensor > atol).sum().item(),
                    }
        
        return diffs

    def print_snapshot_diff(self, name1: str, name2: str, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        """Print a comparison of two snapshots."""
        diffs = self.diff_snapshots(name1, name2, rtol, atol)
        
        print("\n" + "=" * 70)
        print(f"SNAPSHOT DIFF: '{name1}' vs '{name2}'")
        print("=" * 70)
        
        if not diffs:
            print("No differences found.")
        else:
            for key, info in sorted(diffs.items()):
                status = info['status']
                if status == 'added_in_second':
                    print(f"  + {key}: added")
                elif status == 'removed_in_second':
                    print(f"  - {key}: removed")
                elif status == 'shape_changed':
                    print(f"  ~ {key}: shape {info['shape1']} -> {info['shape2']}")
                elif status == 'values_changed':
                    print(f"  ~ {key}: max_diff={info['max_diff']:.6e}, "
                          f"mean_diff={info['mean_diff']:.6e}, "
                          f"changed={info['num_changed']}")
        
        print("=" * 70 + "\n")

    # =========================================================================
    # NICE TO HAVE: Watch Mode
    # =========================================================================
    
    def _check_watched_tensors(self) -> None:
        """Check watched tensors for changes and print updates."""
        if not self.watch_tensors:
            return
        
        refs = self._merged_refs()
        
        for name in self.watch_tensors:
            if name not in refs:
                continue
            
            tensor = refs[name]
            if not isinstance(tensor, torch.Tensor):
                continue
            
            try:
                current_state = {
                    'shape': tuple(tensor.shape),
                    'mean': tensor.float().mean().item(),
                    'std': tensor.float().std().item() if tensor.numel() > 1 else 0.0,
                    'min': tensor.min().item(),
                    'max': tensor.max().item(),
                }
                
                if name in self._watched_tensor_states:
                    prev = self._watched_tensor_states[name]
                    if prev != current_state:
                        print(f"[WATCH] {name}: shape={current_state['shape']}, "
                              f"mean={current_state['mean']:.4f}, "
                              f"min={current_state['min']:.4f}, "
                              f"max={current_state['max']:.4f}")
                
                self._watched_tensor_states[name] = current_state
            except Exception:
                pass

    def watch(self, *tensor_names: str) -> None:
        """Add tensors to watch list."""
        self.watch_tensors.update(tensor_names)

    def unwatch(self, *tensor_names: str) -> None:
        """Remove tensors from watch list."""
        for name in tensor_names:
            self.watch_tensors.discard(name)

    # =========================================================================
    # NICE TO HAVE: In-place Operation Detection
    # =========================================================================
    
    def _check_inplace(self, func: Any, op_name: str) -> None:
        """Check if operation is in-place and log warning."""
        if not self.detect_inplace:
            return
        
        # In-place ops have patterns like "add_.Tensor", "mul_.Scalar", etc.
        # Check for underscore before a dot or at end of name
        # Also check for "_out" variants
        is_inplace = False
        
        # Check for patterns like "add_." or ending with "_"
        if '_.' in op_name or op_name.endswith('_'):
            # Make sure it's actually an in-place op (has underscore before variant)
            base_op = op_name.split('.')[0] if '.' in op_name else op_name
            if base_op.endswith('_'):
                is_inplace = True
        
        # Also check for "_out" variants
        if '_out' in op_name:
            is_inplace = True
        
        if is_inplace:
            self._inplace_ops.append({
                'op': op_name,
                'timestamp': time.time(),
            })
            print(f"[INPLACE WARNING] Detected in-place operation: {op_name}")

    def get_inplace_ops(self) -> List[Dict[str, Any]]:
        """Get list of detected in-place operations."""
        return self._inplace_ops

    # =========================================================================
    # NICE TO HAVE: Device Transfer Tracking
    # =========================================================================
    
    def _check_device_transfer(self, tensor: torch.Tensor, op_name: str) -> None:
        """Check and log device transfers."""
        if not self.track_device_transfers:
            return
        
        prev_device = self._get_tensor_metadata(tensor, "_prev_device")
        current_device = str(tensor.device)
        
        if prev_device is not None and prev_device != current_device:
            tensor_name = self._get_tensor_metadata(tensor, "tensor_name", "<unnamed>")
            self._device_transfers.append({
                'tensor': tensor_name,
                'from_device': prev_device,
                'to_device': current_device,
                'op': op_name,
                'timestamp': time.time(),
            })
            print(f"[DEVICE] {tensor_name}: {prev_device} -> {current_device} (via {op_name})")
        
        self._set_tensor_metadata(tensor, "_prev_device", current_device)

    def get_device_transfers(self) -> List[Dict[str, Any]]:
        """Get list of device transfers."""
        return self._device_transfers

    # =========================================================================
    # NICE TO HAVE: Logging Integration
    # =========================================================================
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get a summary dictionary suitable for logging to wandb/tensorboard."""
        summary = {}
        
        # Op timing stats
        if self._op_timings:
            op_stats = self.get_op_timings()
            total_time = sum(s['total_ms'] for s in op_stats.values())
            summary['total_op_time_ms'] = total_time
            summary['num_ops'] = sum(s['count'] for s in op_stats.values())
        
        # Memory stats
        if self._memory_log:
            peak_mem = max(e['max_allocated_mb'] for e in self._memory_log)
            summary['peak_memory_mb'] = peak_mem
        
        # Gradient stats
        if self._grad_history:
            all_norms = [g['norm'] for grads in self._grad_history.values() for g in grads]
            if all_norms:
                summary['grad_norm_mean'] = sum(all_norms) / len(all_norms)
                summary['grad_norm_max'] = max(all_norms)
            summary['grad_anomalies'] = sum(
                1 for grads in self._grad_history.values() 
                for g in grads if g['has_nan'] or g['has_inf']
            )
        
        # Device transfers
        summary['device_transfers'] = len(self._device_transfers)
        
        # Inplace ops
        summary['inplace_ops'] = len(self._inplace_ops)
        
        return summary

    # =========================================================================
    # Main dispatch and lifecycle
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
        
        # Check for in-place operations
        self._check_inplace(func, op_name)
        
        # Memory tracking: before
        self._log_memory(op_name, 'before')
        
        # Gather computation graphs from inputs
        comp_graphs: List[TreeNode] = []
        
        def gather_comp_graphs(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                node = self._get_tensor_metadata(x, "comp_graph")
                if node is not None:
                    comp_graphs.append(node)
        
        tree_map(gather_comp_graphs, args)
        tree_map(gather_comp_graphs, kwargs)

        # Execute with timing
        start_time = time.perf_counter() if self.profile_ops else 0
        out = func(*args, **kwargs)
        if self.profile_ops:
            elapsed = time.perf_counter() - start_time
            self._op_timings[op_name].append(elapsed)
        
        # Memory tracking: after
        self._log_memory(op_name, 'after')
        
        # Process outputs
        anomaly_detected = False
        break_triggered = False
        
        def process_output(x: Any) -> Any:
            nonlocal anomaly_detected, break_triggered
            
            if isinstance(x, torch.Tensor):
                # Check for NaN/Inf
                if self._check_anomaly(x, op_name):
                    anomaly_detected = True
                
                # Check break condition
                if self._check_break_condition(x, op_name):
                    break_triggered = True
                
                # Device transfer tracking
                self._check_device_transfer(x, op_name)
                
                # Attach computation graph
                if len(comp_graphs) > 0:
                    if not self.attach_leaf_graphs and x.is_leaf:
                        return x
                    
                    existing_name = self._get_tensor_metadata(x, "tensor_name")
                    shape = self._get_tensor_shape(x)
                    
                    new_node = TreeNode(
                        data=existing_name,
                        op_name=op_name,
                        shape=shape
                    )
                    new_node._max_depth = self.comp_graph_max_depth
                    
                    if existing_name:
                        new_node.add_alias(existing_name)
                    
                    for child in comp_graphs:
                        new_node.add_child(child)
                    self._set_tensor_metadata(x, "comp_graph", new_node)
            
            return x
        
        out = tree_map(process_output, out)
        
        # Handle anomaly detection
        if anomaly_detected and self.anomaly_pdb:
            print("Dropping into pdb due to anomaly...")
            pdb.set_trace()
        
        # Handle break condition
        if break_triggered:
            print("Dropping into pdb due to break condition...")
            pdb.set_trace()
        
        return out

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
            
            # Register gradient hook if tracking gradients
            self._register_grad_hook(tensor, name)
            
            # Track device
            if self.track_device_transfers:
                self._set_tensor_metadata(tensor, "_prev_device", str(tensor.device))

            node = self._get_tensor_metadata(tensor, 'comp_graph')
            if node is None:
                shape = self._get_tensor_shape(tensor)
                node = TreeNode(data=name, op_name=None, shape=shape)
                node.add_alias(name)
            else:
                node.add_alias(name)
            
            node._max_depth = self.comp_graph_max_depth
            self._set_tensor_metadata(tensor, 'comp_graph', node)

        # Check watched tensors
        self._check_watched_tensors()

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        ref_dict = self._merged_refs()
        
        # Clean up gradient hooks
        for hook in self._grad_hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._grad_hooks.clear()
        
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

def epic_pytorch_debugger_decorator(
    enabled: bool = True,
    debug_always: bool = False,
    do_pdb: bool = True,
    exception_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    normal_debug_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    run_trace: bool = True,
    capture_non_tensors: bool = False,
    trace_filter: Optional[Callable[[FrameType], bool]] = None,
    exclude_names: Optional[Set[str]] = None,
    only_require_grad: bool = False,
    only_leaf_tensors: bool = False,
    attach_leaf_graphs: bool = True,
    comp_graph_max_depth: Optional[int] = None,
    include_shapes: bool = True,
    detect_anomaly: bool = False,
    anomaly_pdb: bool = True,
    profile_ops: bool = False,
    track_memory: bool = False,
    track_gradients: bool = False,
    break_condition: Optional[Callable[[torch.Tensor], bool]] = None,
    watch_tensors: Optional[List[str]] = None,
    detect_inplace: bool = False,
    track_device_transfers: bool = False,
    **debug_kwargs: Any
) -> Callable:
    """Decorator to wrap a function with EpicPytorchDebugger context."""
    def debug_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with EpicPytorchDebugger(
                debug_always=debug_always,
                enabled=enabled,
                do_pdb=do_pdb,
                exception_fn=exception_fn,
                normal_debug_fn=normal_debug_fn,
                run_trace=run_trace,
                capture_non_tensors=capture_non_tensors,
                trace_filter=trace_filter,
                exclude_names=exclude_names,
                only_require_grad=only_require_grad,
                only_leaf_tensors=only_leaf_tensors,
                attach_leaf_graphs=attach_leaf_graphs,
                comp_graph_max_depth=comp_graph_max_depth,
                include_shapes=include_shapes,
                detect_anomaly=detect_anomaly,
                anomaly_pdb=anomaly_pdb,
                profile_ops=profile_ops,
                track_memory=track_memory,
                track_gradients=track_gradients,
                break_condition=break_condition,
                watch_tensors=watch_tensors,
                detect_inplace=detect_inplace,
                track_device_transfers=track_device_transfers,
                **debug_kwargs,
            ) as dbg:
                result = func(*args, **kwargs)
                # Store debugger reference for inspection
                wrapper._last_debugger = dbg
                return result
        wrapper._last_debugger = None
        return wrapper
    return debug_decorator


# =============================================================================
# Utility Functions
# =============================================================================

def default_trace_filter(root_dirs: List[Union[str, Path]]) -> Callable[[FrameType], bool]:
    """
    Returns a trace filter that only traces frames whose file paths live under any of the given root_dirs.
    """
    if not root_dirs:
        raise ValueError("root_dirs must be a non-empty sequence of paths")
    roots = [Path(p).resolve() for p in root_dirs]

    def _filter(frame: FrameType) -> bool:
        try:
            fname = Path(frame.f_code.co_filename or "").resolve()
        except Exception:
            return False
        for root in roots:
            try:
                fname.relative_to(root)
                return True
            except Exception:
                continue
        return False

    return _filter


# =============================================================================
# Convenience break conditions
# =============================================================================

def break_on_nan(tensor: torch.Tensor) -> bool:
    """Break condition that triggers on NaN values."""
    return torch.isnan(tensor).any().item()


def break_on_inf(tensor: torch.Tensor) -> bool:
    """Break condition that triggers on Inf values."""
    return torch.isinf(tensor).any().item()


def break_on_large_values(threshold: float = 1e6) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition that triggers on values exceeding threshold."""
    def condition(tensor: torch.Tensor) -> bool:
        return tensor.abs().max().item() > threshold
    return condition


def break_on_small_grad_norm(threshold: float = 1e-7) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition for vanishing gradients."""
    def condition(tensor: torch.Tensor) -> bool:
        if tensor.grad is not None:
            return tensor.grad.norm().item() < threshold
        return False
    return condition


def break_on_large_grad_norm(threshold: float = 1e3) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition for exploding gradients."""
    def condition(tensor: torch.Tensor) -> bool:
        if tensor.grad is not None:
            return tensor.grad.norm().item() > threshold
        return False
    return condition
