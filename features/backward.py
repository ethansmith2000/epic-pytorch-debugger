"""
Backward pass tracking mixin - Track operations during backward pass.

OPTIMIZED: No longer uses expensive inspect.stack() calls.
Use mark_backward_start() / mark_backward_end() around loss.backward().
"""
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


class BackwardTrackingMixin:
    """Mixin providing backward pass tracking."""
    
    # Configuration (set by main class)
    track_backward: bool
    track_modules: bool
    _get_current_module: Callable
    
    def _init_backward_tracking(self) -> None:
        """Initialize backward tracking state."""
        self._backward_ops: List[Dict[str, Any]] = []
        self._backward_timings: Dict[str, List[float]] = defaultdict(list)
        self._in_backward: bool = False
        self._backward_start_time: Optional[float] = None
        self._backward_total_time: float = 0.0
        self._backward_hook_tensor_ids: Set[int] = set()
    
    def _is_in_backward(self) -> bool:
        """
        Check if we're currently in a backward pass using the flag.
        
        OPTIMIZED: No longer uses expensive inspect.stack() calls.
        The flag is set by mark_backward_start() or auto-detected via hooks.
        """
        return self._in_backward
    
    def mark_backward_start(self) -> None:
        """
        Manually mark the start of backward pass.
        
        Call this before loss.backward() for accurate backward tracking:
            dbg.mark_backward_start()
            loss.backward()
            dbg.mark_backward_end()
        """
        self._in_backward = True
        self._backward_start_time = time.perf_counter()
    
    def mark_backward_end(self) -> None:
        """
        Manually mark the end of backward pass.
        
        Call this after loss.backward().
        """
        if self._in_backward and self._backward_start_time is not None:
            self._backward_total_time += time.perf_counter() - self._backward_start_time
        self._in_backward = False
        self._backward_start_time = None
    
    def _register_backward_start_hook(self, tensor: torch.Tensor) -> None:
        """
        Register a hook that sets _in_backward=True when backward starts on this tensor.
        Useful for auto-detection if you don't want to manually mark backward.
        """
        if not self.track_backward or not tensor.requires_grad:
            return
        
        tensor_id = id(tensor)
        if tensor_id in self._backward_hook_tensor_ids:
            return
        
        self._backward_hook_tensor_ids.add(tensor_id)
        
        def backward_start_hook(grad: torch.Tensor) -> None:
            self._in_backward = True
            if self._backward_start_time is None:
                self._backward_start_time = time.perf_counter()
        
        tensor.register_hook(backward_start_hook)
    
    def _log_backward_op(
        self,
        op_name: str,
        elapsed: float,
        input_shapes: List[Tuple],
        output_shapes: List[Tuple]
    ) -> None:
        """Log a backward pass operation."""
        if not self.track_backward:
            return
        
        current_module = None
        if self.track_modules and hasattr(self, '_get_current_module'):
            current_module = self._get_current_module()
        
        self._backward_ops.append({
            'op': op_name,
            'module': current_module,
            'time_ms': elapsed * 1000,
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'timestamp': time.time(),
        })
        self._backward_timings[op_name].append(elapsed)
    
    def get_backward_ops(self) -> List[Dict[str, Any]]:
        """Get list of backward pass operations."""
        return self._backward_ops
    
    def get_backward_timings(self) -> Dict[str, Dict[str, float]]:
        """Get backward operation timing statistics."""
        stats = {}
        for op_name, times in self._backward_timings.items():
            if times:
                stats[op_name] = {
                    'count': len(times),
                    'total_ms': sum(times) * 1000,
                    'mean_ms': (sum(times) / len(times)) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                }
        return stats
    
    def print_backward_summary(self, top_n: int = 20) -> None:
        """Print backward pass statistics."""
        if not self._backward_ops:
            print("No backward operations recorded. Enable with track_backward=True")
            return
        
        print("\n" + "=" * 80)
        print("BACKWARD PASS STATISTICS")
        print("=" * 80)
        
        total_time = sum(op['time_ms'] for op in self._backward_ops)
        print(f"\nTotal backward ops: {len(self._backward_ops)}")
        print(f"Total backward time: {total_time:.3f} ms")
        
        stats = self.get_backward_timings()
        if stats:
            print(f"\n--- Backward Ops by Time (top {top_n}) ---")
            print(f"{'Operation':<35} {'Count':>8} {'Total(ms)':>12} {'Mean(ms)':>12}")
            print("-" * 80)
            sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
            for op_name, s in sorted_ops[:top_n]:
                display_name = op_name[:34] if len(op_name) > 34 else op_name
                print(f"{display_name:<35} {s['count']:>8} {s['total_ms']:>12.3f} {s['mean_ms']:>12.3f}")
        
        if self.track_modules:
            module_backward_time: Dict[str, float] = defaultdict(float)
            module_backward_count: Dict[str, int] = defaultdict(int)
            for op in self._backward_ops:
                if op['module']:
                    module_backward_time[op['module']] += op['time_ms']
                    module_backward_count[op['module']] += 1
            
            if module_backward_time:
                print(f"\n--- Backward Time by Module (top {top_n}) ---")
                print(f"{'Module':<50} {'Ops':>8} {'Time(ms)':>12}")
                print("-" * 80)
                sorted_mods = sorted(module_backward_time.items(), key=lambda x: x[1], reverse=True)
                for mod_name, time_ms in sorted_mods[:top_n]:
                    display_name = mod_name[:49] if len(mod_name) > 49 else mod_name
                    count = module_backward_count[mod_name]
                    print(f"{display_name:<50} {count:>8} {time_ms:>12.3f}")
        
        print("=" * 80 + "\n")
    
    def get_backward_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest backward operations."""
        stats = self.get_backward_timings()
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        return [{'op': op, **s} for op, s in sorted_ops[:top_n]]
