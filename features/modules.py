"""
Module tracking mixin - Track which nn.Module each operation belongs to.
"""
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ModuleTrackingMixin:
    """Mixin providing module-aware operation tracking."""
    
    # Configuration (set by main class)
    track_modules: bool
    track_memory: bool
    
    def _init_module_tracking(self) -> None:
        """Initialize module tracking state."""
        self._module_stack: List[Tuple[str, nn.Module]] = []
        self._module_hooks: List[Any] = []
        self._module_timings: Dict[str, List[float]] = defaultdict(list)
        self._module_memory: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self._module_op_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._current_module_start_time: Dict[str, float] = {}
        self._current_module_start_mem: Dict[str, float] = {}
    
    def register_modules(self, model: nn.Module, prefix: str = "") -> None:
        """
        Register hooks on all modules in a model to track which module each op belongs to.
        
        Args:
            model: The nn.Module to track
            prefix: Optional prefix for module names
        """
        if not self.track_modules:
            return
        
        for name, module in model.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name
            if not full_name:
                full_name = model.__class__.__name__
            
            def make_pre_hook(mod_name: str, mod: nn.Module):
                def pre_hook(module, inputs):
                    self._module_stack.append((mod_name, mod))
                    self._current_module_start_time[mod_name] = time.perf_counter()
                    if self.track_memory and torch.cuda.is_available():
                        self._current_module_start_mem[mod_name] = torch.cuda.memory_allocated() / 1024 / 1024
                return pre_hook
            
            def make_post_hook(mod_name: str, mod: nn.Module):
                def post_hook(module, inputs, outputs):
                    if self._module_stack and self._module_stack[-1][0] == mod_name:
                        self._module_stack.pop()
                    
                    if mod_name in self._current_module_start_time:
                        elapsed = time.perf_counter() - self._current_module_start_time[mod_name]
                        self._module_timings[mod_name].append(elapsed)
                        del self._current_module_start_time[mod_name]
                    
                    if self.track_memory and mod_name in self._current_module_start_mem:
                        if torch.cuda.is_available():
                            end_mem = torch.cuda.memory_allocated() / 1024 / 1024
                            self._module_memory[mod_name].append({
                                'start_mb': self._current_module_start_mem[mod_name],
                                'end_mb': end_mem,
                                'delta_mb': end_mem - self._current_module_start_mem[mod_name],
                            })
                        del self._current_module_start_mem[mod_name]
                return post_hook
            
            h1 = module.register_forward_pre_hook(make_pre_hook(full_name, module))
            h2 = module.register_forward_hook(make_post_hook(full_name, module))
            self._module_hooks.extend([h1, h2])
    
    def _get_current_module(self) -> Optional[str]:
        """Get the name of the currently executing module."""
        if self._module_stack:
            return self._module_stack[-1][0]
        return None
    
    def _record_module_op(self, op_name: str) -> None:
        """Record that an operation was executed within the current module."""
        if not self.track_modules:
            return
        current_module = self._get_current_module()
        if current_module:
            self._module_op_counts[current_module][op_name] += 1
    
    def _cleanup_module_hooks(self) -> None:
        """Remove all module hooks."""
        for hook in self._module_hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._module_hooks.clear()
    
    def get_module_timings(self) -> Dict[str, Dict[str, float]]:
        """Get per-module timing statistics."""
        stats = {}
        for mod_name, times in self._module_timings.items():
            if times:
                stats[mod_name] = {
                    'count': len(times),
                    'total_ms': sum(times) * 1000,
                    'mean_ms': (sum(times) / len(times)) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                }
        return stats
    
    def get_module_memory(self) -> Dict[str, Dict[str, float]]:
        """Get per-module memory statistics."""
        stats = {}
        for mod_name, mems in self._module_memory.items():
            if mems:
                deltas = [m['delta_mb'] for m in mems]
                stats[mod_name] = {
                    'count': len(mems),
                    'avg_delta_mb': sum(deltas) / len(deltas),
                    'max_delta_mb': max(deltas),
                    'total_delta_mb': sum(deltas),
                }
        return stats
    
    def get_module_op_counts(self) -> Dict[str, Dict[str, int]]:
        """Get operation counts per module."""
        return {k: dict(v) for k, v in self._module_op_counts.items()}
    
    def print_module_summary(self, top_n: int = 30) -> None:
        """Print per-module statistics."""
        print("\n" + "=" * 80)
        print("MODULE-LEVEL STATISTICS")
        print("=" * 80)
        
        timing_stats = self.get_module_timings()
        if timing_stats:
            print("\n--- Timing (sorted by total time) ---")
            print(f"{'Module':<45} {'Count':>6} {'Total(ms)':>12} {'Mean(ms)':>12}")
            print("-" * 80)
            sorted_mods = sorted(timing_stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
            for mod_name, s in sorted_mods[:top_n]:
                display_name = mod_name[:44] if len(mod_name) > 44 else mod_name
                print(f"{display_name:<45} {s['count']:>6} {s['total_ms']:>12.3f} {s['mean_ms']:>12.3f}")
        
        mem_stats = self.get_module_memory()
        if mem_stats:
            print("\n--- Memory (sorted by max delta) ---")
            print(f"{'Module':<45} {'Count':>6} {'AvgΔ(MB)':>12} {'MaxΔ(MB)':>12}")
            print("-" * 80)
            sorted_mods = sorted(mem_stats.items(), key=lambda x: x[1]['max_delta_mb'], reverse=True)
            for mod_name, s in sorted_mods[:top_n]:
                display_name = mod_name[:44] if len(mod_name) > 44 else mod_name
                print(f"{display_name:<45} {s['count']:>6} {s['avg_delta_mb']:>12.3f} {s['max_delta_mb']:>12.3f}")
        
        if self._module_op_counts:
            print("\n--- Operations per Module (top modules) ---")
            mod_total_ops = {mod: sum(ops.values()) for mod, ops in self._module_op_counts.items()}
            sorted_mods = sorted(mod_total_ops.items(), key=lambda x: x[1], reverse=True)[:top_n]
            for mod_name, total_ops in sorted_mods:
                display_name = mod_name[:60] if len(mod_name) > 60 else mod_name
                print(f"\n  {display_name} ({total_ops} total ops):")
                ops = self._module_op_counts[mod_name]
                sorted_ops = sorted(ops.items(), key=lambda x: x[1], reverse=True)[:5]
                for op, count in sorted_ops:
                    print(f"    {op}: {count}")
        
        print("=" * 80 + "\n")
