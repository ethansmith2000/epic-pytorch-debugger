"""
Profiling mixin - Operation timing and memory tracking.
"""
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch


class ProfilingMixin:
    """Mixin providing operation timing and memory tracking."""
    
    # Configuration (set by main class)
    profile_ops: bool
    track_memory: bool
    
    def _init_profiling(self) -> None:
        """Initialize profiling state."""
        self._op_timings: Dict[str, List[float]] = defaultdict(list)
        self._memory_log: List[Dict[str, Any]] = []
    
    def _log_memory(self, op_name: str, phase: str) -> float:
        """Log current memory usage. Returns allocated memory in MB."""
        if not self.track_memory:
            return 0.0
        
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                self._memory_log.append({
                    'op': op_name,
                    'phase': phase,
                    'timestamp': time.time(),
                    'allocated_mb': allocated,
                    'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
                })
                return allocated
        except Exception:
            pass
        return 0.0
    
    def _record_op_time(self, op_name: str, elapsed: float) -> None:
        """Record timing for an operation."""
        if self.profile_ops:
            self._op_timings[op_name].append(elapsed)
    
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
        
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        
        print("\n" + "=" * 70)
        print("OPERATION TIMING STATISTICS")
        print("=" * 70)
        print(f"{'Operation':<30} {'Count':>8} {'Total(ms)':>12} {'Mean(ms)':>12}")
        print("-" * 70)
        
        for op_name, s in sorted_ops[:top_n]:
            print(f"{op_name:<30} {s['count']:>8} {s['total_ms']:>12.3f} {s['mean_ms']:>12.3f}")
        
        total_time = sum(s['total_ms'] for s in stats.values())
        total_ops = sum(s['count'] for s in stats.values())
        print("-" * 70)
        print(f"{'TOTAL':<30} {total_ops:>8} {total_time:>12.3f}")
        print("=" * 70 + "\n")

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
    
    def get_memory_peak(self) -> float:
        """Get peak memory usage in MB."""
        if not self._memory_log:
            return 0.0
        return max(e['max_allocated_mb'] for e in self._memory_log)
