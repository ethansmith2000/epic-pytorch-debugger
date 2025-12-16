"""
Distributed Training Support mixin - Track collective operations.
"""
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import torch

try:
    import torch.distributed as dist
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

COLLECTIVE_OPS = {'all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 
                  'reduce', 'scatter', 'gather', 'barrier', 'nccl', 'gloo'}

class CollectiveOpRecord:
    __slots__ = ('op_name', 'timestamp', 'duration_ms', 'tensor_size_mb', 'rank', 'world_size')
    def __init__(self, op_name, duration_ms, tensor_size_mb, rank, world_size):
        self.op_name = op_name
        self.timestamp = time.time()
        self.duration_ms = duration_ms
        self.tensor_size_mb = tensor_size_mb
        self.rank = rank
        self.world_size = world_size

class DistributedTrackingMixin:
    track_distributed: bool
    
    def _init_distributed_tracking(self, track_distributed=False):
        self.track_distributed = track_distributed
        self._collective_ops: List[CollectiveOpRecord] = []
        self._collective_timings: Dict[str, List[float]] = defaultdict(list)
        self._collective_sizes: Dict[str, List[float]] = defaultdict(list)
        self._sync_warnings: List[Dict] = []
        self._rank: Optional[int] = None
        self._world_size: Optional[int] = None
        self._update_rank_info()
    
    def _update_rank_info(self):
        if not HAS_DISTRIBUTED:
            return
        try:
            if dist.is_initialized():
                self._rank = dist.get_rank()
                self._world_size = dist.get_world_size()
        except Exception:
            pass
    
    def _is_collective_op(self, op_name: str) -> bool:
        return any(c in op_name.lower() for c in COLLECTIVE_OPS)
    
    def _log_collective_op(self, op_name: str, elapsed: float, tensor_size: int):
        if not self.track_distributed:
            return
        self._update_rank_info()
        size_mb = tensor_size / 1024 / 1024
        duration_ms = elapsed * 1000
        record = CollectiveOpRecord(op_name, duration_ms, size_mb, self._rank or 0, self._world_size or 1)
        self._collective_ops.append(record)
        self._collective_timings[op_name].append(duration_ms)
        self._collective_sizes[op_name].append(size_mb)
        if duration_ms > 100:
            self._sync_warnings.append({'op': op_name, 'duration_ms': duration_ms})
    
    def _check_distributed_op(self, op_name: str, args: Tuple, elapsed: float):
        if not self.track_distributed or not self._is_collective_op(op_name):
            return
        size = sum(a.element_size() * a.numel() for a in args if isinstance(a, torch.Tensor))
        self._log_collective_op(op_name, elapsed, size)
    
    def get_collective_ops(self) -> List[CollectiveOpRecord]:
        return self._collective_ops
    
    def get_collective_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for op, times in self._collective_timings.items():
            if times:
                sizes = self._collective_sizes.get(op, [0])
                stats[op] = {'count': len(times), 'total_ms': sum(times), 
                            'mean_ms': sum(times)/len(times), 'total_mb': sum(sizes)}
        return stats
    
    def print_distributed_summary(self):
        if not self._collective_ops:
            print("No distributed ops recorded. Enable with track_distributed=True")
            return
        print("\n" + "=" * 70)
        print("DISTRIBUTED TRAINING SUMMARY")
        print("=" * 70)
        if self._rank is not None:
            print(f"Rank: {self._rank} / World Size: {self._world_size}")
        stats = self.get_collective_stats()
        if stats:
            print(f"{'Op':<25} {'Count':>8} {'Total(ms)':>12} {'Mean(ms)':>12}")
            print("-" * 70)
            for op, s in sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True):
                print(f"{op[:24]:<25} {s['count']:>8} {s['total_ms']:>12.2f} {s['mean_ms']:>12.2f}")
        if self._sync_warnings:
            print(f"\nWarnings: {len(self._sync_warnings)} slow collective ops detected")
        print("=" * 70)
