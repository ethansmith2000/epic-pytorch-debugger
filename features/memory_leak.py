"""
Memory Leak Detection mixin - Track tensors that aren't being freed.
"""
import gc
import sys
import time
import weakref
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


class TensorLifetimeTracker:
    """Tracks the lifetime of a tensor using weak references."""
    
    def __init__(
        self,
        tensor: torch.Tensor,
        name: str,
        creation_op: str,
        creation_time: float,
        shape: Tuple[int, ...],
        device: str,
        requires_grad: bool,
    ):
        self.name = name
        self.creation_op = creation_op
        self.creation_time = creation_time
        self.shape = shape
        self.device = device
        self.requires_grad = requires_grad
        self.size_bytes = tensor.element_size() * tensor.numel()
        self.tensor_id = id(tensor)
        
        # Weak reference to track when tensor is freed
        self._ref = weakref.ref(tensor, self._on_deleted)
        self.deletion_time: Optional[float] = None
        self.is_alive = True
    
    def _on_deleted(self, ref):
        """Called when the tensor is garbage collected."""
        self.deletion_time = time.time()
        self.is_alive = False
    
    def get_lifetime(self) -> Optional[float]:
        """Get tensor lifetime in seconds."""
        if self.deletion_time is not None:
            return self.deletion_time - self.creation_time
        elif self.is_alive:
            return time.time() - self.creation_time
        return None
    
    def check_alive(self) -> bool:
        """Check if tensor is still alive."""
        self.is_alive = self._ref() is not None
        return self.is_alive


class MemoryLeakDetectionMixin:
    """Mixin providing memory leak detection capabilities."""
    
    # Configuration
    detect_leaks: bool
    leak_threshold_seconds: float
    track_modules: bool
    _get_current_module: Callable
    _get_tensor_metadata: Callable
    
    def _init_leak_detection(
        self,
        detect_leaks: bool = False,
        leak_threshold_seconds: float = 60.0,
    ) -> None:
        """Initialize leak detection state."""
        self.detect_leaks = detect_leaks
        self.leak_threshold_seconds = leak_threshold_seconds
        self._tensor_trackers: Dict[int, TensorLifetimeTracker] = {}
        self._leaked_tensors: List[TensorLifetimeTracker] = []
        self._tensor_creation_counts: Dict[str, int] = defaultdict(int)
        self._tensor_deletion_counts: Dict[str, int] = defaultdict(int)
        self._memory_snapshots: List[Dict[str, Any]] = []
        self._iteration_tensor_counts: List[int] = []
    
    def _track_tensor_creation(
        self,
        tensor: torch.Tensor,
        op_name: str,
        tensor_name: Optional[str] = None,
    ) -> None:
        """Track a newly created tensor."""
        if not self.detect_leaks:
            return
        
        tid = id(tensor)
        if tid in self._tensor_trackers:
            return  # Already tracked
        
        name = tensor_name or f"tensor_{tid}"
        
        tracker = TensorLifetimeTracker(
            tensor=tensor,
            name=name,
            creation_op=op_name,
            creation_time=time.time(),
            shape=tuple(tensor.shape),
            device=str(tensor.device),
            requires_grad=tensor.requires_grad,
        )
        
        self._tensor_trackers[tid] = tracker
        self._tensor_creation_counts[op_name] += 1
    
    def _check_for_leaks(self) -> List[TensorLifetimeTracker]:
        """Check for tensors that have been alive too long."""
        if not self.detect_leaks:
            return []
        
        current_time = time.time()
        new_leaks = []
        
        # Clean up dead trackers and identify leaks
        dead_ids = []
        for tid, tracker in self._tensor_trackers.items():
            if not tracker.check_alive():
                dead_ids.append(tid)
                self._tensor_deletion_counts[tracker.creation_op] += 1
            else:
                lifetime = current_time - tracker.creation_time
                if lifetime > self.leak_threshold_seconds:
                    if tracker not in self._leaked_tensors:
                        self._leaked_tensors.append(tracker)
                        new_leaks.append(tracker)
        
        # Remove dead trackers
        for tid in dead_ids:
            del self._tensor_trackers[tid]
        
        return new_leaks
    
    def take_memory_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a snapshot of current memory state."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Count alive tensors
        alive_count = sum(1 for t in self._tensor_trackers.values() if t.check_alive())
        
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'tracked_tensors': len(self._tensor_trackers),
            'alive_tensors': alive_count,
            'leaked_tensors': len(self._leaked_tensors),
            'total_creations': sum(self._tensor_creation_counts.values()),
            'total_deletions': sum(self._tensor_deletion_counts.values()),
        }
        
        if torch.cuda.is_available():
            snapshot['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            snapshot['cuda_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        self._memory_snapshots.append(snapshot)
        return snapshot
    
    def mark_iteration_boundary(self) -> None:
        """Mark an iteration boundary for tracking tensor accumulation."""
        alive = sum(1 for t in self._tensor_trackers.values() if t.check_alive())
        self._iteration_tensor_counts.append(alive)
        
        # Warn if tensor count is increasing
        if len(self._iteration_tensor_counts) >= 3:
            recent = self._iteration_tensor_counts[-3:]
            if recent[-1] > recent[0] * 1.5:  # 50% increase
                print(f"[LEAK WARNING] Tensor count increasing: {recent}")
    
    def get_alive_tensors(self) -> List[TensorLifetimeTracker]:
        """Get all currently alive tracked tensors."""
        return [t for t in self._tensor_trackers.values() if t.check_alive()]
    
    def get_leaked_tensors(self) -> List[TensorLifetimeTracker]:
        """Get tensors identified as potential leaks."""
        # Refresh leak detection
        self._check_for_leaks()
        return self._leaked_tensors
    
    def get_tensor_by_op(self, op_name: str) -> List[TensorLifetimeTracker]:
        """Get alive tensors created by a specific operation."""
        return [
            t for t in self._tensor_trackers.values()
            if t.check_alive() and op_name in t.creation_op
        ]
    
    def get_memory_growth(self) -> List[Dict[str, Any]]:
        """Analyze memory growth between snapshots."""
        if len(self._memory_snapshots) < 2:
            return []
        
        growth = []
        for i in range(1, len(self._memory_snapshots)):
            prev = self._memory_snapshots[i-1]
            curr = self._memory_snapshots[i]
            
            growth.append({
                'from_label': prev['label'],
                'to_label': curr['label'],
                'tensor_delta': curr['alive_tensors'] - prev['alive_tensors'],
                'cuda_delta_mb': (
                    curr.get('cuda_allocated_mb', 0) - prev.get('cuda_allocated_mb', 0)
                ),
            })
        
        return growth
    
    def print_leak_summary(self) -> None:
        """Print memory leak detection summary."""
        self._check_for_leaks()
        
        print("\n" + "=" * 80)
        print("MEMORY LEAK DETECTION SUMMARY")
        print("=" * 80)
        
        alive = self.get_alive_tensors()
        print(f"\nTracked tensors: {len(self._tensor_trackers)}")
        print(f"Currently alive: {len(alive)}")
        print(f"Potential leaks (>{self.leak_threshold_seconds}s): {len(self._leaked_tensors)}")
        
        # Memory by device
        device_memory: Dict[str, int] = defaultdict(int)
        for tracker in alive:
            device_memory[tracker.device] += tracker.size_bytes
        
        if device_memory:
            print("\n--- Memory by Device ---")
            for device, size in device_memory.items():
                print(f"  {device}: {size / 1024 / 1024:.2f} MB")
        
        # Longest-lived tensors
        if alive:
            print("\n--- Longest-Lived Tensors ---")
            sorted_alive = sorted(alive, key=lambda t: t.get_lifetime() or 0, reverse=True)[:10]
            for tracker in sorted_alive:
                lifetime = tracker.get_lifetime()
                print(f"  {tracker.name}: {lifetime:.1f}s, {tracker.shape}, {tracker.creation_op}")
        
        # Creation hotspots
        print("\n--- Tensor Creation Hotspots ---")
        sorted_creations = sorted(
            self._tensor_creation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for op, count in sorted_creations:
            deleted = self._tensor_deletion_counts.get(op, 0)
            diff = count - deleted
            status = f" [+{diff} alive]" if diff > 0 else ""
            print(f"  {op}: {count} created, {deleted} deleted{status}")
        
        # Potential leaks detail
        if self._leaked_tensors:
            print("\n--- Potential Memory Leaks ---")
            for tracker in self._leaked_tensors[:20]:
                lifetime = tracker.get_lifetime()
                size_mb = tracker.size_bytes / 1024 / 1024
                print(f"  {tracker.name}: {lifetime:.1f}s, {size_mb:.2f}MB, "
                      f"{tracker.shape}, requires_grad={tracker.requires_grad}")
        
        print("=" * 80 + "\n")
    
    def suggest_leak_fixes(self) -> List[str]:
        """Suggest fixes for detected leaks."""
        suggestions = []
        
        # Check for grad-enabled tensors
        grad_leaks = [t for t in self._leaked_tensors if t.requires_grad]
        if grad_leaks:
            suggestions.append(
                f"Found {len(grad_leaks)} leaked tensors with requires_grad=True. "
                "Consider using .detach() or torch.no_grad() context."
            )
        
        # Check for CUDA tensors
        cuda_leaks = [t for t in self._leaked_tensors if 'cuda' in t.device]
        if cuda_leaks:
            suggestions.append(
                f"Found {len(cuda_leaks)} leaked CUDA tensors. "
                "Consider moving to CPU with .cpu() when done, or using torch.cuda.empty_cache()."
            )
        
        # Check for accumulating tensors
        if len(self._iteration_tensor_counts) >= 5:
            trend = self._iteration_tensor_counts[-5:]
            if all(trend[i] <= trend[i+1] for i in range(len(trend)-1)):
                suggestions.append(
                    "Tensor count is monotonically increasing across iterations. "
                    "Check for tensors being appended to lists without clearing."
                )
        
        return suggestions
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return stats."""
        before = len(self.get_alive_tensors())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after = len(self.get_alive_tensors())
        
        return {
            'before': before,
            'after': after,
            'freed': before - after,
        }
