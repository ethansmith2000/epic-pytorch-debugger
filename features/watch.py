"""
Watch mixin - Watch tensors, detect inplace ops, track device transfers.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Set

import torch


class WatchMixin:
    """Mixin providing watch mode, inplace detection, and device transfer tracking."""
    
    # Configuration (set by main class)
    detect_inplace: bool
    track_device_transfers: bool
    _get_tensor_metadata: Callable
    _set_tensor_metadata: Callable
    _merged_refs: Callable
    
    def _init_watch(self, watch_tensors: Optional[List[str]] = None) -> None:
        """Initialize watch state."""
        self.watch_tensors: Set[str] = set(watch_tensors or [])
        self._watched_tensor_states: Dict[str, Dict[str, Any]] = {}
        self._inplace_ops: List[Dict[str, Any]] = []
        self._device_transfers: List[Dict[str, Any]] = []
    
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
    
    def get_watched_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of watched tensors."""
        return dict(self._watched_tensor_states)
    
    def _check_inplace(self, func: Any, op_name: str) -> None:
        """Check if operation is in-place and log warning."""
        if not self.detect_inplace:
            return
        
        is_inplace = False
        
        if '_.' in op_name or op_name.endswith('_'):
            base_op = op_name.split('.')[0] if '.' in op_name else op_name
            if base_op.endswith('_'):
                is_inplace = True
        
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
    
    def print_inplace_summary(self) -> None:
        """Print summary of in-place operations."""
        if not self._inplace_ops:
            print("No in-place operations detected.")
            return
        
        print("\n" + "=" * 60)
        print("IN-PLACE OPERATIONS DETECTED")
        print("=" * 60)
        
        op_counts = {}
        for entry in self._inplace_ops:
            op = entry['op']
            op_counts[op] = op_counts.get(op, 0) + 1
        
        for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op}: {count} times")
        
        print(f"\nTotal in-place operations: {len(self._inplace_ops)}")
        print("=" * 60 + "\n")
    
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
    
    def print_device_transfers(self) -> None:
        """Print device transfer summary."""
        if not self._device_transfers:
            print("No device transfers detected.")
            return
        
        print("\n" + "=" * 70)
        print("DEVICE TRANSFERS")
        print("=" * 70)
        
        for t in self._device_transfers:
            print(f"  {t['tensor']}: {t['from_device']} -> {t['to_device']} via {t['op']}")
        
        print(f"\nTotal transfers: {len(self._device_transfers)}")
        print("=" * 70 + "\n")
