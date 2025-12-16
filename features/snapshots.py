"""
Snapshot mixin - Save and compare tensor states.
"""
from typing import Any, Callable, Dict, List, Optional

import torch


class SnapshotMixin:
    """Mixin providing tensor snapshot capabilities."""
    
    # Will be provided by main class
    _merged_refs: Callable
    
    def _init_snapshots(self) -> None:
        """Initialize snapshot state."""
        self._snapshots: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def snapshot(self, name: str) -> None:
        """Save a snapshot of all tracked tensors."""
        refs = self._merged_refs()
        snapshot_data = {}
        
        for var_name, obj in refs.items():
            if isinstance(obj, torch.Tensor):
                try:
                    snapshot_data[var_name] = obj.detach().clone()
                except Exception:
                    pass
        
        self._snapshots[name] = snapshot_data
        print(f"Snapshot '{name}' saved with {len(snapshot_data)} tensors.")

    def list_snapshots(self) -> List[str]:
        """List all saved snapshots."""
        return list(self._snapshots.keys())
    
    def get_snapshot(self, name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get a specific snapshot by name."""
        return self._snapshots.get(name)

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
                elif not torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol):
                    diff_tensor = (t2.float() - t1.float()).abs()
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
    
    def delete_snapshot(self, name: str) -> bool:
        """Delete a snapshot."""
        if name in self._snapshots:
            del self._snapshots[name]
            return True
        return False
    
    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()
