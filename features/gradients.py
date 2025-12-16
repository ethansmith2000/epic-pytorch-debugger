"""
Gradient tracking mixin - Gradient hooks and flow visualization.

OPTIMIZED: Prevents duplicate hook registration for better performance.
"""
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


class GradientTrackingMixin:
    """Mixin providing gradient tracking and flow visualization."""
    
    # Configuration (set by main class)
    track_gradients: bool
    track_grad_flow: bool
    track_modules: bool
    _get_current_module: Callable
    
    def _init_gradient_tracking(self) -> None:
        """Initialize gradient tracking state."""
        self._grad_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._grad_hooks: List[Any] = []
        self._grad_flow_data: List[Dict[str, Any]] = []
        self._layer_grad_norms: Dict[str, List[float]] = defaultdict(list)
        # OPTIMIZATION: Track which tensors already have hooks to avoid duplicates
        self._hooked_tensor_ids: Set[Tuple[str, int]] = set()
    
    def _register_grad_hook(self, tensor: torch.Tensor, name: str) -> None:
        """Register a gradient hook on a tensor."""
        if not self.track_gradients or not tensor.requires_grad:
            return
        
        # OPTIMIZATION: Skip if this tensor already has a hook registered
        tensor_id = id(tensor)
        hook_key = ('grad', tensor_id)
        if hook_key in self._hooked_tensor_ids:
            return
        self._hooked_tensor_ids.add(hook_key)
        
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
    
    def _register_grad_flow_hook(self, tensor: torch.Tensor, name: str) -> None:
        """Register a hook to track gradient flow through this tensor."""
        if not self.track_grad_flow or not tensor.requires_grad:
            return
        
        # OPTIMIZATION: Skip if this tensor already has a grad flow hook registered
        tensor_id = id(tensor)
        hook_key = ('grad_flow', tensor_id)
        if hook_key in self._hooked_tensor_ids:
            return
        self._hooked_tensor_ids.add(hook_key)
        
        def grad_flow_hook(grad: torch.Tensor) -> None:
            try:
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item() if grad.numel() > 1 else 0.0
                grad_max = grad.abs().max().item()
                
                self._layer_grad_norms[name].append(grad_norm)
                
                current_module = None
                if self.track_modules and hasattr(self, '_get_current_module'):
                    current_module = self._get_current_module()
                
                self._grad_flow_data.append({
                    'tensor': name,
                    'module': current_module,
                    'grad_norm': grad_norm,
                    'grad_mean': grad_mean,
                    'grad_std': grad_std,
                    'grad_max': grad_max,
                    'shape': tuple(grad.shape),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                    'timestamp': time.time(),
                })
            except Exception:
                pass
        
        handle = tensor.register_hook(grad_flow_hook)
        self._grad_hooks.append(handle)
    
    def _cleanup_grad_hooks(self) -> None:
        """Remove all gradient hooks."""
        for hook in self._grad_hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._grad_hooks.clear()

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

    def get_grad_flow_data(self) -> List[Dict[str, Any]]:
        """Get gradient flow data."""
        return self._grad_flow_data
    
    def get_layer_grad_norms(self) -> Dict[str, List[float]]:
        """Get gradient norms per layer/tensor over time."""
        return dict(self._layer_grad_norms)
    
    def print_grad_flow(self) -> None:
        """Print gradient flow summary."""
        if not self._grad_flow_data:
            print("No gradient flow data recorded. Enable with track_grad_flow=True")
            return
        
        print("\n" + "=" * 90)
        print("GRADIENT FLOW ANALYSIS")
        print("=" * 90)
        
        # Aggregate by tensor
        tensor_stats: Dict[str, Dict[str, Any]] = {}
        for entry in self._grad_flow_data:
            name = entry['tensor']
            if name not in tensor_stats:
                tensor_stats[name] = {
                    'norms': [],
                    'means': [],
                    'maxes': [],
                    'has_anomaly': False,
                    'module': entry['module'],
                    'shape': entry['shape'],
                }
            tensor_stats[name]['norms'].append(entry['grad_norm'])
            tensor_stats[name]['means'].append(entry['grad_mean'])
            tensor_stats[name]['maxes'].append(entry['grad_max'])
            if entry['has_nan'] or entry['has_inf']:
                tensor_stats[name]['has_anomaly'] = True
        
        print(f"\n{'Tensor':<30} {'Shape':<20} {'AvgNorm':>12} {'MaxNorm':>12} {'Status':>10}")
        print("-" * 90)
        
        sorted_tensors = sorted(
            tensor_stats.items(),
            key=lambda x: sum(x[1]['norms']) / len(x[1]['norms']) if x[1]['norms'] else 0
        )
        
        for name, stats in sorted_tensors:
            norms = stats['norms']
            avg_norm = sum(norms) / len(norms) if norms else 0
            max_norm = max(norms) if norms else 0
            shape_str = str(stats['shape'])[:19]
            display_name = name[:29] if len(name) > 29 else name
            
            if stats['has_anomaly']:
                status = "⚠ ANOMALY"
            elif avg_norm < 1e-7:
                status = "⚠ VANISH"
            elif avg_norm > 1e3:
                status = "⚠ EXPLODE"
            else:
                status = "OK"
            
            print(f"{display_name:<30} {shape_str:<20} {avg_norm:>12.2e} {max_norm:>12.2e} {status:>10}")
        
        all_norms = [n for stats in tensor_stats.values() for n in stats['norms']]
        if all_norms:
            print(f"\nOverall gradient norm: min={min(all_norms):.2e}, max={max(all_norms):.2e}, "
                  f"mean={sum(all_norms)/len(all_norms):.2e}")
        
        anomaly_count = sum(1 for s in tensor_stats.values() if s['has_anomaly'])
        vanishing_count = sum(1 for s in tensor_stats.values() 
                            if s['norms'] and sum(s['norms'])/len(s['norms']) < 1e-7)
        exploding_count = sum(1 for s in tensor_stats.values() 
                            if s['norms'] and sum(s['norms'])/len(s['norms']) > 1e3)
        
        if anomaly_count or vanishing_count or exploding_count:
            print(f"\n⚠ Issues detected: {anomaly_count} anomalies, "
                  f"{vanishing_count} vanishing, {exploding_count} exploding")
        
        print("=" * 90 + "\n")
    
    def export_grad_flow_dot(self, filename: str, threshold: float = 1e-6) -> None:
        """Export gradient flow visualization to DOT format."""
        if not self._grad_flow_data:
            print("No gradient flow data to export.")
            return
        
        tensor_stats: Dict[str, Dict[str, Any]] = {}
        for entry in self._grad_flow_data:
            name = entry['tensor']
            if name not in tensor_stats:
                tensor_stats[name] = {'avg_norm': 0, 'count': 0, 'has_anomaly': False, 'module': entry['module']}
            tensor_stats[name]['avg_norm'] += entry['grad_norm']
            tensor_stats[name]['count'] += 1
            if entry['has_nan'] or entry['has_inf']:
                tensor_stats[name]['has_anomaly'] = True
        
        for stats in tensor_stats.values():
            if stats['count'] > 0:
                stats['avg_norm'] /= stats['count']
        
        lines = ["digraph grad_flow {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=filled];")
        
        for i, (name, stats) in enumerate(tensor_stats.items()):
            avg_norm = stats['avg_norm']
            
            if stats['has_anomaly']:
                color = "#ff0000"
            elif avg_norm < threshold:
                color = "#87ceeb"
            elif avg_norm > 1e3:
                color = "#ffa500"
            else:
                intensity = min(1.0, avg_norm / 10)
                green = int(200 + 55 * intensity)
                color = f"#90{green:02x}90"
            
            label = f"{name}\\ngrad_norm: {avg_norm:.2e}"
            if stats['module']:
                label += f"\\nmodule: {stats['module']}"
            
            lines.append(f'    node{i} [label="{label}", fillcolor="{color}"];')
        
        lines.append("}")
        
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"Gradient flow visualization saved to {filename}")
