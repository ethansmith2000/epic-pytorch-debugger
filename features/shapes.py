"""
Shape tracking mixin - Log shape transformations.
"""
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.utils._pytree import tree_map


class ShapeTrackingMixin:
    """Mixin providing shape change logging."""
    
    # Configuration (set by main class)
    track_shapes: bool
    track_modules: bool
    _get_current_module: Callable
    
    def _init_shape_tracking(self) -> None:
        """Initialize shape tracking state."""
        self._shape_log: List[Dict[str, Any]] = []
    
    def _extract_shapes(self, obj: Any) -> List[Tuple[int, ...]]:
        """Extract shapes from tensors in a nested structure."""
        shapes = []
        
        def extract(x):
            if isinstance(x, torch.Tensor):
                try:
                    shapes.append(tuple(x.shape))
                except Exception:
                    pass
        
        tree_map(extract, obj)
        return shapes
    
    def _extract_dtypes(self, obj: Any) -> List[torch.dtype]:
        """Extract dtypes from tensors in a nested structure."""
        dtypes = []
        
        def extract(x):
            if isinstance(x, torch.Tensor):
                try:
                    dtypes.append(x.dtype)
                except Exception:
                    pass
        
        tree_map(extract, obj)
        return dtypes
    
    def _log_shape_change(
        self,
        op_name: str,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]],
    ) -> None:
        """Log a shape transformation."""
        if not self.track_shapes:
            return
        
        # Only log if shapes actually changed
        if input_shapes != output_shapes:
            current_module = None
            if self.track_modules and hasattr(self, '_get_current_module'):
                current_module = self._get_current_module()
            
            self._shape_log.append({
                'op': op_name,
                'module': current_module,
                'input_shapes': input_shapes,
                'output_shapes': output_shapes,
                'timestamp': time.time(),
            })
    
    def get_shape_log(self) -> List[Dict[str, Any]]:
        """Get the shape change log."""
        return self._shape_log
    
    def print_shape_log(self, last_n: Optional[int] = None, filter_op: Optional[str] = None) -> None:
        """
        Print shape transformations.
        
        Args:
            last_n: Only show the last N entries
            filter_op: Only show ops containing this string
        """
        log = self._shape_log
        
        if filter_op:
            log = [e for e in log if filter_op.lower() in e['op'].lower()]
        
        if last_n:
            log = log[-last_n:]
        
        if not log:
            print("No shape changes recorded. Enable with track_shapes=True")
            return
        
        print("\n" + "=" * 90)
        print("SHAPE TRANSFORMATIONS")
        print("=" * 90)
        print(f"{'Op':<25} {'Module':<25} {'Input Shapes':<20} {'Output Shapes':<20}")
        print("-" * 90)
        
        for entry in log:
            op = entry['op'][:24] if len(entry['op']) > 24 else entry['op']
            mod = (entry['module'] or '-')[:24]
            in_shapes = str(entry['input_shapes'])[:19]
            out_shapes = str(entry['output_shapes'])[:19]
            print(f"{op:<25} {mod:<25} {in_shapes:<20} {out_shapes:<20}")
        
        print(f"\nTotal shape changes: {len(self._shape_log)}")
        print("=" * 90 + "\n")
    
    def get_shape_flow(self, initial_shape: Tuple[int, ...]) -> List[Dict[str, Any]]:
        """
        Get the sequence of shape changes starting from a specific shape.
        Useful for tracing how a tensor's shape evolved.
        """
        flow = []
        current_shapes: Set[Tuple[int, ...]] = {initial_shape}
        
        for entry in self._shape_log:
            for in_shape in entry['input_shapes']:
                if in_shape in current_shapes:
                    flow.append(entry)
                    current_shapes.update(entry['output_shapes'])
                    break
        
        return flow
    
    def find_shape_mismatch_ops(self) -> List[Dict[str, Any]]:
        """Find operations where input and output shapes don't match (potential reshape/view points)."""
        mismatches = []
        for entry in self._shape_log:
            in_shapes = entry['input_shapes']
            out_shapes = entry['output_shapes']
            
            # Check if total elements changed (actual reshape vs broadcasting)
            in_elements = sum(torch.tensor(s).prod().item() for s in in_shapes if s)
            out_elements = sum(torch.tensor(s).prod().item() for s in out_shapes if s)
            
            if in_elements != out_elements or len(in_shapes) != len(out_shapes):
                mismatches.append(entry)
        
        return mismatches
