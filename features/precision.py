"""
Mixed Precision Debugging mixin - Track dtype conversions and precision loss.
"""
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.utils._pytree import tree_map


# Dtype precision hierarchy (lower = less precision)
# None means "not comparable" (e.g., bool is not a precision type)
DTYPE_PRECISION = {
    torch.float16: 1,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int8: 0,
    torch.int16: 1,
    torch.int32: 2,
    torch.int64: 3,
    torch.bool: None,  # Bool is not comparable - not a precision loss
    torch.complex64: 2,
    torch.complex128: 3,
}

# Ops to skip for precision tracking (internal checks, etc.)
SKIP_PRECISION_OPS = {'isnan', 'isinf', 'any', 'all', 'eq', 'ne', 'lt', 'le', 'gt', 'ge'}


class DtypeConversion:
    """Record of a dtype conversion."""
    
    __slots__ = ('from_dtype', 'to_dtype', 'op_name', 'module_name', 
                 'shape', 'timestamp', 'is_precision_loss', 'tensor_name')
    
    def __init__(
        self,
        from_dtype: torch.dtype,
        to_dtype: torch.dtype,
        op_name: str,
        module_name: Optional[str],
        shape: Tuple[int, ...],
        tensor_name: Optional[str] = None,
    ):
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype
        self.op_name = op_name
        self.module_name = module_name
        self.shape = shape
        self.timestamp = time.time()
        self.tensor_name = tensor_name
        
        # Check if this is a precision loss
        from_prec = DTYPE_PRECISION.get(from_dtype)
        to_prec = DTYPE_PRECISION.get(to_dtype)
        # Only flag as loss if both are comparable numeric types and precision decreased
        if from_prec is None or to_prec is None:
            self.is_precision_loss = False
        else:
            self.is_precision_loss = to_prec < from_prec
    
    def __repr__(self) -> str:
        loss = " [PRECISION LOSS]" if self.is_precision_loss else ""
        return f"{self.from_dtype} -> {self.to_dtype} via {self.op_name}{loss}"


class PrecisionTrackingMixin:
    """Mixin providing mixed precision debugging capabilities."""
    
    # Configuration
    track_precision: bool
    track_modules: bool
    _get_current_module: Callable
    _get_tensor_metadata: Callable
    
    def _init_precision_tracking(self, track_precision: bool = False) -> None:
        """Initialize precision tracking state."""
        self.track_precision = track_precision
        self._dtype_conversions: List[DtypeConversion] = []
        self._tensor_dtypes: Dict[int, torch.dtype] = {}  # tensor id -> last known dtype
        self._precision_loss_count = 0
        self._dtype_op_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._amp_context_depth = 0
    
    def _extract_dtypes(self, obj: Any) -> List[Tuple[torch.dtype, Tuple[int, ...], int]]:
        """Extract dtypes, shapes, and ids from tensors in a nested structure."""
        results = []
        
        def extract(x):
            if isinstance(x, torch.Tensor):
                try:
                    results.append((x.dtype, tuple(x.shape), id(x)))
                except Exception:
                    pass
        
        tree_map(extract, obj)
        return results
    
    def _check_dtype_conversion(
        self,
        op_name: str,
        input_info: List[Tuple[torch.dtype, Tuple[int, ...], int]],
        output_info: List[Tuple[torch.dtype, Tuple[int, ...], int]],
    ) -> None:
        """Check for dtype conversions and log them."""
        if not self.track_precision:
            return
        
        # Skip internal/comparison ops that naturally change dtype to bool
        op_base = op_name.split('.')[0] if '.' in op_name else op_name
        if op_base in SKIP_PRECISION_OPS:
            return
        
        module_name = None
        if self.track_modules and hasattr(self, '_get_current_module'):
            module_name = self._get_current_module()
        
        # Track dtype statistics
        for dtype, shape, tid in input_info:
            self._dtype_op_stats[op_name][f"input_{dtype}"] += 1
        for dtype, shape, tid in output_info:
            self._dtype_op_stats[op_name][f"output_{dtype}"] += 1
        
        # Look for conversions by comparing input/output dtypes
        input_dtypes = {info[0] for info in input_info}
        
        for dtype, shape, tid in output_info:
            # Check if this tensor had a different dtype before
            prev_dtype = self._tensor_dtypes.get(tid)
            
            if prev_dtype is not None and prev_dtype != dtype:
                conv = DtypeConversion(
                    from_dtype=prev_dtype,
                    to_dtype=dtype,
                    op_name=op_name,
                    module_name=module_name,
                    shape=shape,
                )
                self._dtype_conversions.append(conv)
                
                if conv.is_precision_loss:
                    self._precision_loss_count += 1
                    print(f"[PRECISION] {prev_dtype} -> {dtype} in {op_name}"
                          f"{f' ({module_name})' if module_name else ''}")
            
            # Also check if output dtype differs from all input dtypes
            elif dtype not in input_dtypes and input_dtypes:
                # This operation converted dtypes
                from_dtype = next(iter(input_dtypes))  # Pick one input dtype
                conv = DtypeConversion(
                    from_dtype=from_dtype,
                    to_dtype=dtype,
                    op_name=op_name,
                    module_name=module_name,
                    shape=shape,
                )
                self._dtype_conversions.append(conv)
                
                if conv.is_precision_loss:
                    self._precision_loss_count += 1
            
            # Update tracked dtype
            self._tensor_dtypes[tid] = dtype
    
    def get_dtype_conversions(self) -> List[DtypeConversion]:
        """Get all logged dtype conversions."""
        return self._dtype_conversions
    
    def get_precision_losses(self) -> List[DtypeConversion]:
        """Get only conversions that resulted in precision loss."""
        return [c for c in self._dtype_conversions if c.is_precision_loss]
    
    def get_dtype_stats(self) -> Dict[str, Dict[str, int]]:
        """Get dtype usage statistics per operation."""
        return {k: dict(v) for k, v in self._dtype_op_stats.items()}
    
    def print_precision_summary(self) -> None:
        """Print precision tracking summary."""
        if not self._dtype_conversions:
            print("No dtype conversions recorded. Enable with track_precision=True")
            return
        
        print("\n" + "=" * 80)
        print("PRECISION / DTYPE TRACKING SUMMARY")
        print("=" * 80)
        
        # Summary stats
        total = len(self._dtype_conversions)
        losses = self._precision_loss_count
        print(f"\nTotal dtype conversions: {total}")
        print(f"Precision losses: {losses}")
        
        # Group by conversion type
        conv_counts: Dict[str, int] = defaultdict(int)
        for conv in self._dtype_conversions:
            key = f"{conv.from_dtype} -> {conv.to_dtype}"
            conv_counts[key] += 1
        
        print("\n--- Conversion Types ---")
        for conv_type, count in sorted(conv_counts.items(), key=lambda x: x[1], reverse=True):
            loss_marker = " [LOSS]" if "->" in conv_type else ""
            from_str, to_str = conv_type.split(" -> ")
            from_prec = DTYPE_PRECISION.get(eval(f"torch.{from_str.split('.')[-1]}"), 2)
            to_prec = DTYPE_PRECISION.get(eval(f"torch.{to_str.split('.')[-1]}"), 2)
            if to_prec < from_prec:
                loss_marker = " [PRECISION LOSS]"
            print(f"  {conv_type}: {count}{loss_marker}")
        
        # Precision losses by operation
        if losses > 0:
            print("\n--- Precision Loss Locations ---")
            loss_by_op: Dict[str, int] = defaultdict(int)
            for conv in self._dtype_conversions:
                if conv.is_precision_loss:
                    loss_by_op[conv.op_name] += 1
            
            for op, count in sorted(loss_by_op.items(), key=lambda x: x[1], reverse=True)[:20]:
                print(f"  {op}: {count}")
        
        # Operations with most dtype activity
        print("\n--- Most Active Operations (by dtype changes) ---")
        op_activity = [(op, sum(counts.values())) for op, counts in self._dtype_op_stats.items()]
        for op, count in sorted(op_activity, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {op}: {count}")
        
        print("=" * 80 + "\n")
    
    def print_precision_losses(self, last_n: Optional[int] = None) -> None:
        """Print detailed precision loss information."""
        losses = self.get_precision_losses()
        
        if last_n:
            losses = losses[-last_n:]
        
        if not losses:
            print("No precision losses detected.")
            return
        
        print("\n" + "=" * 80)
        print("PRECISION LOSS EVENTS")
        print("=" * 80)
        print(f"{'From':<15} {'To':<15} {'Operation':<25} {'Module':<20}")
        print("-" * 80)
        
        for conv in losses:
            from_str = str(conv.from_dtype).split('.')[-1]
            to_str = str(conv.to_dtype).split('.')[-1]
            op = conv.op_name[:24] if len(conv.op_name) > 24 else conv.op_name
            mod = (conv.module_name or '-')[:19]
            print(f"{from_str:<15} {to_str:<15} {op:<25} {mod:<20}")
        
        print("=" * 80 + "\n")
    
    def find_fp16_ops(self) -> List[str]:
        """Find operations that use fp16/bf16 tensors."""
        fp16_ops = set()
        for op, stats in self._dtype_op_stats.items():
            for key in stats:
                if 'float16' in key or 'bfloat16' in key:
                    fp16_ops.add(op)
        return list(fp16_ops)
    
    def suggest_precision_fixes(self) -> List[str]:
        """Suggest operations that might benefit from higher precision."""
        suggestions = []
        
        # Find ops with precision loss
        loss_ops = defaultdict(int)
        for conv in self._dtype_conversions:
            if conv.is_precision_loss:
                loss_ops[conv.op_name] += 1
        
        for op, count in sorted(loss_ops.items(), key=lambda x: x[1], reverse=True):
            if 'norm' in op.lower() or 'softmax' in op.lower() or 'loss' in op.lower():
                suggestions.append(
                    f"Consider running '{op}' in fp32 for numerical stability "
                    f"({count} precision loss events)"
                )
        
        return suggestions
