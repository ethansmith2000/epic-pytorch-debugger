"""
Operation Replay mixin - Save and replay operations with modified inputs.
"""
import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils._pytree import tree_map


class SavedOp:
    """A saved operation that can be replayed."""
    
    __slots__ = ('name', 'func', 'args', 'kwargs', 'output', 'timestamp', 
                 'input_shapes', 'output_shapes', 'module_name', 'index')
    
    def __init__(
        self,
        name: str,
        func: Callable,
        args: Tuple,
        kwargs: Dict,
        output: Any,
        timestamp: float,
        input_shapes: List[Tuple],
        output_shapes: List[Tuple],
        module_name: Optional[str] = None,
        index: int = 0,
    ):
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.output = output
        self.timestamp = timestamp
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.module_name = module_name
        self.index = index
    
    def replay(self, modified_args: Optional[Tuple] = None, modified_kwargs: Optional[Dict] = None) -> Any:
        """Replay this operation, optionally with modified inputs."""
        args = modified_args if modified_args is not None else self.args
        kwargs = modified_kwargs if modified_kwargs is not None else self.kwargs
        return self.func(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"SavedOp({self.name}, inputs={self.input_shapes}, outputs={self.output_shapes})"


class ReplayMixin:
    """Mixin providing operation replay capabilities."""
    
    # Configuration
    enable_replay: bool
    replay_capacity: int  # Max ops to store
    track_modules: bool
    _get_current_module: Callable
    
    def _init_replay(self, enable_replay: bool = False, replay_capacity: int = 1000) -> None:
        """Initialize replay state."""
        self.enable_replay = enable_replay
        self.replay_capacity = replay_capacity
        self._saved_ops: List[SavedOp] = []
        self._op_index = 0
        self._replay_bookmarks: Dict[str, int] = {}
    
    def _clone_tensor_args(self, obj: Any) -> Any:
        """Deep clone tensor arguments for replay."""
        def clone_if_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().clone()
            return x
        return tree_map(clone_if_tensor, obj)
    
    def _save_op_for_replay(
        self,
        op_name: str,
        func: Callable,
        args: Tuple,
        kwargs: Dict,
        output: Any,
        input_shapes: List[Tuple],
        output_shapes: List[Tuple],
    ) -> None:
        """Save an operation for potential replay."""
        if not self.enable_replay:
            return
        
        # Clone tensors so they're preserved
        try:
            cloned_args = self._clone_tensor_args(args)
            cloned_kwargs = self._clone_tensor_args(kwargs)
            cloned_output = self._clone_tensor_args(output)
        except Exception:
            return  # Skip if cloning fails
        
        module_name = None
        if self.track_modules and hasattr(self, '_get_current_module'):
            module_name = self._get_current_module()
        
        saved = SavedOp(
            name=op_name,
            func=func,
            args=cloned_args,
            kwargs=cloned_kwargs,
            output=cloned_output,
            timestamp=time.time(),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            module_name=module_name,
            index=self._op_index,
        )
        
        self._saved_ops.append(saved)
        self._op_index += 1
        
        # Enforce capacity limit
        if len(self._saved_ops) > self.replay_capacity:
            self._saved_ops.pop(0)
    
    def bookmark(self, name: str) -> None:
        """Create a named bookmark at the current operation index."""
        self._replay_bookmarks[name] = len(self._saved_ops) - 1
    
    def get_saved_ops(self) -> List[SavedOp]:
        """Get all saved operations."""
        return self._saved_ops
    
    def get_op(self, index: int) -> Optional[SavedOp]:
        """Get a specific saved operation by index."""
        if 0 <= index < len(self._saved_ops):
            return self._saved_ops[index]
        return None
    
    def get_ops_by_name(self, op_name: str) -> List[SavedOp]:
        """Get all saved operations with a specific name."""
        return [op for op in self._saved_ops if op_name in op.name]
    
    def get_ops_by_module(self, module_name: str) -> List[SavedOp]:
        """Get all saved operations from a specific module."""
        return [op for op in self._saved_ops if op.module_name and module_name in op.module_name]
    
    def get_ops_in_range(self, start: int, end: int) -> List[SavedOp]:
        """Get saved operations in an index range."""
        return self._saved_ops[start:end]
    
    def get_ops_from_bookmark(self, bookmark_name: str) -> List[SavedOp]:
        """Get all operations since a bookmark."""
        if bookmark_name not in self._replay_bookmarks:
            raise ValueError(f"Bookmark '{bookmark_name}' not found")
        start_idx = self._replay_bookmarks[bookmark_name]
        return self._saved_ops[start_idx:]
    
    def replay_op(
        self,
        index: int,
        modified_args: Optional[Tuple] = None,
        modified_kwargs: Optional[Dict] = None,
    ) -> Any:
        """
        Replay a specific operation.
        
        Args:
            index: Index of the operation to replay
            modified_args: Optional modified positional arguments
            modified_kwargs: Optional modified keyword arguments
        
        Returns:
            The output of the replayed operation
        """
        op = self.get_op(index)
        if op is None:
            raise ValueError(f"No operation at index {index}")
        return op.replay(modified_args, modified_kwargs)
    
    def replay_range(
        self,
        start: int,
        end: int,
        input_override: Optional[Any] = None,
    ) -> List[Any]:
        """
        Replay a range of operations sequentially.
        
        Args:
            start: Start index
            end: End index (exclusive)
            input_override: Optional input to use for the first operation
        
        Returns:
            List of outputs from each operation
        """
        ops = self.get_ops_in_range(start, end)
        outputs = []
        
        current_output = input_override
        
        for i, op in enumerate(ops):
            if i == 0 and current_output is not None:
                # Use the override as input for first op
                result = op.replay(modified_args=(current_output,))
            else:
                result = op.replay()
            outputs.append(result)
            current_output = result
        
        return outputs
    
    def compare_replay(
        self,
        index: int,
        modified_args: Optional[Tuple] = None,
        modified_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Replay an operation and compare with original output.
        
        Returns dict with comparison results.
        """
        op = self.get_op(index)
        if op is None:
            raise ValueError(f"No operation at index {index}")
        
        new_output = op.replay(modified_args, modified_kwargs)
        
        comparison = {
            'op_name': op.name,
            'original_shapes': op.output_shapes,
            'matches': False,
            'differences': [],
        }
        
        def compare_tensors(orig, new, path=""):
            if isinstance(orig, torch.Tensor) and isinstance(new, torch.Tensor):
                if orig.shape != new.shape:
                    comparison['differences'].append({
                        'path': path,
                        'type': 'shape_mismatch',
                        'original': tuple(orig.shape),
                        'new': tuple(new.shape),
                    })
                elif not torch.allclose(orig, new, rtol=1e-5, atol=1e-8):
                    diff = (orig - new).abs()
                    comparison['differences'].append({
                        'path': path,
                        'type': 'value_mismatch',
                        'max_diff': diff.max().item(),
                        'mean_diff': diff.mean().item(),
                    })
        
        tree_map(lambda o, n: compare_tensors(o, n), op.output, new_output)
        
        comparison['matches'] = len(comparison['differences']) == 0
        return comparison
    
    def print_saved_ops(self, last_n: Optional[int] = None, filter_name: Optional[str] = None) -> None:
        """Print saved operations."""
        ops = self._saved_ops
        
        if filter_name:
            ops = [op for op in ops if filter_name.lower() in op.name.lower()]
        
        if last_n:
            ops = ops[-last_n:]
        
        if not ops:
            print("No saved operations. Enable with enable_replay=True")
            return
        
        print("\n" + "=" * 80)
        print("SAVED OPERATIONS (for replay)")
        print("=" * 80)
        print(f"{'Index':<8} {'Operation':<30} {'Module':<25} {'Shapes':<15}")
        print("-" * 80)
        
        for op in ops:
            module = (op.module_name or '-')[:24]
            shapes = str(op.input_shapes)[:14]
            print(f"{op.index:<8} {op.name[:29]:<30} {module:<25} {shapes:<15}")
        
        print(f"\nTotal saved: {len(self._saved_ops)} | Capacity: {self.replay_capacity}")
        if self._replay_bookmarks:
            print(f"Bookmarks: {list(self._replay_bookmarks.keys())}")
        print("=" * 80 + "\n")
    
    def clear_saved_ops(self) -> None:
        """Clear all saved operations."""
        self._saved_ops.clear()
        self._op_index = 0
        self._replay_bookmarks.clear()
