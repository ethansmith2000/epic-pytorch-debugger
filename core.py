"""
Core classes and utilities for the Epic PyTorch Debugger.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch


# =============================================================================
# Exceptions
# =============================================================================

class AnomalyDetectedError(Exception):
    """Raised when NaN/Inf is detected in tensor outputs."""
    pass


class BreakpointTriggeredError(Exception):
    """Raised when a breakpoint condition is met."""
    pass


# =============================================================================
# TreeNode for Computation Graph
# =============================================================================

class TreeNode:
    """
    A node in the computation graph tree.
    
    Attributes:
        data: The current/primary name for this tensor (or operation name if unnamed)
        op_name: The PyTorch operation that created this tensor (e.g., "aten.mm")
        shape: The tensor shape at creation time
        dtype: The tensor dtype
        device: The tensor device
        aliases: All variable names this tensor has been assigned to
        children: Child nodes (input tensors to the operation)
    """
    
    def __init__(
        self,
        data: Any,
        op_name: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> None:
        self.data = data
        self.op_name = op_name
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.aliases: List[str] = []
        self.children: List["TreeNode"] = []
        self._max_depth: Optional[int] = None

    def add_alias(self, name: str) -> None:
        """Add a variable name alias for this tensor."""
        if name not in self.aliases:
            self.aliases.append(name)
        self.data = name

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

    def remove_child(self, child: "TreeNode") -> None:
        self.children = [c for c in self.children if c is not child]
    
    def _format_node_label(self, include_dtype: bool = False) -> str:
        """Format the display label for this node."""
        parts = []
        
        if len(self.aliases) > 1:
            primary = self.aliases[-1]
            others = self.aliases[:-1]
            parts.append(f"{primary} (aka {', '.join(others)})")
        elif self.aliases:
            parts.append(self.aliases[0])
        elif self.data:
            parts.append(str(self.data))
        
        if self.op_name:
            if parts:
                parts.append(f"[{self.op_name}]")
            else:
                parts.append(f"<{self.op_name}>")
        
        if self.shape is not None:
            parts.append(f"{self.shape}")
        
        if include_dtype and self.dtype is not None:
            parts.append(f"({self.dtype})")
        
        return " ".join(parts) if parts else "<unknown>"

    def __repr__(self, level: int = 0, prefix: str = '', is_last: bool = True, max_depth: Optional[int] = None) -> str:
        if max_depth is None:
            max_depth = self._max_depth
        if max_depth is not None and level >= max_depth:
            connector = "└── " if is_last else "├── "
            return f"{prefix}{connector}... (truncated)\n"
        connector = "└── " if is_last else "├── "
        label = self._format_node_label()
        result = f"{prefix}{connector}{label}\n"
        prefix += "    " if is_last else "│   "
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            result += child.__repr__(level + 1, prefix, is_last_child, max_depth=max_depth)
        return result

    def to_dot(self, graph_name: str = "computation_graph") -> str:
        """Export the tree to DOT format for Graphviz visualization."""
        lines = [f"digraph {graph_name} {{"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=rounded];")
        
        node_id = [0]
        
        def _traverse(node: "TreeNode", parent_id: Optional[int] = None) -> int:
            current_id = node_id[0]
            node_id[0] += 1
            
            label = node._format_node_label().replace('"', '\\"')
            lines.append(f'    node{current_id} [label="{label}"];')
            
            if parent_id is not None:
                lines.append(f"    node{current_id} -> node{parent_id};")
            
            for child in node.children:
                _traverse(child, current_id)
            
            return current_id
        
        _traverse(self)
        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, filename: str, graph_name: str = "computation_graph") -> None:
        """Save the tree to a DOT file."""
        dot_content = self.to_dot(graph_name)
        with open(filename, 'w') as f:
            f.write(dot_content)


# =============================================================================
# Tensor Metadata Storage
# =============================================================================

class TensorMetadataStore:
    """
    Manages metadata for tensors using both direct attributes and a fallback dict.
    Handles the complexity of tensor identity and weak references.
    """
    
    def __init__(self):
        self._metadata: Dict[int, Dict[str, Any]] = {}
    
    def set(self, tensor: torch.Tensor, attr: str, value: Any) -> None:
        """Store metadata on a tensor."""
        # Try direct attribute first
        try:
            setattr(tensor, attr, value)
        except AttributeError:
            pass
        # Also store in dict as backup
        tensor_id = id(tensor)
        if tensor_id not in self._metadata:
            self._metadata[tensor_id] = {}
        self._metadata[tensor_id][attr] = value

    def get(self, tensor: torch.Tensor, attr: str, default: Any = None) -> Any:
        """Retrieve metadata from a tensor."""
        # Check direct attribute first
        if hasattr(tensor, attr):
            return getattr(tensor, attr)
        # Fall back to dict
        tensor_id = id(tensor)
        meta = self._metadata.get(tensor_id)
        if meta is not None and attr in meta:
            return meta[attr]
        return default
    
    def clear(self) -> None:
        """Clear all stored metadata."""
        self._metadata.clear()
    
    def cleanup_stale(self, live_ids: set) -> None:
        """Remove metadata for tensors that no longer exist."""
        stale = [tid for tid in self._metadata if tid not in live_ids]
        for tid in stale:
            del self._metadata[tid]


# =============================================================================
# Operation Info
# =============================================================================

class OpInfo:
    """Information about a single operation execution."""
    
    __slots__ = ('name', 'func', 'args', 'kwargs', 'input_shapes', 'input_dtypes',
                 'output_shapes', 'output_dtypes', 'elapsed_time', 'module_name',
                 'is_backward', 'timestamp', 'memory_before', 'memory_after')
    
    def __init__(
        self,
        name: str,
        func: Any = None,
        args: Tuple = (),
        kwargs: Dict = None,
        input_shapes: List[Tuple] = None,
        input_dtypes: List[torch.dtype] = None,
        output_shapes: List[Tuple] = None,
        output_dtypes: List[torch.dtype] = None,
        elapsed_time: float = 0.0,
        module_name: str = None,
        is_backward: bool = False,
        timestamp: float = 0.0,
        memory_before: float = 0.0,
        memory_after: float = 0.0,
    ):
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.input_shapes = input_shapes or []
        self.input_dtypes = input_dtypes or []
        self.output_shapes = output_shapes or []
        self.output_dtypes = output_dtypes or []
        self.elapsed_time = elapsed_time
        self.module_name = module_name
        self.is_backward = is_backward
        self.timestamp = timestamp
        self.memory_before = memory_before
        self.memory_after = memory_after
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'input_shapes': self.input_shapes,
            'input_dtypes': [str(d) for d in self.input_dtypes],
            'output_shapes': self.output_shapes,
            'output_dtypes': [str(d) for d in self.output_dtypes],
            'elapsed_ms': self.elapsed_time * 1000,
            'module': self.module_name,
            'is_backward': self.is_backward,
        }
