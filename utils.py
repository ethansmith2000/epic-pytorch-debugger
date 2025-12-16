"""
Utility functions and break conditions for the Epic PyTorch Debugger.
"""
from pathlib import Path
from types import FrameType
from typing import Callable, List, Union

import torch


# =============================================================================
# Break Conditions
# =============================================================================

def break_on_nan(tensor: torch.Tensor) -> bool:
    """Break condition that triggers on NaN values."""
    try:
        return torch.isnan(tensor).any().item()
    except Exception:
        return False


def break_on_inf(tensor: torch.Tensor) -> bool:
    """Break condition that triggers on Inf values."""
    try:
        return torch.isinf(tensor).any().item()
    except Exception:
        return False


def break_on_nan_or_inf(tensor: torch.Tensor) -> bool:
    """Break condition that triggers on NaN or Inf values."""
    return break_on_nan(tensor) or break_on_inf(tensor)


def break_on_large_values(threshold: float = 1e6) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition that triggers on values exceeding threshold."""
    def condition(tensor: torch.Tensor) -> bool:
        try:
            return tensor.abs().max().item() > threshold
        except Exception:
            return False
    return condition


def break_on_small_values(threshold: float = 1e-7) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition that triggers on very small non-zero values."""
    def condition(tensor: torch.Tensor) -> bool:
        try:
            abs_tensor = tensor.abs()
            nonzero = abs_tensor[abs_tensor > 0]
            if nonzero.numel() > 0:
                return nonzero.min().item() < threshold
            return False
        except Exception:
            return False
    return condition


def break_on_small_grad_norm(threshold: float = 1e-7) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition for vanishing gradients."""
    def condition(tensor: torch.Tensor) -> bool:
        try:
            if tensor.grad is not None:
                return tensor.grad.norm().item() < threshold
            return False
        except Exception:
            return False
    return condition


def break_on_large_grad_norm(threshold: float = 1e3) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition for exploding gradients."""
    def condition(tensor: torch.Tensor) -> bool:
        try:
            if tensor.grad is not None:
                return tensor.grad.norm().item() > threshold
            return False
        except Exception:
            return False
    return condition


def break_on_shape(expected_shape: tuple) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition that triggers when tensor doesn't match expected shape."""
    def condition(tensor: torch.Tensor) -> bool:
        try:
            return tuple(tensor.shape) != expected_shape
        except Exception:
            return False
    return condition


def break_on_dtype(expected_dtype: torch.dtype) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition that triggers when tensor dtype doesn't match."""
    def condition(tensor: torch.Tensor) -> bool:
        try:
            return tensor.dtype != expected_dtype
        except Exception:
            return False
    return condition


def break_on_device(expected_device: str) -> Callable[[torch.Tensor], bool]:
    """Returns a break condition that triggers when tensor is on unexpected device."""
    def condition(tensor: torch.Tensor) -> bool:
        try:
            return str(tensor.device) != expected_device
        except Exception:
            return False
    return condition


def combine_conditions(*conditions: Callable[[torch.Tensor], bool]) -> Callable[[torch.Tensor], bool]:
    """Combine multiple break conditions with OR logic."""
    def combined(tensor: torch.Tensor) -> bool:
        return any(cond(tensor) for cond in conditions)
    return combined


# =============================================================================
# Trace Filters
# =============================================================================

def default_trace_filter(root_dirs: List[Union[str, Path]]) -> Callable[[FrameType], bool]:
    """
    Returns a trace filter that only traces frames whose file paths live under any of the given root_dirs.
    """
    if not root_dirs:
        raise ValueError("root_dirs must be a non-empty sequence of paths")
    roots = [Path(p).resolve() for p in root_dirs]

    def _filter(frame: FrameType) -> bool:
        try:
            fname = Path(frame.f_code.co_filename or "").resolve()
        except Exception:
            return False
        for root in roots:
            try:
                fname.relative_to(root)
                return True
            except Exception:
                continue
        return False

    return _filter


def exclude_modules_filter(module_patterns: List[str]) -> Callable[[FrameType], bool]:
    """
    Returns a trace filter that excludes frames from specified module patterns.
    """
    def _filter(frame: FrameType) -> bool:
        filename = frame.f_code.co_filename or ""
        return not any(pattern in filename for pattern in module_patterns)
    return _filter


def include_only_filter(include_patterns: List[str]) -> Callable[[FrameType], bool]:
    """
    Returns a trace filter that only includes frames matching patterns.
    """
    def _filter(frame: FrameType) -> bool:
        filename = frame.f_code.co_filename or ""
        return any(pattern in filename for pattern in include_patterns)
    return _filter


# =============================================================================
# Helper Functions
# =============================================================================

def get_op_name(func) -> str:
    """Extract a readable operation name from a torch dispatch function."""
    if hasattr(func, '__name__'):
        return func.__name__
    if hasattr(func, 'name'):
        return func.name()
    func_str = str(func)
    if 'aten.' in func_str:
        parts = func_str.split('.')
        for i, part in enumerate(parts):
            if part == 'aten' and i + 1 < len(parts):
                return parts[i + 1]
    return func_str


def tensor_summary(tensor: torch.Tensor) -> dict:
    """Get a summary dict of tensor properties."""
    try:
        return {
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
            'numel': tensor.numel(),
            'size_mb': tensor.element_size() * tensor.numel() / 1024 / 1024,
        }
    except Exception:
        return {}


def tensor_stats(tensor: torch.Tensor) -> dict:
    """Get statistical summary of tensor values."""
    try:
        t = tensor.detach().float()
        return {
            'min': t.min().item(),
            'max': t.max().item(),
            'mean': t.mean().item(),
            'std': t.std().item() if t.numel() > 1 else 0.0,
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item(),
        }
    except Exception:
        return {}
