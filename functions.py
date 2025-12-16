from typing import Any, Dict, List, Optional, Set, Union

import torch


def format_tensor_details(
    tensor: torch.Tensor,
    name: Optional[str] = None,
    _seen: Optional[Set[int]] = None
) -> str:
    """
    Format tensor details as a string.
    
    Args:
        tensor: The tensor to analyze
        name: Optional name prefix for the output
        _seen: Internal set to track already-seen gradient tensors (prevents infinite loops)
    
    Returns:
        Formatted string with tensor details
    """
    if _seen is None:
        _seen = set()
    if name is None:
        name = ""
    prefix = f"{name} " if name else ""
    
    lines: List[str] = []
    lines.append(f"{prefix}Device: {tensor.device}")
    lines.append(f"{prefix}Type: {type(tensor)}")
    lines.append(f"{prefix}Shape: {tuple(tensor.shape)}")
    lines.append(f"{prefix}Dtype: {tensor.dtype}")

    detached = tensor.detach()
    numel = detached.numel()
    is_complex_tensor = detached.is_complex() if hasattr(detached, "is_complex") else torch.is_complex(detached)
    supports_fp_ops = detached.is_floating_point() or is_complex_tensor

    with torch.no_grad():
        if is_complex_tensor:
            lines.append(f"{prefix}Min: n/a (complex tensor)")
            lines.append(f"{prefix}Max: n/a (complex tensor)")
        elif numel > 0:
            lines.append(f"{prefix}Min: {torch.min(detached)}")
            lines.append(f"{prefix}Max: {torch.max(detached)}")
        else:
            lines.append(f"{prefix}Min: n/a (empty tensor)")
            lines.append(f"{prefix}Max: n/a (empty tensor)")

        if supports_fp_ops:
            lines.append(f"{prefix}Is Nan: {torch.isnan(detached).any()}")
            lines.append(f"{prefix}Is Inf: {torch.isinf(detached).any()}")

            if numel > 0:
                lines.append(f"{prefix}Mean: {torch.mean(detached)}")
                if numel >= 2:
                    lines.append(f"{prefix}Std: {torch.std(detached, unbiased=False)}")
                else:
                    lines.append(f"{prefix}Std: n/a (numel < 2)")
            else:
                lines.append(f"{prefix}Mean: n/a (empty tensor)")
                lines.append(f"{prefix}Std: n/a (empty tensor)")

    if hasattr(tensor, "grad") and tensor.grad is not None:
        grad_tensor = tensor.grad
        grad_id = id(grad_tensor)
        if grad_id not in _seen:
            _seen.add(grad_id)
            grad_name = f"{name} Grad" if name else "Grad"
            lines.append(format_tensor_details(grad_tensor, name=grad_name, _seen=_seen))

    return "\n".join(lines)


def print_tensor_details(
    tensor: torch.Tensor,
    name: Optional[str] = None,
    _seen: Optional[Set[int]] = None
) -> None:
    """Print tensor details. See format_tensor_details for details."""
    print(format_tensor_details(tensor, name=name, _seen=_seen))


def format_vars(
    ref_dict: Dict[str, Any],
    names: Optional[List[str]] = None
) -> str:
    """
    Format variable details from a reference dictionary.
    
    Args:
        ref_dict: Dictionary of variable names to values
        names: Optional list of specific variable names to include
    
    Returns:
        Formatted string with variable details
    """
    output_parts: List[str] = []
    keys = list(ref_dict.keys())
    separator = "---" * 5

    for name in keys:
        # only perform debugging for specific variables if we provide that
        if names is not None and name not in names:
            continue

        # if a tensor, format details
        if isinstance(ref_dict[name], torch.Tensor):
            output_parts.append(format_tensor_details(ref_dict[name], name=name))
            output_parts.append(separator)

        # if dictionary, tuple, or list, format a map of its hierarchy
        elif isinstance(ref_dict[name], (dict, tuple, list)):
            output_parts.append(format_hierarchy(ref_dict[name], name))
            output_parts.append(separator)

    return "\n".join(output_parts)


def print_vars(
    ref_dict: Dict[str, Any],
    names: Optional[List[str]] = None
) -> None:
    """Print variable details. See format_vars for details."""
    print(format_vars(ref_dict, names=names))


def _build_hierarchy_lines(
    obj: Any,
    lines: List[str],
    depth: int,
    keyname: Optional[str] = None
) -> List[str]:
    """
    Recursively build hierarchy representation lines.
    
    Args:
        obj: The object to analyze
        lines: Accumulator list for output lines
        depth: Current nesting depth
        keyname: Optional key/name for this object
    
    Returns:
        Updated list of lines
    """
    indent = depth * "- "
    type_str = f" {type(obj)}"
    
    if keyname is not None:
        prefix = f"{indent}{keyname}{type_str}"
    else:
        prefix = f"{indent}{type_str}"

    if isinstance(obj, (list, tuple)):
        lines.append(f"{prefix}, length:{len(obj)}")
        for item in obj:
            _build_hierarchy_lines(item, lines, depth + 1)

    elif isinstance(obj, dict):
        lines.append(f"{prefix}, length:{len(obj)}")
        for key, value in obj.items():
            _build_hierarchy_lines(value, lines, depth + 1, keyname=key)

    else:
        lines.append(prefix)

    return lines


def format_hierarchy(obj: Union[Dict, List, tuple], name: Optional[str] = None) -> str:
    """
    Format a dictionary, list, or tuple hierarchy as a string.
    
    Args:
        obj: The object to decompose
        name: Optional name for the root object
    
    Returns:
        Formatted string showing the hierarchy
    """
    lines: List[str] = []
    _build_hierarchy_lines(obj, lines, depth=0, keyname=name)
    return "\n".join(lines)


def analyze_hierarchy(obj: Union[Dict, List, tuple], name: Optional[str] = None) -> str:
    """
    Decompose a dictionary, list, or tuple into its components and return formatted string.
    
    Note: This function returns the string. Use print(analyze_hierarchy(...)) to print.
    For backwards compatibility, also prints the result.
    
    Args:
        obj: The object to decompose
        name: Optional name for the root object
    
    Returns:
        Formatted string showing the hierarchy
    """
    result = format_hierarchy(obj, name)
    print(result)
    return result


# Backwards compatibility alias
recursive = _build_hierarchy_lines
