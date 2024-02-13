import torch
 
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
 
aten = torch.ops.aten
 
def get_shape(i):
    return i.shape
 
def prod(x):
    res = 1
    for i in x:
        res *= i
    return res
 
def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop
 
def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops
 
def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop
 
def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop
 
def conv_flop_jit(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6]
 
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)
 
def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])
 
def conv_backward_flop_jit(inputs: List[Any], outputs: List[Any]):
    grad_out_shape, x_shape, w_shape = [get_shape(i) for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0
 
    if output_mask[0]:
        grad_input_shape = get_shape(outputs[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = get_shape(outputs[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)
 
    return flop_count
 
 
flop_mapping = {
    aten.mm: matmul_flop_jit,
    aten.matmul: matmul_flop_jit,
    aten.addmm: addmm_flop_jit,
    aten.bmm: bmm_flop_jit,
    aten.convolution: conv_flop_jit,
    aten._convolution: conv_flop_jit,
    aten.convolution_backward: conv_backward_flop_jit,
}
 
flop_counts = defaultdict(lambda: defaultdict(int))
parents = ['Global']
 
def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x
 
class FlopTensor(torch.Tensor):
    elem: torch.Tensor
 
    __slots__ = ['elem']
 
    @staticmethod
    def __new__(cls, elem):
        # The wrapping tensor (FlopTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r
 
    def __repr__(self):
        if self.grad_fn:
            return f"FlopTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"FlopTensor({self.elem})"
 
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, FlopTensor) else e
 
        # no_dispatch is only needed if you use enable_python_mode.
        # It prevents infinite recursion.
        rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        outs = normalize_tuple(rs)
 
        if func in flop_mapping:
            global flop_counts
            flop_count = flop_mapping[func](args, outs)
            for par in parents:
                flop_counts[par][func.__name__] += flop_count
 
        def wrap(e):
            return FlopTensor(e) if isinstance(e, torch.Tensor) else e
 
        rs = tree_map(wrap, rs)
        return rs
 
 
def create_backwards_push(name):
    class PushState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            if len(args) == 1:
                return args[0]
            return args
 
        @staticmethod
        def backward(ctx, *grad_outs):
            global parents
            parents.append(name)
            return grad_outs
 
    return PushState.apply
 
def create_backwards_pop(name):
    class PopState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
            if len(args) == 1:
                return args[0]
            return args
 
        @staticmethod
        def backward(ctx, *grad_outs):
            global parents
            assert(parents[-1] == name)
            parents.pop()
            return grad_outs
 
    return PopState.apply
 
 
 
def enter_module(name):
    def f(module, inputs):
        global parents
        parents.append(name)
        inputs = normalize_tuple(inputs)
        out = create_backwards_pop(name)(*inputs)
        return out
 
    return f
 
def exit_module(name):
    def f(module, inputs, outputs):
        global parents
        assert(parents[-1] == name)
        parents.pop()
        outputs = normalize_tuple(outputs)
        return create_backwards_push(name)(*outputs)
    return f
 
def instrument_module(mod):
    for name, module in dict(mod.named_children()).items():
        module.register_forward_pre_hook(enter_module(name))
        module.register_forward_hook(exit_module(name))
 
def start_counting():
    global parents, flop_counts
    parents = ['Global']
    flop_counts.clear()
 
def display_flops():
    for mod in flop_counts.keys():
        print(f"Module: ", mod)
        for k,v in flop_counts[mod].items():
            print(k, v/1e9)
        print()
 
 
import torchvision.models as models
mod = models.resnet18().cuda()
instrument_module(mod)
 
inp = torch.randn(1, 3, 224, 224, device='cuda')
mod(FlopTensor(inp)).sum().backward()
 
display_flops()