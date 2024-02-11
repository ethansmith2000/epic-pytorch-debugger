import torch


def print_tensor_details(tensor, name=None):
    if name is None:
        name = ""
    print(f"{name} Device: {tensor.device}")
    print(f"{name} Type: {tensor.dtype}")
    print(f"{name} Shape: {tensor.shape}")
    print(f"{name} dtype: {tensor.dtype}")
    print(f"{name} Is Nan: {torch.isnan(tensor).any()}")
    print(f"{name} Is Inf: {torch.isinf(tensor).any()}")
    print(f"{name} Min: {torch.min(tensor)}")
    print(f"{name} Max: {torch.max(tensor)}")
    print(f"{name} Mean: {torch.mean(tensor)}")
    print(f"{name} Std: {torch.std(tensor)}")
    if hasattr(tensor, "grad") and tensor.grad is not None:
        print_tensor_details(tensor.grad, name=name + " Grad" if name else "Grad")
    print("---")


def print_if_tensor(names=None):
    for name, var in globals().items():

        # only perform debugging for specific variables if we provide that
        if names is not None:
            if name not in names:
                continue

        # if a tensor, print details
        if isinstance(var, torch.Tensor):
            print(f"Variable name: {name}")
            print_tensor_details(var, name=name)
            print("---")

        # if dictionary, tuple, or list, print a map of its hiearchy
        elif isinstance(var, (dict, tuple, list)):
            print(f"Variable name: {name}, type: {type(var)}")
            analyze_hierarchy(var, name)
            print("---")


def recursive(obj, string_to_print, depth):
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            string_to_print += f"{depth * '- '}{i}: {type(item)}, length:{len(obj)}\n"
            recursive(item, string_to_print, depth + 1)

    elif isinstance(obj, dict):
        for key, value in obj.items():
            string_to_print += f"{depth * '- '}{key}: {type(value)}, length:{len(obj)}\n"
            recursive(value, string_to_print, depth + 1)

    else:
        string_to_print += f"{depth * '- '}{obj}\n"

    return string_to_print


def analyze_hierarchy(obj, name):
    """
    decomposes a dictionary, list, or tuple into its components and prints attributes
    """
    string_to_print = f"Variable name: {name}, type: {type(obj)}, length{len(obj)}\n"
    depth = 1
    string_to_print = recursive(obj, string_to_print, depth)
    print(string_to_print)



