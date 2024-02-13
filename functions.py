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
    if tensor.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
        print(f"{name} Mean: {torch.mean(tensor)}")
        print(f"{name} Std: {torch.std(tensor)}")
    if hasattr(tensor, "grad") and tensor.grad is not None:
        print_tensor_details(tensor.grad, name=name + " Grad" if name else "Grad")


def print_vars(ref_dict, names=None):
    keys = list(ref_dict.keys())
    for name in keys:

        # only perform debugging for specific variables if we provide that
        if names is not None:
            if name not in names:
                continue

        # if a tensor, print details
        if isinstance(ref_dict[name], torch.Tensor):
            print_tensor_details(ref_dict[name], name=name)
            print("---"*5)

        # if dictionary, tuple, or list, print a map of its hiearchy
        elif isinstance(ref_dict[name], (dict, tuple, list)):
            analyze_hierarchy(ref_dict[name], name)
            print("---"*5)


def recursive(obj, string_to_print, depth, keyname=None):
    string = f"{depth * '- '}"
    if keyname is not None:
        string += f"{keyname}"
    string += f" {type(obj)}"

    if isinstance(obj, (list, tuple)):
        string_to_print += [f"{string}, length:{len(obj)}"]
        for item in obj:
            string_to_print = recursive(item, string_to_print, depth + 1)

    elif isinstance(obj, dict):
        string_to_print += [f"{string}, length:{len(obj)}"]
        for key, value in obj.items():
            string_to_print = recursive(value, string_to_print, depth + 1, keyname=key)

    else:
        string_to_print += [f"{string}"]

    return string_to_print


def analyze_hierarchy(obj, name):
    """
    decomposes a dictionary, list, or tuple into its components and prints attributes
    """
    string_to_print = []
    depth = 0
    string_to_print = recursive(obj, string_to_print, depth, keyname=name)
    string_to_print = "\n".join(string_to_print)
    print(string_to_print)