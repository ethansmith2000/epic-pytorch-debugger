import contextlib
import functools
import pdb
import traceback
import sys
from functools import wraps
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
import torch


from torch.utils._python_dispatch import TorchDispatchMode, _push_mode, _pop_mode
from torch.utils._pytree import tree_map
import traceback
import sys
import torch
import gc

import re


class EpicDebugger(TorchDispatchMode):
    def __init__(self, debug_always=False, enabled=True, do_pdb=True, exception_fn=None, normal_debug_fn=None, run_trace=True, **debug_kwargs):
        """
        debug_always: if True, will always run the normal_debug_fn
        enabled: completely enable/disable the debugger, useful if you're toggling from a config
        do_pdb: if True, will run pdb on exception
        exception_fn: function to run on exception
        normal_debug_fn: function that will always be ran if debug_always is True
        run_trace: if True, will run trace_assignments, this is needed if you need to keep track of variable and tensor names, but can be slow
        debug_kwargs: kwargs to pass to the exception_fn and normal_debug_fn
        """

        super().__init__()
        self.debug_always = debug_always
        self.enabled = enabled
        self.do_pdb = do_pdb
        self.exception_fn = exception_fn
        self.normal_debug_fn = normal_debug_fn
        self.run_trace = run_trace
        self.debug_kwargs = debug_kwargs
        self.oldtrace = None

        self.refs_dict = {}


    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        parent_names = []
        
        def gather_names(x):
            if hasattr(x, "tensor_name"):
                parent_names.append(x.tensor_name)
        
        tree_map(gather_names, args)
        tree_map(gather_names, kwargs)

        out = func(*args, **kwargs)

        if len(parent_names) > 0:
            parent_names = "+".join(parent_names)
            parent_names = "(" + parent_names + ")"

            def name(x):
                if isinstance(x, torch.Tensor):
                    if hasattr(x, "tensor_name"):
                        x.tensor_name = parent_names + "\nV\n"
                    else:
                        x.tensor_name = parent_names
                return x
            
            return tree_map(name, out)
        
        else:
            return out

    def __enter__(self):
        _push_mode(self, self.__dict__.get("_dispatch_key", None))
        if self.run_trace:
            self.oldtrace = sys.gettrace()
            # capture self to use in local_trace
            self_ref = self
            def local_trace(frame, event, arg):
                # call trace_assignments on self_ref to maintain context scope
                if event == 'line':
                    self_ref.trace_assignments(frame)
                return local_trace
            sys.settrace(local_trace)
        return self


    def trace_assignments(self, frame):
        relevant_frame = frame.f_back
        local_vars = relevant_frame.f_locals.items()
        # add refs to the refs_dict
        # if the variable begins and ends with __, it's a python internal variable, so we skip it
        # or any of these listed
        for name, obj in local_vars:
            if re.match(r"__.*__", name) or name in ["_ih", '_oh', '_dh', 'In', 'Out', 'get_ipython', 'exit', 'quit', 'open', '_', '__', '___', '_i', '_ii', '_iii', '_i1',"_i2"]:
                continue
            self.refs_dict[name] = obj

        tensors = {name: obj for name, obj in local_vars if isinstance(obj, torch.Tensor)}
        for name, tensor in tensors.items():
            if not hasattr(tensor, 'tensor_name'):
                setattr(tensor, 'tensor_name', name)
            # replace ending, theres a lot of random vars that end up here that we don't want
            elif re.match(r".*\nV\n.*", tensor.tensor_name):
                base = tensor.tensor_name.split("\nV\n")[:-1]
                setattr(tensor, 'tensor_name', "\nV\n".join(base) + "\nV\n" + name)
            elif tensor.tensor_name.endswith("\nV\n"):
                setattr(tensor, 'tensor_name', tensor.tensor_name + name)


    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.enabled:
            traceback.print_exc()
            if self.exception_fn is not None:
                print("*"*10 + " BEGIN EXCEPTION_FN " + "*"*10)
                self.exception_fn(self.refs_dict, **self.debug_kwargs)
                print("*"*10 + " END EXCEPTION_FN " + "*"*10)
            if self.do_pdb:
                # garbage collect all the objects so we can look at them, can also examine refs_dict
                # objs = gc.get_objects()
                pdb.set_trace()

        if self.debug_always and self.enabled:
            if self.normal_debug_fn is not None:
                print("*"*10 + " BEGIN DEBUG_FN " + "*"*10)
                self.normal_debug_fn(self.refs_dict, **self.debug_kwargs)
                print("*"*10 + " END DEBUG_FN " + "*"*10)

        if self.run_trace:
            sys.settrace(self.oldtrace)

        mb_dk_or_mode_key = self.__dict__.get("_dispatch_key", None)
        if mb_dk_or_mode_key is None:
            # Today, mode keys are not used at all in the per-dispatch-key-mode logic (for pre-dispatch)
            # We should probably revisit this.
            mb_dk_or_mode_key = self.__dict__.get("_mode_key", None)
        _pop_mode(mb_dk_or_mode_key)



# if you want to use as a decorator over an entire function
def epic_debugger_decorator(enabled=True, debug_always=False, do_pdb=True, exception_fn=None, normal_debug_fn=None, **debug_kwargs):
    def debug_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with EpicDebugger(debug_always=debug_always, enabled=enabled, do_pdb=do_pdb, exception_fn=exception_fn, normal_debug_fn=normal_debug_fn, **debug_kwargs):
                return func(*args, **kwargs)
        return wrapper
    return debug_decorator





