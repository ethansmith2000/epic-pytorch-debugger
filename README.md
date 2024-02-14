# epic-pytorch-debugger

## A pytorch debugger with a focus on extremely minimal code changes needed to implement. 

made with love at my time at LeonardoAI after getting tired of putting print statements, pdb.set_trace(), and try/excepts everywhere.

To use, simply put this decorator around any function
```python
@epic_pytorch_debugger_decorator(debug_always=False, enabled=True, do_pdb=True, exception_fn=None, normal_debug_fn=None, **kwargs)
```

or use the context manager for use around certain regions of your code
```python
with EpicPytorchDebugger(debug_always=False, enabled=True, do_pdb=True, exception_fn=None, normal_debug_fn=None, **kwargs):
    do_thing()
```

debug_always: whether or not to run debug_fn, regardless of whether the wrapped code throws an exception or not
enabled: entirely enable/disable debugger functionality
do_pdb: enable python's debugger upon exception
exception_fn: function to run when hitting an exception
normal_debug_fn: function to run every time

both exception_fn and normal_debug_fn should take on the following format

```python
def exception_fn(ref_dict, keyword_arg1=None, keyword_arg2=None):
    ...
```
ref_dict will be a dictionary of references to variables created during the function so we can access them
any other keyword arguments can be passed to the decorator and will be fed through to both your exception_fn and normal_debug_fn

see the provided notebook for examples.

TODO
- [ ] fix the computation graph
- [ ] add support for integrating hooks within the debugger
- [ ] see what happens when used during a backward pass (currently untested)





