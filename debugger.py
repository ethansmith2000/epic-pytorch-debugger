import contextlib
import functools
import pdb
import traceback


# if you want to use as a context manager over certain sections of a function
@contextlib.contextmanager
def epic_debugger(debug_always=False, enabled=True, do_pdb=True, exception_fn=None, normal_debug_fn=None, **debug_kwargs):
    """
    :param debug_always: whether to run a portion of code regardless of whether an exception is raised
    :param enabled: whether to run the debugger at all, so can be easily enabled from an args/config without needing to change code
    :param do_pdb: activate pythons debugger if an exception is raised
    :param exception_fn: function to run if an exception is raised
    :param normal_debug_fn: function to run if no exception is raised
    """
    # default behavior
    try:
        yield

    # if there is an error
    except Exception as e:
        if enabled:
            traceback.print_exc()
            if exception_fn is not None:
                print("*"*10 + " BEGIN EXCEPTION_FN " + "*"*10)
                exception_fn(**debug_kwargs)
                print("*"*10 + " END EXCEPTION_FN " + "*"*10)
            if do_pdb:
                pdb.set_trace()
        raise e

    # if debug_always is enabled
    finally:
        if debug_always and enabled:
            if normal_debug_fn is not None:
                print("*"*10 + " BEGIN DEBUG_FN " + "*"*10)
                normal_debug_fn(**debug_kwargs)
                print("*"*10 + " END DEBUG_FN " + "*"*10)



# if you want to use as a decorator over an entire function
def epic_debugger_decorator(enabled=True, debug_always=False, do_pdb=True, exception_fn=None, normal_debug_fn=None, **debug_kwargs):
    """
    :param enabled: whether to run a portion of code regardless of whether an exception is raised
    :param debug_always:  whether to run the debugger at all, so can be easily enabled from an args/config without needing to change code
    :param do_pdb: activate pythons debugger if an exception is raised
    :param exception_fn: function to run if an exception is raised
    :param normal_debug_fn: function to run if no exception is raised
    """
    def debug_decorator(func):
        # a wrapper to go around your function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # run the actual function
                result = func(*args, **kwargs)
                # if debug_always is enabled run this portion
                if debug_always and enabled:
                    if normal_debug_fn is not None:
                        print("*"*10 + " BEGIN DEBUG_FN " + "*"*10)
                        normal_debug_fn(**debug_kwargs)
                        print("*"*10 + " END DEBUG_FN " + "*"*10)

                # return the output of the original function
                return result

            # if there is an error
            except Exception as e:
                if enabled:
                    traceback.print_exc()
                    if exception_fn is not None:
                        print("*"*10 + " BEGIN EXCEPTION_FN " + "*"*10)
                        exception_fn(**debug_kwargs)
                        print("*"*10 + " END EXCEPTION_FN " + "*"*10)
                    if do_pdb:
                        pdb.set_trace()
                # Re-raise the exception after handling
                raise e

        return wrapper
    return debug_decorator


