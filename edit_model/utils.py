import inspect
from typing import Dict


class cached_property:
    """
    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.

    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def get_method_args_dict(func, locals) -> Dict:
    arg_spec = inspect.getfullargspec(func)
    args = dict()
    
    for arg_name in filter(lambda x: x not in ('self'), arg_spec.args):
        arg_val = locals[arg_name]
        if arg_val is None or isinstance(arg_val, (list, dict, set, str, int, float, bool)):
            args[arg_name] = locals[arg_name]

    return args
