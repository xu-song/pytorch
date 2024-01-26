import sys

from torch._dynamo import register_backend
from torch._dynamo.eval_frame import is_windows


@register_backend
def inductor(*args, **kwargs):
    if is_windows():
        raise RuntimeError("Windows not yet supported for inductor")

    # do import here to avoid loading inductor into memory when it is not used
    from torch._inductor.compile_fx import compile_fx

    return compile_fx(*args, **kwargs)
