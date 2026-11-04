# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
"""Source contracts for the audited native execution paths, not version strings."""
import ast
import hashlib
import inspect
import textwrap
from functools import lru_cache
from .contracts import CONTRACTS


def digest(function):
    source = textwrap.dedent(inspect.getsource(function))
    return hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()


@lru_cache(maxsize=64)
def verified(function, contract):
    try:
        return digest(function) == CONTRACTS[contract]
    except (OSError, TypeError, KeyError, SyntaxError):
        return False


def native_model(inner):
    return (type(inner).__module__ == "comfy.ldm.krea2.model" and
            type(inner).__name__ == "SingleStreamDiT" and
            all(hasattr(inner, x) for x in ("last", "tmlp", "blocks", "patch", "channels", "tdim")))


def model_contract(inner):
    if not native_model(inner):
        return False
    if any(k in inner.__dict__ for k in ("forward", "_forward", "process_img", "_unpack_context")):
        return False
    methods = {"model._forward": type(inner)._forward, "model.forward": type(inner).forward,
               "model.process_img": type(inner).process_img,
               "model._unpack_context": type(inner)._unpack_context,
               "last.forward": type(inner.last).forward}
    if "forward" in inner.last.__dict__:
        return False
    if not all(verified(fn, key) for key, fn in methods.items()):
        return False
    for module in inner.modules():
        if "forward" in module.__dict__ or module._forward_hooks or module._forward_pre_hooks:
            return False
        if type(module).__module__ == "comfy.ldm.krea2.model":
            if not verified(type(module).forward, type(module).__name__ + ".forward"):
                return False
    return True
