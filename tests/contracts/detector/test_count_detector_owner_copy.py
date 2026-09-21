"""Guard: no private device owner may escape the count detector adapter.

F1 shipped because ``CountDetectorCompute.mean_dp`` handed callers the raw
Metal/ANS owner object (``MPSANSArray``) instead of a copied NumPy array; the
same class of bug can reappear in any method that forgets ``_copy_output``.
Native outputs instead use ``_native_output`` to transfer private ownership
into an independently owned device tensor, without a host download. These
checks cover both explicit boundaries; hardware tests verify the device result.
"""

import ast
from pathlib import Path

from quantem.gpu.detector.backends import counts


def _module_tree():
    return ast.parse(Path(counts.__file__).read_text())


def _copy_output_calls(tree):
    """Every host-copy or native-ownership transfer and its input expression."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"_copy_output", "_native_output"}
            and len(node.args) == 1
        ):
            yield node, node.args[0]


def _expression_names(node):
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def test_every_device_call_is_copied_before_it_is_returned():
    """A bare ``*_device(...)`` call must be an argument of ``_copy_output``."""
    tree = _module_tree()
    copied = {
        id(argument)
        for _, argument in _copy_output_calls(tree)
    }
    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.endswith("_device")
            and id(node) not in copied
        ):
            offenders.append(f"line {node.lineno}: {node.func.attr}(...)")
    assert not offenders, (
        "Device owners must be read through _copy_output so the private buffer "
        "is released and callers receive NumPy, not an owner object:\n  "
        + "\n  ".join(offenders)
    )


def test_every_device_getattr_is_read_through_copy_output():
    """A ``getattr(source, "x_device")`` alias must be called inside a copy.

    ``mean_dp`` reaches its device method through ``getattr``, so the lexical
    call check above cannot see the call site; require the alias itself to be
    passed to ``_copy_output``.
    """
    tree = _module_tree()
    offenders = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        aliases = {}
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "getattr"
                and len(value.args) >= 2
                and isinstance(value.args[1], ast.Constant)
                and isinstance(value.args[1].value, str)
                and value.args[1].value.endswith("_device")
            ):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = value.args[1].value
        if not aliases:
            continue
        guarded = set()
        for _, argument in _copy_output_calls(function):
            guarded |= _expression_names(argument)
        for alias, method in aliases.items():
            if alias not in guarded:
                offenders.append(
                    f"line {function.lineno}: {function.name} calls "
                    f"{method}() via {alias!r} outside _copy_output"
                )
    assert not offenders, (
        "Device-owned results must be copied and released before returning:\n  "
        + "\n  ".join(offenders)
    )
