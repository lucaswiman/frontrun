"""Shared helpers for monkey-patching methods and restoring originals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PatchRecord = tuple[Any, str, Any]


def wrap_method_metadata(wrapper: Any, original: Any, *, name: str | None = None) -> Any:
    """Copy method metadata from *original* onto *wrapper*.

    The project only relies on ``__name__`` and ``__qualname__`` staying stable
    for introspection and tests, so we keep the helper minimal.
    """
    wrapper.__name__ = name or getattr(original, "__name__", getattr(wrapper, "__name__", "patched"))
    wrapper.__qualname__ = getattr(original, "__qualname__", wrapper.__name__)
    return wrapper


def patch_method(
    target: Any,
    attr_name: str,
    *,
    originals: dict[tuple[Any, str], Any],
    patches: list[PatchRecord],
    make_wrapper: Callable[[Any], Any],
) -> bool:
    """Patch ``target.attr_name`` once and remember the original.

    Returns ``True`` when a patch was installed and ``False`` when the target
    was missing or already patched.
    """
    key = (target, attr_name)
    if key in originals:
        return False

    original = getattr(target, attr_name, None)
    if original is None:
        return False

    setattr(target, attr_name, make_wrapper(original))
    originals[key] = original
    patches.append((target, attr_name, original))
    return True


def restore_patches(patches: list[PatchRecord]) -> None:
    """Restore a sequence of ``(target, attr_name, original)`` patches."""
    for target, attr_name, original in patches:
        setattr(target, attr_name, original)
