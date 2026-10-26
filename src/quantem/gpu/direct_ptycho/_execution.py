"""Cooperative cancellation at completed SSB GPU batch boundaries."""

from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar


_cancel_check: ContextVar[Callable[[], bool] | None] = ContextVar(
    "ssb_cancel_check", default=None,
)


def checkpoint() -> None:
    """Discard an obsolete preview without returning partial scientific data."""
    check = _cancel_check.get()
    if check is not None and check():
        raise CancelledError("SSB preview was superseded.")


def is_cancellable() -> bool:
    """Whether this preview needs bounded batches for a pending cancellation."""
    return _cancel_check.get() is not None


@contextmanager
def cancellation_scope(check: Callable[[], bool] | None) -> Iterator[None]:
    """Keep a viewer's cancellation predicate local to this preview call."""
    token = _cancel_check.set(check)
    try:
        checkpoint()
        yield
        checkpoint()
    finally:
        _cancel_check.reset(token)
