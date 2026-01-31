"""Utilities for wrapping async functions as sync."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

# Thread pool for running async code from sync contexts
_executor = ThreadPoolExecutor(max_workers=4)


def run_sync(coro: Awaitable[T]) -> T:
    """Run an async coroutine synchronously.

    This function handles the complexity of running async code
    from a sync context, whether or not an event loop is already running.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, we can just run directly
        return asyncio.run(coro)

    # There's a running loop, so we need to run in a separate thread
    import concurrent.futures

    def run_in_thread() -> T:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    future = _executor.submit(run_in_thread)
    return future.result()


def sync_wrapper(async_func: Callable[P, Awaitable[T]]) -> Callable[P, T]:
    """Decorator to create a synchronous wrapper for an async function.

    Args:
        async_func: The async function to wrap.

    Returns:
        A synchronous function that runs the async function.

    Example:
        @sync_wrapper
        async def fetch_data(url: str) -> dict:
            ...

        # Now fetch_data can be called synchronously
        result = fetch_data("https://example.com")
    """

    @functools.wraps(async_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return run_sync(async_func(*args, **kwargs))

    return wrapper


class SyncMixin:
    """Mixin class that provides sync versions of async methods.

    Subclasses can define async methods and access sync versions
    via a `sync` property that wraps all async methods.
    """

    _sync_proxy: "SyncProxy | None" = None

    @property
    def sync(self) -> "SyncProxy":
        """Get a proxy object with sync versions of all async methods."""
        if self._sync_proxy is None:
            self._sync_proxy = SyncProxy(self)
        return self._sync_proxy


class SyncProxy:
    """Proxy object that wraps async methods as sync."""

    def __init__(self, obj: object) -> None:
        self._obj = obj

    def __getattr__(self, name: str) -> Callable[..., object]:
        attr = getattr(self._obj, name)
        if asyncio.iscoroutinefunction(attr):
            return sync_wrapper(attr)
        return attr
