"""Observability shim around Langfuse. ON locally (visualize/debug every task as a trace tree),
OFF and fully no-op in the graded sandbox -- which is offline and which the FLYWHEEL harness itself
runs with `langfuse off`.

Contract: nothing here may ever raise or block the agent. If the keys are absent (the graded
sandbox never sets them) or the langfuse package is missing (it lives in requirements-dev only, not
the graded requirements.txt), every call degrades to a null object. Same agent code path local and
graded; only the flag flips.

Usage:
    import obs
    with obs.task("solve", instruction=instr, task_id=tid) as t:
        with obs.generation("plan", model="gemini-3-flash-preview", input=messages) as g:
            ...
            g.update(output=text, usage=usage_dict)
        with obs.span("run_code", input=code) as s:
            s.update(output=result)
"""
import contextlib
import os

_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not _ENABLED:
        return None
    try:
        from langfuse import get_client  # langfuse v3
        _client = get_client()
    except Exception:
        _client = None
    return _client


class _Null:
    """No-op span/generation: accepts any update and any nesting, does nothing."""
    def update(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Wrap:
    """Thin wrapper so the caller's .update(output=..., usage=...) maps onto the langfuse object,
    regardless of which optional kwargs are passed."""
    def __init__(self, cm):
        self._cm = cm
        self._obj = None

    def __enter__(self):
        try:
            self._obj = self._cm.__enter__()
        except Exception:
            self._obj = None
        return self

    def update(self, **kwargs):
        if self._obj is None:
            return
        try:
            self._obj.update(**kwargs)
        except Exception:
            pass

    def __exit__(self, *a):
        try:
            return self._cm.__exit__(*a)
        except Exception:
            return False


def task(name, instruction=None, task_id=None):
    c = _get_client()
    if c is None:
        return _Null()
    try:
        cm = c.start_as_current_span(name=name)
        w = _Wrap(cm)
        w.__enter__()
        w.update(input=instruction, metadata={"task_id": task_id})
        # name the enclosing trace so the dashboard list is readable
        try:
            c.update_current_trace(name=f"task:{task_id}" if task_id else name,
                                   input=instruction,
                                   metadata={"task_id": task_id}, tags=["flywheel"])
        except Exception:
            pass
        return w
    except Exception:
        return _Null()


def generation(name, model=None, input=None):
    c = _get_client()
    if c is None:
        return _Null()
    try:
        cm = c.start_as_current_generation(name=name, model=model, input=input)
        w = _Wrap(cm)
        w.__enter__()
        return w
    except Exception:
        return _Null()


def span(name, input=None):
    c = _get_client()
    if c is None:
        return _Null()
    try:
        cm = c.start_as_current_span(name=name, input=input)
        w = _Wrap(cm)
        w.__enter__()
        return w
    except Exception:
        return _Null()


def flush():
    c = _get_client()
    if c is None:
        return
    with contextlib.suppress(Exception):
        c.flush()


def enabled():
    return _get_client() is not None
