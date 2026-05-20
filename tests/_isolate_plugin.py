"""Subprocess-per-test isolation plugin.

Why this exists
---------------
``pytest-xdist`` workers are long-lived processes. Module-level dicts/sets
and ContextVars leak between tests on the same worker, causing the classic
"works alone, flakes in CI" failure pattern. The historic mitigation was a
giant ``_reset_module_state`` autouse fixture in ``conftest.py`` that
manually cleared each known state bucket. That approach is fragile (every
new module-level dict needs a corresponding line in conftest) and ugly.

This plugin replaces that fixture with true process isolation: each test
runs in a fresh Python interpreter via ``multiprocessing.Process`` with the
``spawn`` start method. ``spawn`` is the only multiprocessing context that
works cross-platform (Linux, macOS, Windows) — ``fork`` is POSIX-only, and
``forkserver`` doesn't exist on Windows.

The child process:
  1. Inherits the parent's args/options via the spawn payload (pickled).
  2. Sets ``HERMES_ISOLATE_CHILD=1`` so this plugin is a no-op there
     (otherwise the child would try to spawn its own grandchildren — fork
     bomb).
  3. Runs ``runtestprotocol(item)`` for the single test it was given.
  4. Serializes test reports via ``pytest_report_to_serializable`` and
     pushes them through an ``mp.Queue`` back to the parent.

The parent:
  1. Reads serialized reports off the queue.
  2. Rehydrates them via ``pytest_report_from_serializable`` and emits via
     ``pytest_runtest_logreport`` so xdist / terminal reporter / JUnit all
     see normal-looking results.
  3. Honors ``isolate_timeout`` (ini key) — kills the child if it hangs and
     synthesizes a failure report.

Performance
-----------
Per-test overhead is dominated by ``python`` interpreter startup +
collecting just the one nodeid (~0.3-1.0 s depending on the test file's
import graph). xdist parallelism amortizes this across cores. On a 20-core
workstation a 17 k-test suite finishes in roughly the same wall time as the
old shared-state approach, with zero leakage risk.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

import _pytest.runner
import pytest


# ── Env-var sentinel ────────────────────────────────────────────────────────
# Set in every child process so the plugin disables itself there. Without
# this, every child would try to spawn its own grandchild (fork bomb) when
# pytest_runtest_protocol fires.
_CHILD_SENTINEL = "HERMES_ISOLATE_CHILD"

# Default timeout for one test. Overridable via the ``isolate_timeout``
# ini key in pyproject.toml.
_DEFAULT_TIMEOUT = 30.0


# ── pytest plugin hooks ─────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI/ini options for subprocess isolation."""
    group = parser.getgroup("hermes-isolate", "Subprocess-per-test isolation")
    group.addoption(
        "--no-isolate",
        action="store_true",
        dest="no_isolate",
        default=False,
        help=(
            "Disable subprocess-per-test isolation. Tests will run in the "
            "xdist worker process, sharing module-level state with siblings."
        ),
    )
    parser.addini(
        "isolate_timeout",
        "Per-test timeout in seconds for the isolation subprocess. "
        "If the child exceeds this it is killed and a failure report is "
        "synthesized.",
        type="string",
        default=str(_DEFAULT_TIMEOUT),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Sanity-check the env on the parent side."""
    # Only run on the *parent* — children inherit envvar and short-circuit.
    if os.environ.get(_CHILD_SENTINEL) == "1":
        return

    # spawn is mandatory: it's the only context that works on Windows.
    try:
        mp.get_context("spawn")
    except (ValueError, AttributeError) as exc:  # pragma: no cover
        raise pytest.UsageError(
            "hermes-isolate: multiprocessing 'spawn' context is unavailable. "
            f"({exc})"
        ) from exc


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[bool]:
    """Intercept test execution; run in a spawned subprocess.

    Returning ``True`` tells pytest "I handled this — skip the normal
    runtestprotocol". Returning ``None`` falls through to the default.
    """
    # Disable in child processes (fork-bomb prevention) and when user
    # passed --no-isolate.
    if os.environ.get(_CHILD_SENTINEL) == "1":
        return None
    if item.config.getoption("no_isolate", default=False):
        return None

    timeout = _parse_timeout(item.config.getini("isolate_timeout"))

    reports = _run_in_spawned_subprocess(item, timeout)

    # Emit reports through the normal pytest channels so xdist + terminal
    # reporter + junit etc. all see them as if the test ran normally.
    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    for rep in reports:
        ihook.pytest_runtest_logreport(report=rep)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


# ── Internal: subprocess machinery ──────────────────────────────────────────


def _parse_timeout(raw: Any) -> float:
    """Coerce the ini value to a float; fall back to the default."""
    try:
        value = float(raw)
        if value <= 0:
            raise ValueError
        return value
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT


@contextmanager
def _suspend_sigalrm() -> Iterator[None]:
    """Temporarily disarm any pending SIGALRM in this thread.

    pytest-timeout (when ``--timeout-method=signal``) installs a SIGALRM
    handler and arms ``ITIMER_REAL`` for each test. That timer is meant to
    interrupt the test code, but our isolation hook intercepts
    ``pytest_runtest_protocol`` before the test runs in this process — so
    if the SIGALRM fires while we're blocked on ``proc.join``, it lands in
    our parent process instead of the test, raising ``Failed: Timeout``
    from inside the hook and crashing the xdist worker.

    Suspend the timer + handler for the duration of the wait, then restore
    them. Best-effort: on platforms without SIGALRM (Windows) this is a
    no-op. We restore the previous handler on exit so pytest-timeout's
    own ``cancel_timeout`` still works post-isolation.
    """
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        # Windows path — no SIGALRM, no risk of the bug.
        yield
        return

    # Disarm any pending alarm and remember how much time was left so we
    # can restore it. ``setitimer`` returns the remaining seconds (0.0
    # means no timer was armed).
    prev_remaining, prev_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)
    prev_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
    try:
        yield
    finally:
        # Restore the previous handler first so any re-armed timer fires
        # into the right place, not SIG_DFL (which would terminate us).
        try:
            signal.signal(signal.SIGALRM, prev_handler)
        except Exception:
            pass
        if prev_remaining > 0:
            try:
                signal.setitimer(
                    signal.ITIMER_REAL, prev_remaining, prev_interval
                )
            except Exception:
                pass


def _run_in_spawned_subprocess(
    item: pytest.Item, timeout: float
) -> List[pytest.TestReport]:
    """Spawn a new Python process, run the test there, collect reports.

    The child is given:
      * The exact nodeid we want it to run.
      * The full ``sys.argv`` of the parent pytest invocation, minus a few
        flags that would mess with the child (xdist, our own --no-isolate).
      * The parent's working directory.

    The child runs ``pytest.main([...nodeid])`` against a one-shot in-process
    plugin that captures reports into a list and pushes them through an
    ``mp.Queue``.
    """
    ctx = mp.get_context("spawn")
    result_q: "mp.Queue[Tuple[str, Any]]" = ctx.Queue()

    rootdir = str(item.config.rootpath)
    nodeid = item.nodeid

    proc = ctx.Process(
        target=_child_entrypoint,
        args=(result_q, rootdir, nodeid, os.getpid(), timeout),
        # Helpful in process listings; spawn ignores name visually on most
        # systems but JUnit output uses it for pid attribution.
        name=f"pytest-isolate:{nodeid}",
        daemon=False,
    )
    proc.start()

    timed_out = False
    collected: List[Any] = []
    try:
        # The parent process owns this join. pytest-timeout (when active)
        # has armed a SIGALRM via setitimer that fires after the per-test
        # timeout — but in our world, "the test" is running in the child,
        # not in this parent thread. If we let the SIGALRM fire here, it
        # raises ``Failed: Timeout`` inside ``proc.join`` and crashes the
        # xdist worker (the worker can't recover from a hook handler that
        # raised mid-protocol). We enforce the timeout ourselves via the
        # ``timeout`` argument to ``proc.join``, so suspend pytest-timeout's
        # alarm for the duration of the join and restore it after.
        #
        # We pad the parent-side timeout by 5s so the child's pytest-timeout
        # has time to fire first and produce a clean ``Failed: Timeout``
        # report. If that fails (e.g. the test is hung in a C extension
        # that ignores SIGALRM), the parent kill kicks in as a backstop.
        with _suspend_sigalrm():
            proc.join(timeout + 5.0)
        if proc.is_alive():
            timed_out = True
            proc.terminate()
            proc.join(5)
            if proc.is_alive():  # pragma: no cover — terminate refused
                proc.kill()
                proc.join()

        # Drain the queue. There may be zero items on timeout/crash.
        while True:
            try:
                kind, payload = result_q.get_nowait()
            except Exception:  # queue.Empty raises across mp contexts
                break
            if kind == "report":
                collected.append(payload)
            elif kind == "error":
                # Child raised before producing reports — synthesize a
                # failure below.
                collected.append(("__error__", payload))
    finally:
        result_q.close()
        result_q.join_thread()

    # Convert serializable reports back to live TestReport instances.
    reports: List[pytest.TestReport] = []
    for payload in collected:
        if isinstance(payload, tuple) and payload and payload[0] == "__error__":
            reports.append(_synthesize_crash_report(item, payload[1]))
            continue
        rep = item.config.hook.pytest_report_from_serializable(
            config=item.config, data=payload
        )
        reports.append(rep)

    if timed_out and not reports:
        reports.append(
            _synthesize_crash_report(
                item, f"Test exceeded isolate_timeout ({timeout:.1f}s)"
            )
        )
    elif not reports and proc.exitcode not in (0, None):
        reports.append(
            _synthesize_crash_report(
                item, f"Child exited with code {proc.exitcode} and no reports"
            )
        )

    return reports


def _synthesize_crash_report(item: pytest.Item, message: str) -> pytest.TestReport:
    """Build a failed TestReport when the child died before sending one."""
    longrepr = f"hermes-isolate: {message}"
    call_info = _pytest.runner.CallInfo.from_call(
        lambda: (_ for _ in ()).throw(RuntimeError(longrepr)), "call"
    )
    return _pytest.runner.pytest_runtest_makereport(item, call_info)


# ── Internal: child entrypoint ──────────────────────────────────────────────
# Module-level so ``spawn`` can pickle it (lambdas/closures don't pickle).


def _child_entrypoint(
    result_q: "mp.Queue", rootdir: str, nodeid: str, parent_pid: int, timeout: float
) -> None:
    """Run a single test in the spawned process and ship reports home.

    Must be importable at the top level for ``spawn`` to find it via
    ``mp.spawn``'s pickling machinery.
    """
    os.environ[_CHILD_SENTINEL] = "1"
    os.environ["PYTEST_PARENT_PID"] = str(parent_pid)

    try:
        # Move into the rootdir so relative test paths resolve correctly.
        os.chdir(rootdir)

        # Use pytest.main in-process. ``-p no:cacheprovider`` keeps the
        # child from racing on .pytest_cache. ``-p no:xdist`` is critical:
        # the child must NOT try to spawn its own xdist workers.
        # ``--no-header --no-summary -q`` keeps output noise down (the
        # parent re-renders reports anyway via logreport).
        #
        # We pass ``--timeout`` explicitly because ``-o addopts=`` purges
        # the parent's addopts (which carry the timeout config). Without
        # it, a hanging test would only be caught by the parent-side
        # ``proc.join`` timeout, surfaced as a generic SIGTERM crash. With
        # it, pytest-timeout fires inside the child and produces a clean
        # "Timeout" failure report.
        argv = [
            nodeid,
            "-p",
            "no:cacheprovider",
            "-p",
            "no:xdist",
            "-p",
            "no:hermes_isolate",  # belt-and-suspenders against re-entry
            "--no-header",
            "-q",
            "-o",
            "addopts=",  # purge parent's addopts (would re-add -n auto)
            f"--timeout={timeout:.1f}",
            "--timeout-method=signal",
        ]

        collector = _ReportCollector(result_q)
        # Note: we DO want the child to load tests/conftest.py — that's
        # what provides _hermetic_environment + _live_system_guard. The
        # in-process plugin just intercepts reports.
        exit_code = pytest.main(argv, plugins=[collector])

        # If pytest.main exited cleanly with no reports captured, surface
        # an explicit error so the parent doesn't think the test passed.
        if not collector.sent_any and exit_code != 0:
            result_q.put(
                ("error", f"child pytest.main exited {exit_code} without reports")
            )
    except BaseException as exc:  # noqa: BLE001 — must catch everything
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            result_q.put(("error", f"child crashed: {tb}"))
        except Exception:
            # Queue may already be closed if the parent gave up.
            pass


class _ReportCollector:
    """In-process pytest plugin that ships TestReport objects via queue."""

    def __init__(self, result_q: "mp.Queue") -> None:
        self._q = result_q
        self._config: Optional[pytest.Config] = None
        self.sent_any = False

    @pytest.hookimpl
    def pytest_configure(self, config: pytest.Config) -> None:
        self._config = config

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:  # noqa: D401
        # Serialize through the config's hook so plugins that extend the
        # serialization (xdist, etc.) participate.
        data: Any
        if self._config is not None:
            try:
                data = self._config.hook.pytest_report_to_serializable(
                    config=self._config, report=report
                )
            except Exception:
                data = None
        else:
            data = None

        if data is None:
            # Last-resort minimal serialization. Better than dropping the
            # report and silently "passing" a broken test.
            data = {
                "$report_type": "TestReport",
                "nodeid": report.nodeid,
                "when": report.when,
                "outcome": report.outcome,
                "longrepr": str(report.longrepr) if report.longrepr else None,
            }

        try:
            self._q.put(("report", data))
            self.sent_any = True
        except Exception:
            pass


# ── Identification helpers (for testing the plugin itself) ─────────────────


__all__ = [
    "pytest_addoption",
    "pytest_configure",
    "pytest_runtest_protocol",
]
