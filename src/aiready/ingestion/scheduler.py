"""Scheduler: timed runs and change events.

Schedules (per source, ``schedule:``):
  * ``every 15m`` / ``interval:15m`` / ``every 2h`` / ``every 30s`` / ``every 1d``;
  * ``@hourly``, ``@daily``, ``@weekly``;
  * five-field cron ``m h dom mon dow`` with ``*``, ``*/n``, ``a-b``, ``a-b/n`` and lists (``1,15,30``).

Event-driven runs: ``trigger(source)`` (from the API, a webhook or a change notification) queues an
immediate run; change capture and the file crawler can also put document messages on the work queue
directly. Last-run times live in the DocStore so a restarted scheduler does not re-run everything.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any

from aiready.core.stores import get_docstore

log = logging.getLogger(__name__)
NS_SCHED = "ingestion.schedule"
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_ALIASES = {"@hourly": "0 * * * *", "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@weekly": "0 0 * * 0"}


def _field(spec: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            step = int(s)
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
        else:
            a = b = int(part)
        if a < lo or b > hi or a > b:
            raise ValueError(f"cron value {part} out of range {lo}-{hi}")
        out.update(range(a, b + 1, step))
    return out


class Schedule:
    """A parsed schedule. ``next_after(t)`` gives the next due time (epoch seconds, local time for cron)."""

    def __init__(self, expr: str):
        self.expr = expr.strip()
        e = _ALIASES.get(self.expr, self.expr)
        m = re.fullmatch(r"(?:every\s+|interval:)\s*(\d+)\s*([smhd])", e, flags=re.IGNORECASE)
        self.interval: float | None = None
        self.cron: tuple[set[int], ...] | None = None
        self._dom_any = self._dow_any = True
        if m:
            self.interval = int(m.group(1)) * _UNITS[m.group(2).lower()]
            if self.interval <= 0:
                raise ValueError("interval must be positive")
            return
        parts = e.split()
        if len(parts) != 5:
            raise ValueError(f"invalid schedule '{expr}': use 'every 15m', '@daily' or a 5-field cron expression")
        self.cron = (_field(parts[0], 0, 59), _field(parts[1], 0, 23), _field(parts[2], 1, 31),
                     _field(parts[3], 1, 12), {d % 7 for d in _field(parts[4], 0, 7)})
        self._dom_any, self._dow_any = parts[2] == "*", parts[4] == "*"

    def _cron_match(self, dt: datetime) -> bool:
        assert self.cron is not None
        minute, hour, dom, mon, dow = self.cron
        if dt.minute not in minute or dt.hour not in hour or dt.month not in mon:
            return False
        cron_dow = (dt.weekday() + 1) % 7  # cron: 0 = Sunday
        dom_ok, dow_ok = dt.day in dom, cron_dow in dow
        if self._dom_any or self._dow_any:  # standard cron: either restricted field may match
            return dom_ok and dow_ok
        return dom_ok or dow_ok

    def next_after(self, t: float) -> float:
        if self.interval is not None:
            return t + self.interval
        dt = datetime.fromtimestamp(t).replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(366 * 24 * 60):
            if self._cron_match(dt):
                return dt.timestamp()
            dt += timedelta(minutes=1)
        raise ValueError(f"schedule '{self.expr}' never fires")

    def is_due(self, last_run: float | None, now: float) -> bool:
        if last_run is None:
            return True
        return self.next_after(last_run) <= now


class Scheduler:
    """Fires scheduled and triggered runs. ``process=False`` leaves the documents to the worker pool and
    completes (verifies) each run once the work queue is drained."""

    def __init__(self, pipeline: Any, *, process: bool = True, tick_seconds: float | None = None):
        self.pipeline = pipeline
        self.process = process
        self.tick = tick_seconds or pipeline.cfg.scheduler_tick_seconds
        self.schedules: dict[str, Schedule] = {}
        for s in pipeline.cfg.sources:
            if s.enabled and s.schedule:
                self.schedules[s.name] = Schedule(s.schedule)
        self._events: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._pending_runs: list[str] = []

    def last_run(self, source: str) -> float | None:
        v = get_docstore().get(NS_SCHED, source)
        return float(v["last_run"]) if v else None

    def _record(self, source: str, when: float) -> None:
        get_docstore().put(NS_SCHED, source, {"last_run": when})

    def due(self, now: float | None = None) -> list[str]:
        now = now or time.time()
        return [name for name, sch in self.schedules.items() if sch.is_due(self.last_run(name), now)]

    async def trigger(self, source: str, reason: str = "event") -> None:
        """Queue an immediate run (webhook, change notification, operator)."""
        self.pipeline.source(source)  # unknown sources fail fast
        await self._events.put((source, reason))

    async def _run(self, source: str, trigger: str) -> None:
        self._record(source, time.time())
        try:
            run = await self.pipeline.run_source(source, trigger=trigger, process=self.process)
            if not self.process:
                self._pending_runs.append(run.run_id)
        except Exception as exc:  # one failing source must not stop the others
            log.error("scheduled run of %s failed: %s", source, exc)

    async def tick_once(self, now: float | None = None) -> list[str]:
        started = []
        while not self._events.empty():
            source, reason = self._events.get_nowait()
            await self._run(source, reason)
            started.append(source)
        for name in self.due(now):
            if name not in started:
                await self._run(name, "schedule")
                started.append(name)
        if self._pending_runs and await self.pipeline.queue.depth(self.pipeline.topic) == 0:
            pending, self._pending_runs = self._pending_runs, []
            for run_id in pending:
                try:
                    await self.pipeline.finish_run(run_id)
                except Exception as exc:
                    log.error("finishing run %s failed: %s", run_id, exc)
        return started

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.tick_once()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.tick)
            except TimeoutError:
                pass
