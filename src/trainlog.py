"""Reusable logging + checkpointing for the hypernetwork training scripts.

Two small pieces plus a bundle:

* ``MetricLogger``  -- prints a metrics line on an interval and appends every
                       step to a CSV (so loss curves survive the run).
* ``Checkpointer``  -- atomically pickles training state on an interval into a
                       distinct ``<tag>_step<N>.pkl`` per save (never overwritten);
                       resume reloads the highest-numbered one.
* ``TrainMonitor``  -- bundles both behind a single ``record(...)`` call.

These mirror main.py's ``--save_every`` / ``--eval_every`` convention but work
for the epoch-based auto-encoder / best-response trainers. They are deliberately
state-agnostic: the checkpoint payload is whatever dict the caller passes
(params, opt_state, rng, ...), so every script can reuse the same logic.

Note: this is NOT used by main.py, whose per-step ``p1_state.pkl`` / ``p2_state``
layout is the on-disk data format consumed by the dataset loader and must not
change.
"""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _fmt(v: float) -> str:
    """Compact fixed/scientific formatting that stays readable across scales."""
    av = abs(v)
    if v == 0 or 1e-3 <= av < 1e6:
        return f"{v:.4f}"
    return f"{v:.3e}"


# ---------------------------------------------------------------------------
# Metric logging (console + CSV)
# ---------------------------------------------------------------------------

@dataclass
class MetricLogger:
    out_dir: Path
    tag: str
    console_every: int = 1            # print every N steps (also always at end)
    csv: bool = True
    step_name: str = "epoch"
    _writer: Any = field(default=None, init=False, repr=False)
    _fh: Any = field(default=None, init=False, repr=False)
    _keys: list[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: dict[str, float], *,
            force: bool = False, append_csv: bool = True) -> None:
        """Append a CSV row and (every console_every, or if forced) print."""
        if self.csv and append_csv:
            self._csv_row(step, metrics)
        if force or step % self.console_every == 0:
            body = "  ".join(f"{k}={_fmt(float(v))}" for k, v in metrics.items())
            print(f"{self.step_name} {step:4d}  {body}")

    def _csv_row(self, step: int, metrics: dict[str, float]) -> None:
        if self._writer is None:
            self._keys = list(metrics.keys())
            # Resume-friendly: append if the log already exists with a header.
            path = self.out_dir / f"{self.tag}_log.csv"
            exists = path.exists() and path.stat().st_size > 0
            self._fh = open(path, "a", newline="")
            self._writer = csv.writer(self._fh)
            if not exists:
                self._writer.writerow([self.step_name, *self._keys])
        self._writer.writerow([step, *(float(metrics[k]) for k in self._keys)])
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = self._writer = None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

@dataclass
class Checkpointer:
    out_dir: Path
    tag: str
    every: int = 0                    # 0 disables periodic saving

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(self, step: int, state: dict | Callable[[], dict]) -> Path | None:
        """Save if periodic saving is on and ``step`` is on the interval.

        ``state`` may be a dict or a zero-arg callable returning one (the
        callable is only invoked when a save actually happens, so building the
        payload costs nothing on skipped steps).
        """
        if self.every <= 0 or step % self.every != 0:
            return None
        return self.save(step, state)

    def save(self, step: int, state: dict | Callable[[], dict]) -> Path:
        """Pickle to a distinct <tag>_step<step>.pkl (never overwrites earlier
        steps; re-saving the same step is idempotent)."""
        payload = {"step": step, **(state() if callable(state) else state)}
        path = self.out_dir / f"{self.tag}_step{step}.pkl"
        # Atomic: write to a temp file in the same dir, then rename over target.
        tmp = self.out_dir / f"{path.name}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        tmp.replace(path)
        return path

    def _checkpoints(self) -> list[tuple[int, Path]]:
        prefix = f"{self.tag}_step"
        found = []
        for p in self.out_dir.glob(f"{prefix}*.pkl"):
            try:
                found.append((int(p.name[len(prefix):-len(".pkl")]), p))
            except ValueError:
                pass
        return sorted(found)

    def load_latest(self) -> dict | None:
        """Return the highest-numbered checkpoint payload, or None."""
        ckpts = self._checkpoints()
        if not ckpts:
            return None
        with open(ckpts[-1][1], "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class TrainMonitor:
    """Convenience wrapper: one ``record`` call logs metrics and checkpoints."""
    out_dir: Path
    tag: str
    log_every: int = 1
    ckpt_every: int = 0
    step_name: str = "epoch"

    def __post_init__(self) -> None:
        self.logger = MetricLogger(self.out_dir, self.tag,
                                   console_every=self.log_every,
                                   step_name=self.step_name)
        self.ckpt = Checkpointer(self.out_dir, self.tag, every=self.ckpt_every)

    def record(self, step: int, metrics: dict[str, float],
               state: dict | Callable[[], dict] | None = None, *,
               force_log: bool = False) -> None:
        self.logger.log(step, metrics, force=force_log)
        if state is not None:
            self.ckpt.maybe_save(step, state)

    def resume(self) -> dict | None:
        """Return the latest checkpoint payload (or None) for resuming."""
        return self.ckpt.load_latest()

    def close(self) -> None:
        self.logger.close()
