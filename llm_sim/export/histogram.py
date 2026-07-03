"""Log-linear histogram for Rezolus export — a thin adapter over `h2histogram`.

`h2histogram` is iopsystems' canonical Python implementation of the h2 histogram
(pure Python, zero deps, interoperable with Rezolus). We adapt it behind a small
stable interface (`record` / `snapshot` / `num_buckets` / `value_to_index`) so
the exporter is insulated from the library's exact 0.x API, and so our unit
tests double as a compatibility guard: they pin the bucket layout our Parquet
writer assumes (grouping_power=3, max_value_power=64 -> 496 buckets) against
whatever `h2histogram` version is installed.

Only the export stage depends on this (opt-in `[rezolus]` extra); the sim core
never imports it.
"""

from typing import List

from h2histogram import Histogram as _H2Histogram


class LogLinearHistogram:
    def __init__(self, grouping_power: int = 3, max_value_power: int = 64):
        self.grouping_power = grouping_power
        self.max_value_power = max_value_power
        self._h = _H2Histogram(grouping_power, max_value_power)
        self._config = self._h.config
        self.num_buckets = self._config.total_buckets

    def value_to_index(self, value: int) -> int:
        return self._config.value_to_index(value)

    def record(self, value: int, count: int = 1) -> None:
        self._h.record(value, count)

    def snapshot(self) -> List[int]:
        """Return a copy of the current (cumulative) dense bucket counts."""
        return list(self._h.buckets)
