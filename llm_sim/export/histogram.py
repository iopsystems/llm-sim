"""Pure-Python port of iopsystems `histogram::Config` (log-linear bucketing).

Faithful to histogram/src/config.rs so the bucket layout is byte-compatible with
Rezolus recordings. Rezolus's native default is grouping_power=3,
max_value_power=64 -> 496 buckets, which is what the Rezolus viewer expects when
it reads `<metric>:buckets` List<UInt64> columns.

Bucketing (grouping_power g, cutoff_power = g+1, cutoff_value = 2^(g+1)):
  - values [0, cutoff_value)  -> linear region, index == value
  - values >= cutoff_value    -> log-linear, 2^g sub-divisions per octave

No numpy/pyarrow: this is the dependency-free core, unit-tested against the Rust
formula. The Parquet writer lives in export/rezolus.py.
"""

from typing import List


class LogLinearHistogram:
    def __init__(self, grouping_power: int = 3, max_value_power: int = 64):
        self.grouping_power = grouping_power
        self.max_value_power = max_value_power
        self.cutoff_power = grouping_power + 1
        self.cutoff_value = 1 << self.cutoff_power
        self.upper_bin_divisions = 1 << grouping_power
        self.lower_bin_count = self.cutoff_value
        self.upper_bin_count = (max_value_power - self.cutoff_power) * self.upper_bin_divisions
        self.num_buckets = self.lower_bin_count + self.upper_bin_count
        self.counts: List[int] = [0] * self.num_buckets

    def value_to_index(self, value: int) -> int:
        if value < 0:
            raise ValueError(f"value must be non-negative, got {value}")
        if value < self.cutoff_value:
            return value  # linear region: index == value
        power = value.bit_length() - 1  # == 63 - u64::leading_zeros(value)
        log_bin = power - self.cutoff_power
        offset = (value - (1 << power)) >> (power - self.grouping_power)
        return self.lower_bin_count + log_bin * self.upper_bin_divisions + offset

    def record(self, value: int, count: int = 1) -> None:
        self.counts[self.value_to_index(value)] += count

    def snapshot(self) -> List[int]:
        """Return a copy of the current (cumulative) bucket counts."""
        return list(self.counts)
