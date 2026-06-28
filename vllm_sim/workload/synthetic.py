"""Seeded synthetic workload generator.

Arrivals are either a Poisson process (``arrival_rate`` reqs/sec) or a fixed
``interval``. Prompt / output lengths are sampled uniformly from inclusive
``(lo, hi)`` ranges. Everything is driven by a single seeded ``random.Random``
so a given config reproduces byte-identical specs.
"""

import random
from typing import Iterable, Optional, Tuple

from vllm_sim.workload.base import RequestSpec


class SyntheticWorkload:
    def __init__(
        self,
        num_requests: int,
        prompt_len: Tuple[int, int],
        output_len: Tuple[int, int],
        arrival_rate: Optional[float] = None,
        interval: Optional[float] = None,
        seed: int = 0,
        id_prefix: str = "req",
    ):
        if arrival_rate is None and interval is None:
            raise ValueError("provide either arrival_rate or interval")
        if arrival_rate is not None and arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        self.num_requests = num_requests
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.arrival_rate = arrival_rate
        self.interval = interval
        self.seed = seed
        self.id_prefix = id_prefix

    def _sample_len(self, rng: random.Random, bounds: Tuple[int, int]) -> int:
        lo, hi = bounds
        return max(1, rng.randint(lo, hi))

    def generate(self) -> Iterable[RequestSpec]:
        rng = random.Random(self.seed)
        specs = []
        t = 0.0
        for i in range(self.num_requests):
            if i == 0:
                arrival = 0.0
            elif self.arrival_rate is not None:
                # Poisson process: inter-arrival ~ Exponential(rate).
                t += rng.expovariate(self.arrival_rate)
                arrival = t
            else:
                arrival = i * self.interval
            specs.append(
                RequestSpec(
                    request_id=f"{self.id_prefix}{i}",
                    arrival_time=arrival,
                    prompt_len=self._sample_len(rng, self.prompt_len),
                    output_len=self._sample_len(rng, self.output_len),
                )
            )
        return specs
