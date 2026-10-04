import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupTrainingSchedule:
    """Quantize iteration-based training requests to complete dataset groups."""

    group_size: int
    requested_iterations: int

    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.requested_iterations <= 0:
            raise ValueError(
                "requested_iterations must be positive, "
                f"got {self.requested_iterations}"
            )

    @property
    def total_groups(self):
        return math.ceil(self.requested_iterations / self.group_size)

    @property
    def effective_iterations(self):
        return self.total_groups * self.group_size

    @property
    def rounding_overhead(self):
        return self.effective_iterations - self.requested_iterations

    def group_for_iteration(self, iteration):
        iteration = int(iteration)
        if iteration <= 0:
            raise ValueError(f"iteration must be positive, got {iteration}")
        return math.ceil(iteration / self.group_size)

    def group_start_iteration(self, group):
        self._validate_group(group)
        return (int(group) - 1) * self.group_size

    def group_end_iteration(self, group):
        self._validate_group(group)
        return int(group) * self.group_size

    def quantize_events(self, iterations):
        """Return {group: requested iterations}, merging events in one group."""
        events = {}
        for iteration in iterations:
            iteration = int(iteration)
            if iteration <= 0 or iteration > self.requested_iterations:
                continue
            group = self.group_for_iteration(iteration)
            if group > self.total_groups:
                continue
            events.setdefault(group, []).append(iteration)
        return {
            group: tuple(sorted(set(requested)))
            for group, requested in sorted(events.items())
        }

    def periodic_events(
        self,
        interval,
        *,
        from_iteration=0,
        until_iteration=None,
        until_inclusive=True,
    ):
        """Build periodic iteration targets, then quantize every target separately."""
        interval = int(interval)
        from_iteration = int(from_iteration)
        requested_until = (
            self.requested_iterations
            if until_iteration is None
            else int(until_iteration)
        )
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        if requested_until <= 0:
            return {}

        first = (from_iteration // interval + 1) * interval
        last = min(
            self.requested_iterations,
            requested_until if until_inclusive else requested_until - 1,
        )
        stop = last + 1
        return self.quantize_events(range(first, stop, interval))

    def _validate_group(self, group):
        group = int(group)
        if group <= 0:
            raise ValueError(f"group must be positive, got {group}")
