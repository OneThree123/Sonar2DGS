"""Group-boundary quality triggers; querying does not consume targets."""

import math


QUALITY_TEST_SECONDS = (
    10, 20, 30, 50, 75, 100, 120, 150, 200, 300, 400, 500, 650, 800,
    1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 4000, 5000, 6000, 7000,
)
OPACITY_RESET_POLICY_VERSION = 1


class QualityTimeSchedule:
    def __init__(self, targets=QUALITY_TEST_SECONDS, *, consumed=(), missed=()):
        values = [float(value) for value in targets]
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Quality time targets must be finite positive seconds")
        self.targets = tuple(sorted(set(values)))
        self.consumed = set(float(v) for v in consumed)
        self.missed = set(float(v) for v in missed)
        if not (self.consumed | self.missed) <= set(self.targets):
            raise ValueError("Restored quality targets do not match the configuration")
        if self.consumed & self.missed:
            raise ValueError("A quality target cannot be both consumed and missed")

    def due(self, actual_seconds):
        if not math.isfinite(actual_seconds) or actual_seconds < 0:
            raise ValueError("Training time must be finite and non-negative")
        return [v for v in self.targets if v <= actual_seconds and v not in self.consumed | self.missed]

    def triggers(self, actual_seconds, *, iteration=False, final=False):
        due = self.due(actual_seconds)
        reasons = []
        if due:
            reasons.append("time")
        if iteration:
            reasons.append("iteration")
        if final:
            reasons.append("final")
        return reasons, due

    def commit(self, targets):
        targets = set(targets)
        if not targets <= set(self.targets) - self.missed:
            raise ValueError("Cannot consume unknown or historically missed targets")
        self.consumed.update(targets)

    @property
    def remaining(self):
        return [v for v in self.targets if v not in self.consumed | self.missed]

    def state(self):
        return dict(targets=list(self.targets), consumed_targets=sorted(self.consumed),
                    missed_before_resume_targets=sorted(self.missed))
