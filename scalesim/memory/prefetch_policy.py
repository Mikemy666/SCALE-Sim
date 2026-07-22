"""No-prefetch, naive, Bank-aware, oracle, and safe-prefetch policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Mapping, Optional, Sequence, Tuple

from scalesim.memory.chunk_residency import ChunkResidencyManager, WeightChunk
from scalesim.memory.virtual_bank_mapping import BankPressure


class PrefetchAction(str, Enum):
    DEMAND = "demand"
    PREFETCH = "prefetch"
    DELAY = "delay"
    CANCEL = "cancel"


@dataclass(frozen=True)
class PrefetchDecision:
    chunk_id: str
    action: PrefetchAction
    decision_cycle: int
    issue_cycle: Optional[int] = None
    target_banks: Tuple[int, ...] = field(default_factory=tuple)
    redirected: bool = False
    reason: str = ""


@dataclass(frozen=True)
class BankSnapshot:
    cycle: int
    pressure: Mapping[int, BankPressure]
    free_bytes: Mapping[int, int]

    def __post_init__(self) -> None:
        if self.cycle < 0:
            raise ValueError("snapshot cycle must be non-negative")
        if any(value < 0 for value in self.free_bytes.values()):
            raise ValueError("free Bank capacity must be non-negative")


@dataclass(frozen=True)
class PrefetchOutcome:
    name: str
    kind: str
    total_cycles: int
    decisions: Tuple[PrefetchDecision, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in {"none", "naive", "bank_aware", "oracle", "safe"}:
            raise ValueError(f"unsupported prefetch outcome kind: {self.kind}")
        if self.total_cycles < 0:
            raise ValueError("prefetch outcome cycles must be non-negative")


class NoPrefetchPolicy:
    def decide(self, chunk: WeightChunk) -> PrefetchDecision:
        return PrefetchDecision(
            chunk.chunk_id, PrefetchAction.DEMAND, chunk.use_cycle,
            issue_cycle=chunk.use_cycle, reason="prefetch_disabled",
        )


class NaivePrefetchPolicy:
    """Fixed look-ahead policy that deliberately ignores Bank pressure."""

    def __init__(self, window: int, fixed_banks: Sequence[int]):
        if window < 0 or not fixed_banks:
            raise ValueError("naive window must be non-negative and target Banks non-empty")
        self.window = int(window)
        self.fixed_banks = tuple(int(bank) for bank in fixed_banks)

    def plan(self, chunks: Sequence[WeightChunk]) -> Tuple[PrefetchDecision, ...]:
        ordered = tuple(sorted(chunks, key=lambda item: (item.use_cycle, item.chunk_id)))
        decisions = []
        for index, chunk in enumerate(ordered):
            if self.window == 0:
                decisions.append(NoPrefetchPolicy().decide(chunk))
                continue
            trigger_index = index - self.window
            issue = 0 if trigger_index < 0 else ordered[trigger_index].use_cycle
            decisions.append(PrefetchDecision(
                chunk.chunk_id, PrefetchAction.PREFETCH, issue, issue,
                self.fixed_banks, False, "fixed_window",
            ))
        return tuple(decisions)


class BankAwarePrefetchPolicy:
    """Online decision using only the supplied current Bank snapshot."""

    def __init__(
        self,
        queue_threshold: int = 2,
        conflict_threshold: int = 4,
        busy_threshold: int = 32,
    ):
        if min(queue_threshold, conflict_threshold, busy_threshold) < 0:
            raise ValueError("pressure thresholds must be non-negative")
        self.queue_threshold = queue_threshold
        self.conflict_threshold = conflict_threshold
        self.busy_threshold = busy_threshold

    def _hot(self, pressure: BankPressure) -> bool:
        return (
            pressure.queue_depth > self.queue_threshold
            or pressure.conflicts > self.conflict_threshold
            or pressure.busy_cycles > self.busy_threshold
        )

    def decide(
        self,
        chunk: WeightChunk,
        snapshot: BankSnapshot,
        estimated_transfer_cycles: int,
        default_banks: Sequence[int] = (),
    ) -> PrefetchDecision:
        if estimated_transfer_cycles <= 0:
            raise ValueError("estimated transfer cycles must be positive")
        if snapshot.cycle >= chunk.use_cycle:
            return PrefetchDecision(
                chunk.chunk_id, PrefetchAction.DEMAND, snapshot.cycle,
                issue_cycle=snapshot.cycle, reason="use_deadline_reached",
            )

        candidates = []
        for bank, free in snapshot.free_bytes.items():
            pressure = snapshot.pressure.get(bank, BankPressure())
            if free > 0 and not self._hot(pressure):
                candidates.append(bank)
        feasible_groups = []
        for group in combinations(sorted(candidates), chunk.bank_group_size):
            capacity = sum(snapshot.free_bytes.get(bank, 0) for bank in group)
            if capacity >= chunk.size_bytes:
                pressure_score = sum(
                    snapshot.pressure.get(bank, BankPressure()).score for bank in group
                )
                feasible_groups.append((pressure_score, -capacity, group))
        feasible_groups.sort()
        selected = feasible_groups[0][2] if feasible_groups else ()
        enough_capacity = bool(selected)
        enough_banks = len(selected) == chunk.bank_group_size
        slack = chunk.use_cycle - snapshot.cycle

        if not enough_banks or not enough_capacity:
            if slack > estimated_transfer_cycles + 1:
                return PrefetchDecision(
                    chunk.chunk_id, PrefetchAction.DELAY, snapshot.cycle,
                    issue_cycle=snapshot.cycle + 1, reason="capacity_or_pressure_high",
                )
            return PrefetchDecision(
                chunk.chunk_id, PrefetchAction.CANCEL, snapshot.cycle,
                reason="insufficient_safe_prefetch_slack",
            )

        defaults = tuple(default_banks)
        redirected = bool(defaults and selected != defaults[:len(selected)])
        return PrefetchDecision(
            chunk.chunk_id, PrefetchAction.PREFETCH, snapshot.cycle,
            issue_cycle=snapshot.cycle, target_banks=selected,
            redirected=redirected, reason="pressure_and_capacity_available",
        )


def apply_prefetch_decision(
    manager: ChunkResidencyManager,
    decision: PrefetchDecision,
    pressure: Optional[Mapping[int, BankPressure]] = None,
) -> None:
    if decision.action == PrefetchAction.PREFETCH:
        manager.prefetch(
            decision.chunk_id, int(decision.issue_cycle), pressure,
            decision.target_banks or None,
        )
    elif decision.action in (PrefetchAction.DEMAND, PrefetchAction.CANCEL):
        # CANCEL means cancel the speculative action and preserve correctness
        # through demand loading at the declared use cycle.
        manager.demand_load(decision.chunk_id, pressure=pressure)
    elif decision.action == PrefetchAction.DELAY:
        return
    else:
        raise ValueError(f"unsupported decision action: {decision.action}")


def select_oracle_prefetch(outcomes: Sequence[PrefetchOutcome]) -> PrefetchOutcome:
    if not outcomes or not any(item.kind == "none" for item in outcomes):
        raise ValueError("oracle prefetch candidates must include no-prefetch")
    selected = min(outcomes, key=lambda item: (item.total_cycles, item.name))
    return PrefetchOutcome(
        f"oracle:{selected.name}", "oracle", selected.total_cycles,
        selected.decisions,
        {"selected_candidate": selected.name, "fallback_to_no_prefetch": selected.kind == "none"},
    )


def select_safe_prefetch(
    no_prefetch: PrefetchOutcome, bank_aware: PrefetchOutcome
) -> PrefetchOutcome:
    if no_prefetch.kind != "none" or bank_aware.kind != "bank_aware":
        raise ValueError("safe prefetch requires none and bank_aware outcomes")
    selected = bank_aware if bank_aware.total_cycles <= no_prefetch.total_cycles else no_prefetch
    return PrefetchOutcome(
        f"safe:{selected.name}", "safe", selected.total_cycles,
        selected.decisions,
        {
            "selected_candidate": selected.name,
            "fallback_to_no_prefetch": selected.kind == "none",
            "no_prefetch_total_cycles": no_prefetch.total_cycles,
            "bank_aware_total_cycles": bank_aware.total_cycles,
        },
    )
