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
    guard_committed: bool = True


@dataclass(frozen=True)
class BankSnapshot:
    cycle: int
    pressure: Mapping[int, BankPressure]
    free_bytes: Mapping[int, int]
    capacity_bytes: Mapping[int, int] = field(default_factory=dict)
    queue_capacity: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cycle < 0:
            raise ValueError("snapshot cycle must be non-negative")
        if any(value < 0 for value in self.free_bytes.values()):
            raise ValueError("free Bank capacity must be non-negative")
        if any(value <= 0 for value in self.capacity_bytes.values()):
            raise ValueError("Bank capacity must be positive")
        if any(value <= 0 for value in self.queue_capacity.values()):
            raise ValueError("queue capacity must be positive")


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
        guard_incumbent: bool = False,
        switching_cost_cycles: int = 0,
    ) -> PrefetchDecision:
        if estimated_transfer_cycles <= 0:
            raise ValueError("estimated transfer cycles must be positive")
        if snapshot.cycle >= chunk.use_cycle:
            return PrefetchDecision(
                chunk.chunk_id, PrefetchAction.DEMAND, snapshot.cycle,
                issue_cycle=snapshot.cycle, reason="use_deadline_reached",
            )

        # P6: a hot Bank is an optimization cost, not a reason to remove the
        # matched NaivePF request. Search every capacity-feasible group and
        # minimize an end-to-end incremental-cost estimate.
        candidates = [
            bank for bank, free in snapshot.free_bytes.items() if free > 0
        ]
        feasible_groups = []
        defaults = tuple(default_banks)
        slack = chunk.use_cycle - snapshot.cycle
        for group in combinations(sorted(candidates), chunk.bank_group_size):
            capacity = sum(snapshot.free_bytes.get(bank, 0) for bank in group)
            if capacity >= chunk.size_bytes:
                pressures = [
                    snapshot.pressure.get(bank, BankPressure()) for bank in group
                ]
                queue_cost = sum(
                    item.queue_depth * estimated_transfer_cycles
                    for item in pressures
                )
                occupancy_queue_cost = 0
                for bank in group:
                    capacity_bytes = snapshot.capacity_bytes.get(
                        bank, snapshot.free_bytes[bank]
                    )
                    occupied_bytes = max(
                        0, capacity_bytes - snapshot.free_bytes[bank]
                    )
                    occupancy_queue_cost += (
                        (occupied_bytes + chunk.size_bytes - 1)
                        // chunk.size_bytes
                    ) * estimated_transfer_cycles
                interference_cost = sum(
                    min(item.busy_cycles, estimated_transfer_cycles)
                    + item.conflicts * max(1, estimated_transfer_cycles // 2)
                    for item in pressures
                )
                predicted_completion = (
                    snapshot.cycle + estimated_transfer_cycles
                    + queue_cost + occupancy_queue_cost
                )
                late_cost = max(0, predicted_completion - chunk.use_cycle)
                incumbent_penalty = int(
                    bool(defaults) and group != defaults[:len(group)]
                )
                feasible_groups.append((
                    late_cost,
                    occupancy_queue_cost,
                    interference_cost,
                    queue_cost,
                    incumbent_penalty,
                    -capacity,
                    group,
                ))
        feasible_groups.sort()
        selected = feasible_groups[0][-1] if feasible_groups else ()
        # A virtual mapping decision supplies a feasible Bank pool rather than
        # pinning the object to one prematurely chosen physical Bank. The
        # lifetime-aware mapping table performs the final group selection at
        # the real allocation event.
        pool_width = max(chunk.bank_group_size, len(defaults))
        selected_pool = []
        for candidate in feasible_groups:
            for bank in candidate[-1]:
                if bank not in selected_pool:
                    selected_pool.append(bank)
                if len(selected_pool) >= pool_width:
                    break
            if len(selected_pool) >= pool_width:
                break
        target_banks = tuple(selected_pool) if selected_pool else selected
        enough_capacity = bool(selected)
        enough_banks = len(selected) == chunk.bank_group_size

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

        guard_committed = True
        if guard_incumbent and defaults:
            incumbent_groups = [
                item for item in feasible_groups
                if set(item[-1]).issubset(set(defaults))
            ]
            incumbent = min(incumbent_groups) if incumbent_groups else None
            dynamic_cost = list(feasible_groups[0][:-1])
            dynamic_cost[0] += max(
                0, int(switching_cost_cycles) - max(0, slack)
            )
            dynamic_cost = tuple(dynamic_cost)
            incumbent_cost = incumbent[:-1] if incumbent is not None else None
            guard_committed = (
                incumbent_cost is None or dynamic_cost < incumbent_cost
            )
            if not guard_committed:
                target_banks = defaults

        redirected = bool(defaults and target_banks != defaults[:len(target_banks)])
        return PrefetchDecision(
            chunk.chunk_id, PrefetchAction.PREFETCH, snapshot.cycle,
            issue_cycle=snapshot.cycle, target_banks=target_banks,
            redirected=redirected,
            reason=(
                "online_guard_commit" if guard_incumbent and guard_committed
                else "online_guard_incumbent" if guard_incumbent
                else "minimum_incremental_cost"
            ),
            guard_committed=guard_committed,
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
