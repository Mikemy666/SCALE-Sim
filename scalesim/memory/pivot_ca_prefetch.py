"""Coverage/accuracy-constrained online prefetch policy for PIVOT DATE3.

The module deliberately contains no model-specific tuning.  Every policy
constant is supplied by :class:`CoverageAccuracyPolicyConfig`, and decisions
use only the current snapshot plus feedback from completed epochs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from itertools import combinations
from math import ceil
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from scalesim.memory.prefetch_policy import BankSnapshot
from scalesim.memory.virtual_bank_mapping import BankPressure


@dataclass(frozen=True)
class TileLifetime:
    tile_id: str
    size_bytes: int
    first_required_cycle: Optional[int] = None
    prefetch_issue_cycle: Optional[int] = None
    prefetch_complete_cycle: Optional[int] = None
    first_use_cycle: Optional[int] = None
    release_cycle: Optional[int] = None
    eviction_cycle: Optional[int] = None
    resident_at_decision: bool = False

    def __post_init__(self) -> None:
        if not self.tile_id or self.size_bytes <= 0:
            raise ValueError("Tile identity and size must be valid")
        values = (
            self.first_required_cycle, self.prefetch_issue_cycle,
            self.prefetch_complete_cycle, self.first_use_cycle,
            self.release_cycle, self.eviction_cycle,
        )
        if any(value is not None and value < 0 for value in values):
            raise ValueError("Tile lifetime cycles must be non-negative")


@dataclass(frozen=True)
class PrefetchQualityStats:
    required_bytes: int
    prefetched_bytes: int
    useful_timely_bytes: int
    late_bytes: int
    unused_bytes: int
    evicted_before_use_bytes: int
    coverage: Optional[float]
    accuracy: Optional[float]
    coverage_valid: bool
    accuracy_valid: bool

    def __post_init__(self) -> None:
        counts = (
            self.required_bytes, self.prefetched_bytes,
            self.useful_timely_bytes, self.late_bytes, self.unused_bytes,
            self.evicted_before_use_bytes,
        )
        if min(counts) < 0:
            raise ValueError("quality byte counters must be non-negative")
        if self.useful_timely_bytes > self.required_bytes:
            raise ValueError("useful bytes exceed required bytes")
        if self.useful_timely_bytes > self.prefetched_bytes:
            raise ValueError("useful bytes exceed prefetched bytes")
        for value, valid in ((self.coverage, self.coverage_valid),
                             (self.accuracy, self.accuracy_valid)):
            if valid != (value is not None):
                raise ValueError("ratio validity does not match value")
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError("quality ratios must be in [0, 1]")


def quality_from_lifetimes(
    lifetimes: Iterable[TileLifetime],
) -> PrefetchQualityStats:
    """Classify unique Tiles once and compute the DATE3 byte ratios."""
    unique: Dict[str, TileLifetime] = {}
    for item in lifetimes:
        previous = unique.get(item.tile_id)
        if previous is None:
            unique[item.tile_id] = item
        elif previous != item:
            raise ValueError(f"conflicting duplicate Tile lifetime: {item.tile_id}")

    required = prefetched = useful = late = unused = evicted = 0
    for item in unique.values():
        is_required = (
            item.first_required_cycle is not None
            and not item.resident_at_decision
        )
        is_prefetched = item.prefetch_issue_cycle is not None
        if is_required:
            required += item.size_bytes
        if is_prefetched:
            prefetched += item.size_bytes
        if not is_prefetched:
            continue
        used = item.first_use_cycle is not None
        completed = item.prefetch_complete_cycle is not None
        evicted_first = (
            item.eviction_cycle is not None and used
            and item.eviction_cycle < int(item.first_use_cycle)
        )
        timely = (
            used and completed
            and int(item.prefetch_complete_cycle) <= int(item.first_use_cycle)
            and not evicted_first
        )
        if timely and is_required:
            useful += item.size_bytes
        elif evicted_first:
            evicted += item.size_bytes
        elif used and completed and int(item.prefetch_complete_cycle) > int(item.first_use_cycle):
            late += item.size_bytes
        elif not used:
            unused += item.size_bytes

    coverage_valid = required > 0
    accuracy_valid = prefetched > 0
    return PrefetchQualityStats(
        required, prefetched, useful, late, unused, evicted,
        min(1.0, useful / required) if coverage_valid else None,
        min(1.0, useful / prefetched) if accuracy_valid else None,
        coverage_valid, accuracy_valid,
    )


@dataclass(frozen=True)
class ScoreWeights:
    latency: float = 1.0
    occupancy: float = 0.20
    pressure: float = 0.20
    conflict: float = 0.15
    mapping: float = 0.05


@dataclass(frozen=True)
class CoverageAccuracyPolicyConfig:
    enabled: bool = True
    reference_mode: str = "shadow_fixed"
    reference_chunk: int = 4
    reference_window: int = 8
    eta_coverage: float = 0.25
    eta_accuracy: float = 0.25
    min_coverage: float = 0.50
    min_accuracy: float = 0.50
    epsilon_coverage: float = 0.05
    epsilon_accuracy: float = 0.05
    ema_warmup_epochs: int = 2
    candidate_chunks: Tuple[int, ...] = (1, 2, 4, 8)
    candidate_windows: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    bank_candidate_count: int = 4
    max_residency_ratio: float = 0.85
    pressure_threshold: float = 0.90
    adaptation_epoch: int = 1
    adaptation_cooldown: int = 1
    score_hysteresis: float = 0.02
    max_chunk_step: int = 1
    max_window_step: int = 2
    base_safety_margin: float = 2.0
    timing_margin_scale: float = 0.25
    minimum_positive_score: float = -1.0
    severe_late_ratio: float = 0.50
    online_incumbent_guard: bool = True
    pressure_mode: str = "mean_max"
    pressure_weights: Mapping[str, float] = field(default_factory=lambda: {
        "queue": 0.30, "busy": 0.25, "conflict": 0.25, "residency": 0.20,
    })
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CoverageAccuracyPolicyConfig":
        # DATE3 integration cases may colocate workload annotations (for
        # example speculative Tile IDs) beside policy settings.  Ignore those
        # annotations here while still rejecting misspelled core settings in
        # the generator/contract validator.
        allowed = {item.name for item in fields(cls)}
        data = {key: item for key, item in value.items() if key in allowed}
        if "score_weights" in data:
            data["score_weights"] = ScoreWeights(**dict(data["score_weights"]))
        if "candidate_chunks" in data:
            data["candidate_chunks"] = tuple(int(x) for x in data["candidate_chunks"])
        if "candidate_windows" in data:
            data["candidate_windows"] = tuple(int(x) for x in data["candidate_windows"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if self.reference_mode not in {"shadow_fixed", "profiled_threshold", "static_threshold"}:
            raise ValueError("unsupported reference_mode")
        if self.pressure_mode not in {"mean", "max", "mean_max"}:
            raise ValueError("unsupported pressure_mode")
        ratios = (self.eta_coverage, self.eta_accuracy, self.min_coverage,
                  self.min_accuracy, self.epsilon_coverage,
                  self.epsilon_accuracy, self.max_residency_ratio,
                  self.pressure_threshold, self.severe_late_ratio)
        if any(not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("policy ratios must be in [0, 1]")
        if not self.candidate_chunks or not self.candidate_windows:
            raise ValueError("candidate sets must not be empty")
        if any(x <= 0 for x in (*self.candidate_chunks, *self.candidate_windows)):
            raise ValueError("Chunk and Window candidates must be positive")
        if min(self.bank_candidate_count, self.adaptation_epoch,
               self.max_chunk_step, self.max_window_step) <= 0:
            raise ValueError("policy counts must be positive")


@dataclass(frozen=True)
class CandidateTile:
    tile_id: str
    size_bytes: int
    first_use_cycle: int
    predicted_use_probability: float = 1.0
    demand_latency: float = 1.0

    def __post_init__(self) -> None:
        if (not self.tile_id or self.size_bytes <= 0 or self.first_use_cycle < 0
                or not 0.0 <= self.predicted_use_probability <= 1.0
                or self.demand_latency < 0):
            raise ValueError("invalid candidate Tile")


@dataclass(frozen=True)
class PrefetchCandidate:
    chunk_size: int
    window: int
    bank_group: Tuple[int, ...]
    predicted_coverage: float
    predicted_accuracy: float
    predicted_latency_benefit: float
    predicted_occupancy: float
    predicted_conflict: float
    pressure: float
    mapping_cost: float
    score: float
    feasible: bool
    rejection_reason: str = ""


@dataclass(frozen=True)
class PrefetchDecisionRecord:
    decision_id: int
    cycle: int
    layer: str
    expert: int
    stage: int
    current_chunk: int
    current_window: int
    current_bank_group: Tuple[int, ...]
    candidate: PrefetchCandidate
    coverage_threshold: float
    accuracy_threshold: float
    selected: bool
    fallback_used: bool = False
    fallback_level: int = 0
    fallback_reason: str = ""

    def to_dict(self) -> Mapping[str, object]:
        row = {
            "decision_id": self.decision_id, "cycle": self.cycle,
            "layer": self.layer, "expert": self.expert, "stage": self.stage,
            "current_chunk": self.current_chunk,
            "current_window": self.current_window,
            "current_bank_group": ":".join(map(str, self.current_bank_group)),
            "coverage_threshold": self.coverage_threshold,
            "accuracy_threshold": self.accuracy_threshold,
            "selected": self.selected, "fallback_used": self.fallback_used,
            "fallback_level": self.fallback_level,
            "fallback_reason": self.fallback_reason,
        }
        candidate = asdict(self.candidate)
        candidate["candidate_bank_group"] = ":".join(map(str, candidate.pop("bank_group")))
        candidate["candidate_chunk"] = candidate.pop("chunk_size")
        candidate["candidate_window"] = candidate.pop("window")
        candidate["latency_benefit"] = candidate.pop("predicted_latency_benefit")
        candidate["occupancy_cost"] = candidate.pop("predicted_occupancy")
        row.update(candidate)
        return row


@dataclass
class PrefetchFeedbackState:
    coverage_ema: Optional[float] = None
    accuracy_ema: Optional[float] = None
    baseline_coverage_ema: Optional[float] = None
    baseline_accuracy_ema: Optional[float] = None
    late_ratio_ema: Optional[float] = None
    timing_error_ema: Optional[float] = None
    pressure_ema: Optional[float] = None
    occupancy_ema: Optional[float] = None
    current_chunk: int = 1
    current_window: int = 1
    current_bank_group: Tuple[int, ...] = ()
    current_score: float = float("-inf")
    cooldown_remaining: int = 0
    epoch_count: int = 0
    valid_coverage_epochs: int = 0
    valid_accuracy_epochs: int = 0


def _ema(old: Optional[float], sample: float, alpha: float) -> float:
    return sample if old is None else (1.0 - alpha) * old + alpha * sample


class CoverageAccuracyConstrainedPrefetchPolicy:
    """Bounded two-level online action search with quality constraints."""

    def __init__(self, config: CoverageAccuracyPolicyConfig):
        config.validate()
        self.config = config
        self.state = PrefetchFeedbackState(
            current_chunk=config.reference_chunk,
            current_window=config.reference_window,
        )
        self.decision_count = 0
        self.adaptation_count = 0

    @property
    def warm(self) -> bool:
        return min(self.state.valid_coverage_epochs,
                   self.state.valid_accuracy_epochs) >= self.config.ema_warmup_epochs

    def thresholds(self) -> Tuple[float, float]:
        if self.config.reference_mode == "static_threshold":
            return self.config.min_coverage, self.config.min_accuracy
        cov = self.state.baseline_coverage_ema
        acc = self.state.baseline_accuracy_ema
        return (
            max(self.config.min_coverage,
                (cov if cov is not None else 1.0) - self.config.epsilon_coverage),
            max(self.config.min_accuracy,
                (acc if acc is not None else 1.0) - self.config.epsilon_accuracy),
        )

    def update_feedback(
        self, stats: PrefetchQualityStats, *, baseline: Optional[PrefetchQualityStats] = None,
        mean_pressure: float = 0.0, occupancy_byte_cycles: float = 0.0,
        mean_timing_error: float = 0.0,
    ) -> None:
        cfg, state = self.config, self.state
        if stats.coverage_valid:
            state.coverage_ema = _ema(state.coverage_ema, float(stats.coverage), cfg.eta_coverage)
            state.valid_coverage_epochs += 1
        if stats.accuracy_valid:
            state.accuracy_ema = _ema(state.accuracy_ema, float(stats.accuracy), cfg.eta_accuracy)
            state.valid_accuracy_epochs += 1
        if baseline is not None and baseline.coverage_valid:
            state.baseline_coverage_ema = _ema(
                state.baseline_coverage_ema, float(baseline.coverage), cfg.eta_coverage)
        if baseline is not None and baseline.accuracy_valid:
            state.baseline_accuracy_ema = _ema(
                state.baseline_accuracy_ema, float(baseline.accuracy), cfg.eta_accuracy)
        prefetch_total = stats.prefetched_bytes
        late_ratio = stats.late_bytes / prefetch_total if prefetch_total else 0.0
        state.late_ratio_ema = _ema(state.late_ratio_ema, late_ratio, cfg.eta_coverage)
        state.timing_error_ema = _ema(state.timing_error_ema, mean_timing_error, cfg.eta_coverage)
        state.pressure_ema = _ema(state.pressure_ema, mean_pressure, cfg.eta_coverage)
        state.occupancy_ema = _ema(state.occupancy_ema, occupancy_byte_cycles, cfg.eta_coverage)
        state.epoch_count += 1
        state.cooldown_remaining = max(0, state.cooldown_remaining - 1)

    def _bank_pressure(self, group: Tuple[int, ...], snapshot: BankSnapshot,
                       horizon: int) -> Tuple[float, float]:
        parts = []
        for bank in group:
            pressure = snapshot.pressure.get(bank, BankPressure())
            capacity = snapshot.capacity_bytes.get(bank, snapshot.free_bytes.get(bank, 1))
            free = snapshot.free_bytes.get(bank, 0)
            components = {
                "queue": min(1.0, pressure.queue_depth / max(
                    1, snapshot.queue_capacity.get(bank, 1))),
                "busy": min(1.0, pressure.busy_cycles / max(1, horizon)),
                "conflict": min(1.0, pressure.conflicts / max(
                    1, snapshot.queue_capacity.get(bank, 1))),
                "residency": min(1.0, max(0, capacity - free) / max(1, capacity)),
            }
            parts.append(sum(self.config.pressure_weights[name] * value
                             for name, value in components.items()))
        mean = sum(parts) / len(parts)
        maximum = max(parts)
        if self.config.pressure_mode == "mean":
            combined = mean
        elif self.config.pressure_mode == "max":
            combined = maximum
        else:
            combined = 0.5 * (mean + maximum)
        conflict = sum(snapshot.pressure.get(bank, BankPressure()).conflicts
                       for bank in group) / max(
                           1.0, sum(snapshot.queue_capacity.get(bank, 1)
                                    for bank in group))
        return min(1.0, combined), min(1.0, conflict)

    def _groups(self, snapshot: BankSnapshot, group_size: int,
                required_bytes: int, horizon: int) -> Tuple[Tuple[Tuple[int, ...], float, float], ...]:
        feasible = []
        banks = tuple(sorted(snapshot.free_bytes))
        # The physical Bank count is small, but bound the group construction:
        # for multi-Bank objects use deterministic contiguous virtual groups.
        groups = ((bank,) for bank in banks) if group_size == 1 else (
            tuple(banks[(start + offset) % len(banks)] for offset in range(group_size))
            for start in range(len(banks))
        )
        seen = set()
        for group in groups:
            group = tuple(group)
            if group in seen:
                continue
            seen.add(group)
            capacity = sum(snapshot.capacity_bytes.get(bank, 0) for bank in group)
            free = sum(snapshot.free_bytes.get(bank, 0) for bank in group)
            if required_bytes > free or required_bytes > self.config.max_residency_ratio * capacity:
                continue
            pressure, conflict = self._bank_pressure(group, snapshot, horizon)
            feasible.append((group, pressure, conflict))
        feasible.sort(key=lambda item: (item[1], item[2], item[0]))
        return tuple(feasible[:self.config.bank_candidate_count])

    @staticmethod
    def minimum_window(candidate_windows: Sequence[int], lead_cycles: Mapping[int, float],
                       transfer_cycles: float, margin: float) -> Optional[int]:
        for window in sorted(candidate_windows):
            if lead_cycles.get(window, 0.0) >= transfer_cycles + margin:
                return int(window)
        return None

    def choose(
        self, *, cycle: int, layer: str, expert: int, stage: int,
        tiles: Sequence[CandidateTile], snapshot: BankSnapshot,
        lead_cycles: Mapping[int, float], bandwidth_bytes_per_cycle: float,
        setup_cycles: float = 0.0, mapping_cycles: float = 0.0,
        group_size: int = 1, chunk_granularity: bool = False,
        hbm_backlog_cycles: float = 0.0,
    ) -> Tuple[PrefetchCandidate, Tuple[PrefetchDecisionRecord, ...]]:
        if not tiles or bandwidth_bytes_per_cycle <= 0:
            raise ValueError("online decision requires Tiles and positive bandwidth")
        cfg, state = self.config, self.state
        self.decision_count += 1
        tau_cov, tau_acc = self.thresholds()
        horizon = max(1, max(lead_cycles.values(), default=1))
        candidates = []
        total_expected = sum(t.size_bytes * t.predicted_use_probability for t in tiles)
        # Warm-up controls feedback-driven pruning/hysteresis, not whether the
        # hardware may evaluate its legal Chunk granularities.  Restricting a
        # cold controller to ``reference_chunk`` created a lock-in: if the
        # first C=8 request was late, no accuracy-valid epoch accumulated and
        # C=1/2/4 were never allowed to compete.  Prediction-only search is
        # safe before warm-up; measured feedback refines it afterwards.
        chunks = cfg.candidate_chunks
        if self.warm:
            coverage_low = (
                state.coverage_ema is not None and state.coverage_ema < tau_cov
            )
            accuracy_high = (
                state.accuracy_ema is not None and state.accuracy_ema >= tau_acc
            )
            accuracy_low = (
                state.accuracy_ema is not None and state.accuracy_ema < tau_acc
            )
            late_high = (state.late_ratio_ema or 0.0) > cfg.severe_late_ratio
            if coverage_low and accuracy_high and not late_high:
                chunks = tuple(item for item in chunks
                               if item >= state.current_chunk) or chunks
            elif accuracy_low:
                chunks = tuple(item for item in chunks
                               if item <= state.current_chunk) or chunks
        for chunk_size in chunks:
            # Legacy unit/fault-injection cases use ``chunk_size`` as a
            # prefetch degree.  DATE3 paper executions use the architecturally
            # correct interpretation: Chunk is the number of atomic weight
            # tiles coalesced into one request, while every tile in the epoch
            # remains part of the required stream.
            selected_tiles = (
                tuple(tiles) if chunk_granularity
                else tuple(tiles[:min(chunk_size, len(tiles))])
            )
            request_groups = (
                tuple(selected_tiles[start:start + chunk_size]
                      for start in range(0, len(selected_tiles), chunk_size))
                if chunk_granularity else (selected_tiles,)
            )
            bytes_selected = sum(t.size_bytes for t in selected_tiles)
            largest_request = max(
                (sum(t.size_bytes for t in request) for request in request_groups),
                default=bytes_selected,
            )
            groups = self._groups(
                snapshot, group_size,
                largest_request if chunk_granularity else bytes_selected,
                horizon,
            )
            for group, pressure, conflict in groups:
                queue_cycles = pressure * horizon
                request_count = len(request_groups) if chunk_granularity else 1
                margin = (
                    cfg.base_safety_margin
                    + cfg.timing_margin_scale * abs(state.timing_error_ema or 0.0)
                    + (cfg.base_safety_margin * (state.late_ratio_ema or 0.0))
                )
                if chunk_granularity:
                    # Runtime Chunk is a transfer granularity, not merely a
                    # request-count divisor.  Model the shared HBM stream one
                    # coalesced request at a time so an early small Chunk can
                    # become usable before later requests finish.  The old
                    # epoch-wide completion timestamp erased this first-byte
                    # advantage and made the largest Chunk dominate every
                    # phase by construction.
                    first_deadline = min(t.first_use_cycle for t in selected_tiles)
                    per_request_fixed = (
                        setup_cycles + mapping_cycles + queue_cycles
                        + conflict * horizon
                    )
                    for window in sorted(cfg.candidate_windows):
                        predicted_issue = max(
                            cycle, first_deadline - lead_cycles.get(window, 0.0)
                        )
                        cursor = max(
                            float(predicted_issue),
                            float(cycle) + hbm_backlog_cycles,
                        )
                        completions = {}
                        for request in request_groups:
                            request_bytes = sum(t.size_bytes for t in request)
                            cursor += (
                                per_request_fixed
                                + ceil(request_bytes / bandwidth_bytes_per_cycle)
                            )
                            for tile in request:
                                completions[tile.tile_id] = cursor
                        useful = sum(
                            t.size_bytes * t.predicted_use_probability
                            for t in selected_tiles
                            if completions[t.tile_id] + margin <= t.first_use_cycle
                        )
                        coverage = useful / total_expected if total_expected else 0.0
                        accuracy_den = sum(t.size_bytes for t in selected_tiles)
                        accuracy = useful / accuracy_den if accuracy_den else 0.0
                        occupancy = sum(
                            t.size_bytes * max(
                                t.first_use_cycle - completions[t.tile_id], 0
                            )
                            for t in selected_tiles
                        )
                        latency = sum(
                            t.predicted_use_probability * max(
                                t.demand_latency - max(
                                    completions[t.tile_id] - t.first_use_cycle, 0
                                ),
                                0,
                            )
                            for t in selected_tiles
                        )
                        transfer = max(1.0, cursor - predicted_issue)
                        norm_latency = latency / max(
                            1.0, sum(t.demand_latency for t in tiles)
                        )
                        max_occ = max(
                            1.0, sum(t.size_bytes for t in tiles) * horizon
                        )
                        norm_occ = occupancy / max_occ
                        norm_mapping = (
                            request_count * mapping_cycles / transfer
                        )
                        weights = cfg.score_weights
                        score = (
                            weights.latency * norm_latency
                            - weights.occupancy * norm_occ
                            - weights.pressure * pressure
                            - weights.conflict * conflict
                            - weights.mapping * norm_mapping
                        )
                        reason = ""
                        if coverage < tau_cov:
                            reason = "coverage_below_threshold"
                        elif accuracy < tau_acc:
                            reason = "accuracy_below_threshold"
                        elif pressure > cfg.pressure_threshold:
                            reason = "pressure_above_threshold"
                        elif score < cfg.minimum_positive_score:
                            reason = "score_below_minimum"
                        candidates.append(PrefetchCandidate(
                            chunk_size, window, group,
                            min(1.0, coverage), min(1.0, accuracy),
                            latency, occupancy, conflict, pressure,
                            mapping_cycles, score, not reason, reason,
                        ))
                    continue
                transfer = (
                    hbm_backlog_cycles
                    +
                    request_count * (
                        setup_cycles + mapping_cycles + queue_cycles
                        + conflict * horizon
                    )
                    + ceil(bytes_selected / bandwidth_bytes_per_cycle)
                )
                window = self.minimum_window(cfg.candidate_windows, lead_cycles, transfer, margin)
                if window is None:
                    candidates.append(PrefetchCandidate(
                        chunk_size, max(cfg.candidate_windows), group, 0.0, 0.0,
                        0.0, 0.0, conflict, pressure, mapping_cycles, float("-inf"),
                        False, "no_timely_window"))
                    continue
                # Window is an actual lead distance, not merely a label.  The
                # decision is made at ``cycle`` and the request is issued at
                # the earliest legal point represented by the selected lead.
                first_deadline = min(t.first_use_cycle for t in selected_tiles)
                predicted_issue = max(
                    cycle, first_deadline - lead_cycles.get(window, 0.0)
                )
                completion = predicted_issue + transfer
                useful = sum(
                    t.size_bytes * t.predicted_use_probability
                    for t in selected_tiles if completion <= t.first_use_cycle
                )
                coverage = useful / total_expected if total_expected else 0.0
                accuracy_den = sum(t.size_bytes for t in selected_tiles)
                accuracy = useful / accuracy_den if accuracy_den else 0.0
                occupancy = sum(
                    t.size_bytes * max(t.first_use_cycle - completion, 0)
                    for t in selected_tiles
                )
                latency = sum(
                    t.predicted_use_probability * max(
                        t.demand_latency - max(completion - t.first_use_cycle, 0), 0)
                    for t in selected_tiles
                )
                norm_latency = latency / max(1.0, sum(t.demand_latency for t in tiles))
                max_occ = max(1.0, sum(t.size_bytes for t in tiles) * horizon)
                norm_occ = occupancy / max_occ
                norm_mapping = (
                    request_count * mapping_cycles / max(1.0, transfer)
                )
                weights = cfg.score_weights
                score = (weights.latency * norm_latency
                         - weights.occupancy * norm_occ
                         - weights.pressure * pressure
                         - weights.conflict * conflict
                         - weights.mapping * norm_mapping)
                reason = ""
                if coverage < tau_cov:
                    reason = "coverage_below_threshold"
                elif accuracy < tau_acc:
                    reason = "accuracy_below_threshold"
                elif pressure > cfg.pressure_threshold:
                    reason = "pressure_above_threshold"
                elif score < cfg.minimum_positive_score:
                    reason = "score_below_minimum"
                candidates.append(PrefetchCandidate(
                    chunk_size, window, group, min(1.0, coverage), min(1.0, accuracy),
                    latency, occupancy, conflict, pressure, mapping_cycles, score,
                    not reason, reason,
                ))

        feasible = [item for item in candidates if item.feasible]
        fallback_level = 0
        fallback_reason = ""
        if feasible:
            chosen = max(feasible, key=lambda item: (item.score, -item.window,
                                                      -item.chunk_size, item.bank_group))
        else:
            fallback_reason = ",".join(sorted({item.rejection_reason for item in candidates})) or "no_candidate"
            # In the multi-layer runtime-granularity path, an absolute quality
            # target may be temporarily unreachable even though several
            # candidates have useful timely bytes.  Falling straight back to
            # the frozen C/W reference locked every later epoch to C=8/W=2.
            # Retain the best measured-quality proposal as a best-effort
            # action; the runner's three-way prefix-cost guard still compares
            # it with FixedPF and Coalesced-Demand before it can commit.
            quality_relaxed = [
                item for item in candidates
                if item.rejection_reason in {
                    "coverage_below_threshold", "accuracy_below_threshold"
                }
                and item.score != float("-inf")
                and (item.predicted_coverage > 0 or item.predicted_accuracy > 0)
            ]
            if chunk_granularity and quality_relaxed:
                fallback_level = 1
                chosen = replace(
                    max(
                        quality_relaxed,
                        key=lambda item: (
                            item.score, item.predicted_coverage,
                            item.predicted_accuracy, -item.window,
                            -item.chunk_size, item.bank_group,
                        ),
                    ),
                    feasible=True,
                    rejection_reason="quality_best_effort_fallback",
                )
            else:
                groups = self._groups(
                    snapshot, group_size,
                    sum(t.size_bytes for t in tiles[:cfg.reference_chunk]),
                    horizon,
                )
                if groups:
                    fallback_level = 1
                    chosen = PrefetchCandidate(
                        cfg.reference_chunk, cfg.reference_window, groups[0][0],
                        tau_cov, tau_acc, 0.0, 0.0, groups[0][2], groups[0][1],
                        mapping_cycles, 0.0, True,
                        "reference_fixed_fallback",
                    )
                else:
                    small = min(cfg.candidate_chunks)
                    groups = self._groups(
                        snapshot, group_size,
                        sum(t.size_bytes for t in tiles[:small]), horizon,
                    )
                    if groups:
                        fallback_level = 2
                        chosen = PrefetchCandidate(
                            small, min(cfg.candidate_windows), groups[0][0],
                            0.0, 0.0, 0.0, 0.0, groups[0][2], groups[0][1],
                            mapping_cycles, 0.0, True,
                            "conservative_fallback",
                        )
                    else:
                        fallback_level = 3
                        chosen = PrefetchCandidate(
                            0, 0, (), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                            0.0, 0.0, True, "noprefetch_fallback")

        emergency = (
            (state.coverage_ema is not None and state.coverage_ema < cfg.min_coverage)
            or (state.accuracy_ema is not None and state.accuracy_ema < cfg.min_accuracy)
            or (state.pressure_ema or 0.0) > cfg.pressure_threshold
            or (state.late_ratio_ema or 0.0) > cfg.severe_late_ratio
        )
        # Fallback actions are safety controls, not adaptation proposals;
        # cooldown, hysteresis, and step limiting must never rewrite NoPF or a
        # reference/conservative fallback into a different action.
        adaptation_epoch_hold = (
            state.current_bank_group
            and state.epoch_count % cfg.adaptation_epoch != 0
        )
        if fallback_level:
            pass
        elif adaptation_epoch_hold:
            chosen = PrefetchCandidate(
                state.current_chunk, state.current_window, state.current_bank_group,
                chosen.predicted_coverage, chosen.predicted_accuracy,
                chosen.predicted_latency_benefit, chosen.predicted_occupancy,
                chosen.predicted_conflict, chosen.pressure, chosen.mapping_cost,
                state.current_score, True, "adaptation_epoch_hold")
        elif self.warm and state.cooldown_remaining and not emergency:
            chosen = PrefetchCandidate(
                state.current_chunk, state.current_window, state.current_bank_group,
                chosen.predicted_coverage, chosen.predicted_accuracy,
                chosen.predicted_latency_benefit, chosen.predicted_occupancy,
                chosen.predicted_conflict, chosen.pressure, chosen.mapping_cost,
                state.current_score, True, "cooldown_hold")
        elif self.warm and state.current_bank_group and not emergency:
            if chosen.score < state.current_score + cfg.score_hysteresis:
                chosen = PrefetchCandidate(
                    state.current_chunk, state.current_window, state.current_bank_group,
                    chosen.predicted_coverage, chosen.predicted_accuracy,
                    chosen.predicted_latency_benefit, chosen.predicted_occupancy,
                    chosen.predicted_conflict, chosen.pressure, chosen.mapping_cost,
                    state.current_score, True, "hysteresis_hold")
            else:
                chosen = self._bounded_action(chosen)
        elif self.warm:
            chosen = self._bounded_action(chosen)

        changed = (chosen.chunk_size != state.current_chunk
                   or chosen.window != state.current_window
                   or chosen.bank_group != state.current_bank_group)
        if changed:
            self.adaptation_count += 1
            state.cooldown_remaining = cfg.adaptation_cooldown
        state.current_chunk = chosen.chunk_size
        state.current_window = chosen.window
        state.current_bank_group = chosen.bank_group
        state.current_score = chosen.score

        rows = tuple(PrefetchDecisionRecord(
            self.decision_count, cycle, layer, expert, stage,
            state.current_chunk, state.current_window, state.current_bank_group,
            candidate, tau_cov, tau_acc, candidate == chosen,
            bool(fallback_level), fallback_level, fallback_reason,
        ) for candidate in candidates)
        if not rows or not any(row.selected for row in rows):
            rows += (PrefetchDecisionRecord(
                self.decision_count, cycle, layer, expert, stage,
                state.current_chunk, state.current_window, state.current_bank_group,
                chosen, tau_cov, tau_acc, True, bool(fallback_level),
                fallback_level, fallback_reason,
            ),)
        return chosen, rows

    def _bounded_action(self, chosen: PrefetchCandidate) -> PrefetchCandidate:
        cfg, state = self.config, self.state
        def bounded(value: int, current: int, candidates: Sequence[int], step: int) -> int:
            ordered = sorted(set(candidates))
            current_index = min(range(len(ordered)), key=lambda i: abs(ordered[i] - current))
            target_index = min(range(len(ordered)), key=lambda i: abs(ordered[i] - value))
            target_index = max(current_index - step, min(current_index + step, target_index))
            return ordered[target_index]
        return PrefetchCandidate(
            bounded(chosen.chunk_size, state.current_chunk, cfg.candidate_chunks,
                    cfg.max_chunk_step),
            bounded(chosen.window, state.current_window, cfg.candidate_windows,
                    cfg.max_window_step),
            chosen.bank_group, chosen.predicted_coverage, chosen.predicted_accuracy,
            chosen.predicted_latency_benefit, chosen.predicted_occupancy,
            chosen.predicted_conflict, chosen.pressure, chosen.mapping_cost,
            chosen.score, chosen.feasible, chosen.rejection_reason,
        )
