"""
Pure bank-conflict memory model for SCALE-Sim.

This module intentionally ignores all old memory effects (prefetching, SRAM/DRAM latency,
hit/miss) and models:
1) bank ownership/allocation for ifmap, filter, ofmap
2) one-request-per-bank-per-cycle service constraint
3) delay introduced by same-cycle bank conflicts
4) effective per-bank bandwidth impact
5) extra DRAM penalty when assigned SRAM capacity cannot hold tensor data
"""

import os
import numpy as np

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo


class BankAllocator:
    """Deterministic static/dynamic bank allocation helper."""

    TENSORS = ("ifmap", "filter", "ofmap")

    def __init__(self, total_banknum, enable_dynamic=False):
        self.total_banknum = int(total_banknum)
        self.enable_dynamic = bool(enable_dynamic)

    def _validate_counts(self, counts):
        if any(int(counts[t]) < 1 for t in self.TENSORS):
            raise ValueError("Bank model requires at least 1 bank for ifmap/filter/ofmap")
        if sum(int(counts[t]) for t in self.TENSORS) != self.total_banknum:
            raise ValueError("Sum of ifmap/filter/ofmap banks must equal total_banknum")

    def allocate(self, static_counts, element_counts):
        static_counts = {
            "ifmap": int(static_counts["ifmap"]),
            "filter": int(static_counts["filter"]),
            "ofmap": int(static_counts["ofmap"]),
        }

        if not self.enable_dynamic:
            self._validate_counts(static_counts)
            return static_counts

        if self.total_banknum < 3:
            raise ValueError("Dynamic mode requires total_banknum >= 3 to keep min 1 bank/tensor")

        ifmap_e = int(element_counts["ifmap"])
        filter_e = int(element_counts["filter"])
        ofmap_e = int(element_counts["ofmap"])
        total_elements = ifmap_e + filter_e + ofmap_e

        if total_elements <= 0:
            self._validate_counts(static_counts)
            return static_counts

        ideals = {
            "ifmap": self.total_banknum * (ifmap_e / total_elements),
            "filter": self.total_banknum * (filter_e / total_elements),
            "ofmap": self.total_banknum * (ofmap_e / total_elements),
        }

        counts = {
            t: max(1, int(np.floor(ideals[t])))
            for t in self.TENSORS
        }

        current_sum = sum(counts.values())

        if current_sum < self.total_banknum:
            remainders = sorted(
                self.TENSORS,
                key=lambda t: (-(ideals[t] - np.floor(ideals[t])), self.TENSORS.index(t))
            )
            add = self.total_banknum - current_sum
            idx = 0
            while add > 0:
                t = remainders[idx % len(remainders)]
                counts[t] += 1
                add -= 1
                idx += 1

        elif current_sum > self.total_banknum:
            remove = current_sum - self.total_banknum
            while remove > 0:
                removable = [t for t in self.TENSORS if counts[t] > 1]
                if not removable:
                    raise ValueError("Cannot reduce bank counts without violating min 1 bank/tensor")
                removable = sorted(removable, key=lambda t: (-counts[t], self.TENSORS.index(t)))
                target = removable[0]
                counts[target] -= 1
                remove -= 1

        self._validate_counts(counts)
        return counts


class TensorBankModel:
    """Per-tensor bank conflict simulator with one request per bank per cycle."""

    def __init__(self, name, bank_base, bank_count, service_cycles=1):
        self.name = name
        self.bank_base = int(bank_base)
        self.bank_count = int(bank_count)
        self.service_cycles = max(1, int(service_cycles))
        if self.bank_count < 1:
            raise ValueError(f"{name} bank_count must be >= 1")

        self.next_free_cycle = np.zeros((self.bank_count,), dtype=int)
        self.total_conflict_delay = 0
        self.total_requests = 0

        self.per_bank_access = {
            self.bank_base + i: 0
            for i in range(self.bank_count)
        }

        self.serviced_cycle_per_line = []

    def service_line(self, req_cycle, demand_line):
        line_service_cycle = int(req_cycle)

        for raw_addr in demand_line:
            addr = int(raw_addr)
            if addr == -1:
                continue

            local_bank = addr % self.bank_count
            physical_bank = self.bank_base + local_bank

            available_cycle = int(self.next_free_cycle[local_bank])
            service_cycle = max(int(req_cycle), available_cycle)
            self.next_free_cycle[local_bank] = service_cycle + self.service_cycles

            self.total_conflict_delay += (service_cycle - int(req_cycle))
            self.total_requests += 1
            self.per_bank_access[physical_bank] += 1

            if service_cycle > line_service_cycle:
                line_service_cycle = service_cycle

        self.serviced_cycle_per_line.append(line_service_cycle)
        line_delay = line_service_cycle - int(req_cycle)
        return line_service_cycle, line_delay


class banked_memory_system:
    """Pure bank-conflict memory system with an old-memory-compatible interface."""

    TENSORS = ("ifmap", "filter", "ofmap")

    def __init__(self, config=cfg(), topo_obj=topo(), layer_id=0, verbose=True):
        self.config = config
        self.topo = topo_obj
        self.layer_id = int(layer_id)
        self.verbose = verbose

        self.params_valid_flag = False
        self.traces_valid = False

        self.enable_bank_model = False
        self.enable_dynamic = False
        self.enable_capacity_penalty = True

        self.word_size_bytes = 1

        self.total_cycles = 0
        self.stall_cycles = 0

        self.ifmap_trace_matrix = np.zeros((1, 1), dtype=int)
        self.filter_trace_matrix = np.zeros((1, 1), dtype=int)
        self.ofmap_trace_matrix = np.zeros((1, 1), dtype=int)

        self.ifmap_sram_start_cycle = 0
        self.ifmap_sram_stop_cycle = 0
        self.filter_sram_start_cycle = 0
        self.filter_sram_stop_cycle = 0
        self.ofmap_sram_start_cycle = 0
        self.ofmap_sram_stop_cycle = 0

        self.ifmap_dram_start_cycle = 0
        self.ifmap_dram_stop_cycle = 0
        self.ifmap_dram_reads = 0
        self.filter_dram_start_cycle = 0
        self.filter_dram_stop_cycle = 0
        self.filter_dram_reads = 0
        self.ofmap_dram_start_cycle = 0
        self.ofmap_dram_stop_cycle = 0
        self.ofmap_dram_writes = 0

        self.bank_report = {}

        self.ifmap_model = None
        self.filter_model = None
        self.ofmap_model = None

        self.allocation = None
        self.static_counts = None
        self.total_banknum = 0
        self.bank_capacity_kb = 0.0
        self.bank_conflict_penalty = 1
        self.dram_penalty_scale = 8

        self.ifmap_elements = 0
        self.filter_elements = 0
        self.ofmap_elements = 0

        self.ifmap_sram_bw_cfg = 1
        self.filter_sram_bw_cfg = 1
        self.ofmap_sram_bw_cfg = 1
        self.effective_per_bank_bw = 1.0
        self.simulation_fell_back_to_static = False

        self.tensor_service_cycles = {
            "ifmap": 1,
            "filter": 1,
            "ofmap": 1,
        }
        self.tensor_service_breakdown = {
            "ifmap": {},
            "filter": {},
            "ofmap": {},
        }

        # Optional inputs used to make dynamic allocation conflict-aware
        self.request_counts = None
        self.ifmap_demand_for_alloc = None
        self.filter_demand_for_alloc = None
        self.ofmap_demand_for_alloc = None

    def _compute_tensor_elements(self):
        ifmap_h, ifmap_w = self.topo.get_layer_ifmap_dims(layer_id=self.layer_id)
        filter_h, filter_w = self.topo.get_layer_filter_dims(layer_id=self.layer_id)
        num_ch = self.topo.get_layer_num_channels(layer_id=self.layer_id)
        num_filters = self.topo.get_layer_num_filters(layer_id=self.layer_id)
        ofmap_elements = self.topo.get_layer_num_ofmap_px(layer_id=self.layer_id)

        self.ifmap_elements = int(ifmap_h) * int(ifmap_w) * int(num_ch)
        self.filter_elements = int(filter_h) * int(filter_w) * int(num_ch) * int(num_filters)
        self.ofmap_elements = int(ofmap_elements)

    def _get_tensor_elements(self, tensor_name):
        if tensor_name == "ifmap":
            return int(self.ifmap_elements)
        if tensor_name == "filter":
            return int(self.filter_elements)
        if tensor_name == "ofmap":
            return int(self.ofmap_elements)
        raise ValueError(f"Unknown tensor name: {tensor_name}")

    def _resolve_bandwidth_stats(self):
        self.ifmap_sram_bw_cfg = max(1, int(getattr(self.config, "ifmap_sram_bank_bandwidth", 1)))
        self.filter_sram_bw_cfg = max(1, int(getattr(self.config, "filter_sram_bank_bandwidth", 1)))

        ofmap_bw = getattr(self.config, "ofmap_sram_bank_bandwidth", None)
        if ofmap_bw is None:
            ofmap_bw = self.filter_sram_bw_cfg
        self.ofmap_sram_bw_cfg = max(1, int(ofmap_bw))

        total_bw = self.ifmap_sram_bw_cfg + self.filter_sram_bw_cfg + self.ofmap_sram_bw_cfg
        self.effective_per_bank_bw = float(total_bw) / float(max(1, self.total_banknum))
        self.effective_per_bank_bw = max(1e-6, self.effective_per_bank_bw)

    def _set_allocation_from_counts(self, counts):
        counts = {
            "ifmap": int(counts["ifmap"]),
            "filter": int(counts["filter"]),
            "ofmap": int(counts["ofmap"]),
        }

        ifmap_base = 0
        filter_base = counts["ifmap"]
        ofmap_base = counts["ifmap"] + counts["filter"]

        self.allocation = {
            "ifmap": {"base": ifmap_base, "count": counts["ifmap"]},
            "filter": {"base": filter_base, "count": counts["filter"]},
            "ofmap": {"base": ofmap_base, "count": counts["ofmap"]},
        }

    def _get_allocation_counts(self):
        return {
            "ifmap": int(self.allocation["ifmap"]["count"]),
            "filter": int(self.allocation["filter"]["count"]),
            "ofmap": int(self.allocation["ofmap"]["count"]),
        }

    def _build_capacity_stats(self):
        ifmap_kb, filter_kb, ofmap_kb = self.config.get_mem_sizes()
        total_kb = float(ifmap_kb + filter_kb + ofmap_kb)
        self.bank_capacity_kb = total_kb / float(max(1, self.total_banknum))

    def _get_bandwidth_scaled_service_cycles(self):
        # effective_per_bank_bw is treated as "elements per cycle per bank".
        bw_cycles = int(np.ceil(float(self.word_size_bytes) / float(self.effective_per_bank_bw)))
        return max(1, int(self.bank_conflict_penalty) * max(1, bw_cycles))

    def _get_tensor_service_cycles(self, tensor_name, bank_count):
        bank_count = max(1, int(bank_count))
        base_service_cycles = self._get_bandwidth_scaled_service_cycles()

        tensor_elems = self._get_tensor_elements(tensor_name)
        required_bytes = float(tensor_elems * self.word_size_bytes)
        capacity_bytes = float(bank_count) * float(self.bank_capacity_kb) * 1024.0
        overflow_bytes = max(0.0, required_bytes - capacity_bytes)

        dram_penalty_cycles = 0
        overflow_ratio = 0.0
        if self.enable_capacity_penalty and capacity_bytes > 0 and overflow_bytes > 0:
            overflow_ratio = overflow_bytes / capacity_bytes
            severity = float(np.log2(1.0 + overflow_ratio))
            dram_penalty_cycles = max(
                base_service_cycles * 4,
                int(np.ceil(base_service_cycles * self.dram_penalty_scale * severity))
            )

        total_service_cycles = max(1, int(base_service_cycles + dram_penalty_cycles))

        breakdown = {
            "base_conflict_cycles": int(self.bank_conflict_penalty),
            "bandwidth_scaled_cycles": int(base_service_cycles),
            "dram_penalty_cycles": int(dram_penalty_cycles),
            "required_bytes": int(required_bytes),
            "capacity_bytes": int(capacity_bytes),
            "overflow_bytes": int(overflow_bytes),
            "overflow_ratio": float(overflow_ratio),
            "overflow_to_dram": bool(overflow_bytes > 0),
        }

        return total_service_cycles, breakdown

    def _build_models_for_counts(self, counts, use_allocation_bases=False):
        ifmap_count = int(counts["ifmap"])
        filter_count = int(counts["filter"])
        ofmap_count = int(counts["ofmap"])

        ifmap_service, ifmap_breakdown = self._get_tensor_service_cycles("ifmap", ifmap_count)
        filter_service, filter_breakdown = self._get_tensor_service_cycles("filter", filter_count)
        ofmap_service, ofmap_breakdown = self._get_tensor_service_cycles("ofmap", ofmap_count)

        if use_allocation_bases:
            ifmap_base = 0
            filter_base = ifmap_count
            ofmap_base = ifmap_count + filter_count
        else:
            ifmap_base = 0
            filter_base = 0
            ofmap_base = 0

        ifmap_model = TensorBankModel(
            "ifmap",
            bank_base=ifmap_base,
            bank_count=ifmap_count,
            service_cycles=ifmap_service,
        )
        filter_model = TensorBankModel(
            "filter",
            bank_base=filter_base,
            bank_count=filter_count,
            service_cycles=filter_service,
        )
        ofmap_model = TensorBankModel(
            "ofmap",
            bank_base=ofmap_base,
            bank_count=ofmap_count,
            service_cycles=ofmap_service,
        )

        service_cycles = {
            "ifmap": int(ifmap_service),
            "filter": int(filter_service),
            "ofmap": int(ofmap_service),
        }
        breakdown = {
            "ifmap": ifmap_breakdown,
            "filter": filter_breakdown,
            "ofmap": ofmap_breakdown,
        }

        return ifmap_model, filter_model, ofmap_model, service_cycles, breakdown

    def _simulate_with_counts(self, counts, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat, use_allocation_bases=False):
        ifmap_model, filter_model, ofmap_model, service_cycles, breakdown = self._build_models_for_counts(
            counts=counts,
            use_allocation_bases=use_allocation_bases,
        )

        num_lines = int(ofmap_demand_mat.shape[0])
        stall_cycles = 0

        ifmap_serviced_cycles = []
        filter_serviced_cycles = []
        ofmap_serviced_cycles = []

        for req_line in range(num_lines):
            req_cycle = int(req_line + stall_cycles)

            ifmap_cycle, ifmap_line_delay = ifmap_model.service_line(req_cycle, ifmap_demand_mat[req_line, :])
            filter_cycle, filter_line_delay = filter_model.service_line(req_cycle, filter_demand_mat[req_line, :])
            ofmap_cycle, ofmap_line_delay = ofmap_model.service_line(req_cycle, ofmap_demand_mat[req_line, :])

            ifmap_serviced_cycles.append(int(ifmap_cycle))
            filter_serviced_cycles.append(int(filter_cycle))
            ofmap_serviced_cycles.append(int(ofmap_cycle))

            stall_cycles += int(max(ifmap_line_delay, filter_line_delay, ofmap_line_delay))

        total_cycles = int(max(
            max(ifmap_serviced_cycles) if ifmap_serviced_cycles else 0,
            max(filter_serviced_cycles) if filter_serviced_cycles else 0,
            max(ofmap_serviced_cycles) if ofmap_serviced_cycles else 0,
        ))

        return {
            "ifmap_model": ifmap_model,
            "filter_model": filter_model,
            "ofmap_model": ofmap_model,
            "ifmap_serviced_cycles": ifmap_serviced_cycles,
            "filter_serviced_cycles": filter_serviced_cycles,
            "ofmap_serviced_cycles": ofmap_serviced_cycles,
            "stall_cycles": int(stall_cycles),
            "total_cycles": int(total_cycles),
            "service_cycles": dict(service_cycles),
            "breakdown": dict(breakdown),
        }

    def _estimate_stall_for_counts(self, counts):
        if self.ifmap_demand_for_alloc is None or self.filter_demand_for_alloc is None or self.ofmap_demand_for_alloc is None:
            return None

        sim = self._simulate_with_counts(
            counts=counts,
            ifmap_demand_mat=self.ifmap_demand_for_alloc,
            filter_demand_mat=self.filter_demand_for_alloc,
            ofmap_demand_mat=self.ofmap_demand_for_alloc,
            use_allocation_bases=False,
        )
        return int(sim["stall_cycles"])

    def _build_allocation(self):
        static_counts = {
            "ifmap": int(self.config.ifmap_sram_bank_num),
            "filter": int(self.config.filter_sram_bank_num),
            "ofmap": int(self.config.ofmap_sram_bank_num),
        }
        self.static_counts = dict(static_counts)

        self.total_banknum = static_counts["ifmap"] + static_counts["filter"] + static_counts["ofmap"]
        if self.total_banknum < 1:
            raise ValueError("total_banknum must be >= 1")

        self._build_capacity_stats()
        self._resolve_bandwidth_stats()

        allocator = BankAllocator(
            total_banknum=self.total_banknum,
            enable_dynamic=self.enable_dynamic,
        )

        element_counts = {
            "ifmap": self.ifmap_elements,
            "filter": self.filter_elements,
            "ofmap": self.ofmap_elements,
        }

        # Dynamic allocation should follow request pressure (conflict source) when available.
        allocation_weights = dict(element_counts)
        if self.enable_dynamic and isinstance(self.request_counts, dict):
            allocation_weights = {
                "ifmap": int(self.request_counts.get("ifmap", element_counts["ifmap"])),
                "filter": int(self.request_counts.get("filter", element_counts["filter"])),
                "ofmap": int(self.request_counts.get("ofmap", element_counts["ofmap"])),
            }

        dynamic_counts = allocator.allocate(static_counts=static_counts, element_counts=allocation_weights)
        counts = dict(dynamic_counts)

        # Safety net during allocation: fallback to static when dynamic estimate is worse.
        if self.enable_dynamic and dynamic_counts != static_counts:
            static_stall = self._estimate_stall_for_counts(static_counts)
            dynamic_stall = self._estimate_stall_for_counts(dynamic_counts)
            if static_stall is not None and dynamic_stall is not None and dynamic_stall > static_stall:
                counts = dict(static_counts)

        self._set_allocation_from_counts(counts)

    def _init_tensor_models(self):
        counts = self._get_allocation_counts()
        self.ifmap_model, self.filter_model, self.ofmap_model, self.tensor_service_cycles, self.tensor_service_breakdown = self._build_models_for_counts(
            counts=counts,
            use_allocation_bases=True,
        )

    def _is_sim_result_better(self, candidate_sim, reference_sim):
        if reference_sim is None:
            return True

        cand_stall = int(candidate_sim["stall_cycles"])
        ref_stall = int(reference_sim["stall_cycles"])
        if cand_stall < ref_stall:
            return True
        if cand_stall > ref_stall:
            return False

        cand_total = int(candidate_sim["total_cycles"])
        ref_total = int(reference_sim["total_cycles"])
        return cand_total < ref_total

    def _iter_allocation_candidates(self):
        total = int(self.total_banknum)
        if total < 3:
            return

        for ifmap_count in range(1, total - 1):
            for filter_count in range(1, total - ifmap_count):
                ofmap_count = total - ifmap_count - filter_count
                if ofmap_count < 1:
                    continue
                yield {
                    "ifmap": int(ifmap_count),
                    "filter": int(filter_count),
                    "ofmap": int(ofmap_count),
                }

    def _find_oracle_best_allocation(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        best_counts = None
        best_sim = None

        for counts in self._iter_allocation_candidates():
            sim_result = self._simulate_with_counts(
                counts=counts,
                ifmap_demand_mat=ifmap_demand_mat,
                filter_demand_mat=filter_demand_mat,
                ofmap_demand_mat=ofmap_demand_mat,
                use_allocation_bases=True,
            )

            if self._is_sim_result_better(sim_result, best_sim):
                best_counts = dict(counts)
                best_sim = sim_result

        return best_counts, best_sim

    def set_params(self, layer_id=0, word_size=1, verbose=True, config=cfg(), topo=topo(), **kwargs):
        self.layer_id = int(layer_id)
        self.word_size_bytes = int(word_size)
        self.verbose = verbose
        self.config = config
        self.topo = topo

        self.enable_bank_model = bool(self.config.get_enable_bank_model())
        self.enable_dynamic = bool(self.config.get_enable_dynamic())
        self.enable_capacity_penalty = bool(self.config.get_enable_capacity_penalty())
        self.bank_conflict_penalty = max(1, int(self.config.get_bank_conflict_penalty()))
        self.dram_penalty_scale = max(1, int(self.config.get_dram_penalty_scale()))

        self.request_counts = kwargs.get("request_counts", None)
        self.ifmap_demand_for_alloc = kwargs.get("ifmap_demand_mat", None)
        self.filter_demand_for_alloc = kwargs.get("filter_demand_mat", None)
        self.ofmap_demand_for_alloc = kwargs.get("ofmap_demand_mat", None)

        self._compute_tensor_elements()
        self._build_allocation()
        self._init_tensor_models()

        self.total_cycles = 0
        self.stall_cycles = 0
        self.simulation_fell_back_to_static = False
        self.traces_valid = False
        self.params_valid_flag = True

    def set_read_buf_prefetch_matrices(self, ifmap_prefetch_mat=np.zeros((1, 1)), filter_prefetch_mat=np.zeros((1, 1))):
        _ = ifmap_prefetch_mat
        _ = filter_prefetch_mat

    def _build_trace(self, serviced_cycles, demand_mat):
        cycle_col = np.asarray(serviced_cycles, dtype=int).reshape((len(serviced_cycles), 1))
        demand_int = demand_mat.astype(int)
        return np.concatenate((cycle_col, demand_int), axis=1)

    def _adopt_simulation_result(self, sim_result, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        self.ifmap_model = sim_result["ifmap_model"]
        self.filter_model = sim_result["filter_model"]
        self.ofmap_model = sim_result["ofmap_model"]

        self.tensor_service_cycles = dict(sim_result["service_cycles"])
        self.tensor_service_breakdown = dict(sim_result["breakdown"])

        self.ifmap_trace_matrix = self._build_trace(sim_result["ifmap_serviced_cycles"], ifmap_demand_mat)
        self.filter_trace_matrix = self._build_trace(sim_result["filter_serviced_cycles"], filter_demand_mat)
        self.ofmap_trace_matrix = self._build_trace(sim_result["ofmap_serviced_cycles"], ofmap_demand_mat)

        self.stall_cycles = int(sim_result["stall_cycles"])
        self.total_cycles = int(sim_result["total_cycles"])

    def service_memory_requests(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        assert self.params_valid_flag, "Memories not initialized yet"

        self.total_cycles = 0
        self.stall_cycles = 0
        self.simulation_fell_back_to_static = False

        selected_counts = self._get_allocation_counts() # 当前用来仿真的bank分配方案，初始为static分配方案
        selected_sim = self._simulate_with_counts( # 先按照当前方式跑一次仿真，得到stall cycles和total cycles等结果
            counts=selected_counts,
            ifmap_demand_mat=ifmap_demand_mat,
            filter_demand_mat=filter_demand_mat,
            ofmap_demand_mat=ofmap_demand_mat,
            use_allocation_bases=True,
        )

        # Dynamic mode uses an oracle search over all valid static splits and picks the best per layer.
        if self.enable_dynamic:
            best_counts, best_sim = self._find_oracle_best_allocation(
                ifmap_demand_mat=ifmap_demand_mat,
                filter_demand_mat=filter_demand_mat,
                ofmap_demand_mat=ofmap_demand_mat,
            )

            if best_counts is not None and best_sim is not None:
                selected_counts = dict(best_counts)
                selected_sim = best_sim
                self._set_allocation_from_counts(selected_counts)
                self.simulation_fell_back_to_static = bool(
                    self.static_counts is not None and selected_counts == self.static_counts
                )

        self._adopt_simulation_result(selected_sim, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat)

        self._populate_bank_report()
        self.traces_valid = True

    def _calc_start_stop(self, trace_matrix):
        start_cycle = 0
        stop_cycle = 0

        found = False
        for ridx in range(trace_matrix.shape[0]):
            row = trace_matrix[ridx, 1:]
            if np.any(row != -1):
                start_cycle = int(trace_matrix[ridx][0])
                found = True
                break

        if found:
            for ridx in range(trace_matrix.shape[0] - 1, -1, -1):
                row = trace_matrix[ridx, 1:]
                if np.any(row != -1):
                    stop_cycle = int(trace_matrix[ridx][0])
                    break

        return start_cycle, stop_cycle

    def _capacity_util(self, elem_count, bank_count):
        bytes_in_tensor = float(elem_count * self.word_size_bytes)
        capacity_bytes = float(bank_count) * self.bank_capacity_kb * 1024.0
        if capacity_bytes <= 0:
            return 0.0
        return bytes_in_tensor / capacity_bytes

    def _per_bank_cycle_utilization(self, per_bank_access):
        denom = float(self.total_cycles + 1) if self.total_cycles >= 0 else 1.0
        return {
            int(bank_id): float(accesses) / denom
            for bank_id, accesses in per_bank_access.items()
        }

    def _populate_bank_report(self):
        ifmap_banknum = int(self.allocation["ifmap"]["count"])
        filter_banknum = int(self.allocation["filter"]["count"])
        ofmap_banknum = int(self.allocation["ofmap"]["count"])

        ifmap_total_capacity_kb = ifmap_banknum * self.bank_capacity_kb
        filter_total_capacity_kb = filter_banknum * self.bank_capacity_kb
        ofmap_total_capacity_kb = ofmap_banknum * self.bank_capacity_kb

        ifmap_delay = int(self.ifmap_model.total_conflict_delay)
        filter_delay = int(self.filter_model.total_conflict_delay)
        ofmap_delay = int(self.ofmap_model.total_conflict_delay)
        total_delay = ifmap_delay + filter_delay + ofmap_delay

        self.bank_report = {
            "layer_id": int(self.layer_id),
            "EnableBankModel": bool(self.enable_bank_model),
            "EnableDynamic": bool(self.enable_dynamic),
            "EnableCapacityPenalty": bool(self.enable_capacity_penalty),
            "DRAMPenaltyScale": int(self.dram_penalty_scale),
            "dynamic_fallback_to_static": bool(self.simulation_fell_back_to_static),
            "bank_conflict_penalty": int(self.bank_conflict_penalty),
            "total_banknum": int(self.total_banknum),
            "ifmap_banknum": ifmap_banknum,
            "filter_banknum": filter_banknum,
            "ofmap_banknum": ofmap_banknum,
            "allocation_ratio": f"{ifmap_banknum}:{filter_banknum}:{ofmap_banknum}",
            "effective_per_bank_bandwidth": float(self.effective_per_bank_bw),
            "ifmap_cfg_bank_bandwidth": int(self.ifmap_sram_bw_cfg),
            "filter_cfg_bank_bandwidth": int(self.filter_sram_bw_cfg),
            "ofmap_cfg_bank_bandwidth": int(self.ofmap_sram_bw_cfg),
            "bank_capacity_kb": float(self.bank_capacity_kb),
            "ifmap_total_capacity_kb": float(ifmap_total_capacity_kb),
            "filter_total_capacity_kb": float(filter_total_capacity_kb),
            "ofmap_total_capacity_kb": float(ofmap_total_capacity_kb),
            "ifmap_elements": int(self.ifmap_elements),
            "filter_elements": int(self.filter_elements),
            "ofmap_elements": int(self.ofmap_elements),
            "ifmap_capacity_utilization": self._capacity_util(self.ifmap_elements, ifmap_banknum),
            "filter_capacity_utilization": self._capacity_util(self.filter_elements, filter_banknum),
            "ofmap_capacity_utilization": self._capacity_util(self.ofmap_elements, ofmap_banknum),
            "ifmap_service_cycles_per_request": int(self.tensor_service_cycles.get("ifmap", 1)),
            "filter_service_cycles_per_request": int(self.tensor_service_cycles.get("filter", 1)),
            "ofmap_service_cycles_per_request": int(self.tensor_service_cycles.get("ofmap", 1)),
            "ifmap_dram_penalty_cycles_per_request": int(self.tensor_service_breakdown.get("ifmap", {}).get("dram_penalty_cycles", 0)),
            "filter_dram_penalty_cycles_per_request": int(self.tensor_service_breakdown.get("filter", {}).get("dram_penalty_cycles", 0)),
            "ofmap_dram_penalty_cycles_per_request": int(self.tensor_service_breakdown.get("ofmap", {}).get("dram_penalty_cycles", 0)),
            "ifmap_overflow_to_dram": bool(self.tensor_service_breakdown.get("ifmap", {}).get("overflow_to_dram", False)),
            "filter_overflow_to_dram": bool(self.tensor_service_breakdown.get("filter", {}).get("overflow_to_dram", False)),
            "ofmap_overflow_to_dram": bool(self.tensor_service_breakdown.get("ofmap", {}).get("overflow_to_dram", False)),
            "ifmap_overflow_bytes": int(self.tensor_service_breakdown.get("ifmap", {}).get("overflow_bytes", 0)),
            "filter_overflow_bytes": int(self.tensor_service_breakdown.get("filter", {}).get("overflow_bytes", 0)),
            "ofmap_overflow_bytes": int(self.tensor_service_breakdown.get("ofmap", {}).get("overflow_bytes", 0)),
            "ifmap_bank_conflict_delay": ifmap_delay,
            "filter_bank_conflict_delay": filter_delay,
            "ofmap_bank_conflict_delay": ofmap_delay,
            "total_bank_conflict_delay": int(total_delay),
            "total_cycles": int(self.total_cycles),
            "stall_cycles_due_to_bank_conflict": int(self.stall_cycles),
            "per_bank_access_count": {
                "ifmap": dict(self.ifmap_model.per_bank_access),
                "filter": dict(self.filter_model.per_bank_access),
                "ofmap": dict(self.ofmap_model.per_bank_access),
            },
            "per_bank_cycle_utilization": {
                "ifmap": self._per_bank_cycle_utilization(self.ifmap_model.per_bank_access),
                "filter": self._per_bank_cycle_utilization(self.filter_model.per_bank_access),
                "ofmap": self._per_bank_cycle_utilization(self.ofmap_model.per_bank_access),
            },
        }

    def get_bank_report_dict(self):
        assert self.traces_valid, "Traces not generated yet"
        return dict(self.bank_report)

    def get_total_compute_cycles(self):
        assert self.traces_valid, "Traces not generated yet"
        return int(self.total_cycles)

    def get_stall_cycles(self):
        assert self.traces_valid, "Traces not generated yet"
        return int(self.stall_cycles)

    def get_ifmap_sram_start_stop_cycles(self):
        assert self.traces_valid, "Traces not generated yet"
        self.ifmap_sram_start_cycle, self.ifmap_sram_stop_cycle = self._calc_start_stop(self.ifmap_trace_matrix)
        return self.ifmap_sram_start_cycle, self.ifmap_sram_stop_cycle

    def get_filter_sram_start_stop_cycles(self):
        assert self.traces_valid, "Traces not generated yet"
        self.filter_sram_start_cycle, self.filter_sram_stop_cycle = self._calc_start_stop(self.filter_trace_matrix)
        return self.filter_sram_start_cycle, self.filter_sram_stop_cycle

    def get_ofmap_sram_start_stop_cycles(self):
        assert self.traces_valid, "Traces not generated yet"
        self.ofmap_sram_start_cycle, self.ofmap_sram_stop_cycle = self._calc_start_stop(self.ofmap_trace_matrix)
        return self.ofmap_sram_start_cycle, self.ofmap_sram_stop_cycle

    def get_ifmap_dram_details(self):
        assert self.traces_valid, "Traces not generated yet"
        return 0, 0, 0

    def get_filter_dram_details(self):
        assert self.traces_valid, "Traces not generated yet"
        return 0, 0, 0

    def get_ofmap_dram_details(self):
        assert self.traces_valid, "Traces not generated yet"
        return 0, 0, 0

    def get_ifmap_sram_trace_matrix(self):
        assert self.traces_valid, "Traces not generated yet"
        return self.ifmap_trace_matrix

    def get_filter_sram_trace_matrix(self):
        assert self.traces_valid, "Traces not generated yet"
        return self.filter_trace_matrix

    def get_ofmap_sram_trace_matrix(self):
        assert self.traces_valid, "Traces not generated yet"
        return self.ofmap_trace_matrix

    def get_sram_trace_matrices(self):
        assert self.traces_valid, "Traces not generated yet"
        return self.ifmap_trace_matrix, self.filter_trace_matrix, self.ofmap_trace_matrix

    def get_ifmap_dram_trace_matrix(self):
        return np.zeros((0, 2), dtype=int)

    def get_filter_dram_trace_matrix(self):
        return np.zeros((0, 2), dtype=int)

    def get_ofmap_dram_trace_matrix(self):
        return np.zeros((0, 2), dtype=int)

    def get_dram_trace_matrices(self):
        empty = np.zeros((0, 2), dtype=int)
        return empty, empty, empty

    def print_ifmap_sram_trace(self, filename):
        assert self.traces_valid, "Traces not generated yet"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, self.ifmap_trace_matrix, fmt="%i", delimiter=",")

    def print_filter_sram_trace(self, filename):
        assert self.traces_valid, "Traces not generated yet"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, self.filter_trace_matrix, fmt="%i", delimiter=",")

    def print_ofmap_sram_trace(self, filename):
        assert self.traces_valid, "Traces not generated yet"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, self.ofmap_trace_matrix, fmt="%i", delimiter=",")

    def _print_empty_trace(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("")

    def print_ifmap_dram_trace(self, filename):
        self._print_empty_trace(filename)

    def print_filter_dram_trace(self, filename):
        self._print_empty_trace(filename)

    def print_ofmap_dram_trace(self, filename):
        self._print_empty_trace(filename)
