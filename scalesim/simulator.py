"""
This file contains the 'simulator' class that simulates the entire model using the class
'single_layer_sim' and generates the reports (.csv files).
"""

import os
import re
import csv
import heapq
import hashlib

import numpy as np

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from scalesim.layout_utils import layouts as layout
from scalesim.single_layer_sim import single_layer_sim as layer_sim
from scalesim.linear_model.tpu import tpuv4_linear_model, tpuv5e_linear_model, tpuv6e_linear_model


class simulator:
    """
    Class which runs the simulations and manages generated data across various layers
    """
    #
    def __init__(self):
        """
        __init__ method
        """
        self.conf = cfg()
        self.topo = topo()
        self.layout = layout()

        self.top_path = "./"
        self.verbose = True
        self.save_trace = True

        self.num_layers = 0

        self.single_layer_sim_object_list = []
        self.ep_moe_execution_plan = []
        self.ep_moe_groups = []
        self.ep_moe_runtime_states = {}
        self.ep_moe_timeline_rows = []
        self.ep_moe_report_rows = []
        self.ep_moe_bank_allocation_rows = []
        self.ep_moe_routing_rows = []
        self.ep_moe_event_rows = []
        self.ep_moe_chunk_rows = []
        self.input_sources = {}
        self.ep_moe_blackbox_layer_ids = set()

        self.params_set_flag = False
        self.all_layer_run_done = False

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   layout_obj=layout(),
                   top_path="./",
                   verbosity=True,
                   save_trace=True,
                   input_sources=None,
                   ):
        """
        Method to set the run parameters including inputs and parameters for housekeeping.
        """
        self.conf = config_obj
        self.topo = topo_obj
        self.layout = layout_obj

        self.top_path = top_path
        self.verbose = verbosity
        self.save_trace = save_trace
        self.input_sources = dict(input_sources or {})

        # Calculate inferrable parameters here
        self.num_layers = self.topo.get_num_layers()
        self.ep_moe_execution_plan = []
        self.ep_moe_groups = []
        self.ep_moe_runtime_states = {}
        self.ep_moe_timeline_rows = []
        self.ep_moe_report_rows = []
        self.ep_moe_bank_allocation_rows = []
        self.ep_moe_routing_rows = []
        self.ep_moe_event_rows = []
        self.ep_moe_chunk_rows = []
        self.ep_moe_blackbox_layer_ids = set()

        self.params_set_flag = True

    @staticmethod
    def _parse_moe_layer_name(layer_name):
        """Parse legacy MoE-E3-FF2 and explicit MoE-L1-E3-FF2 names."""
        match = re.match(r'^MoE(?:-L(\d+))?-E(\d+)-FF(\d+)$', str(layer_name).strip())
        if match is None:
            return None
        return {
            'moe_layer_id': int(match.group(1)) if match.group(1) is not None else None,
            'expert_id': int(match.group(2)),
            'ffn_part': int(match.group(3)),
        }

    def _build_ep_moe_execution_plan(self):
        """Build a mixed normal-layer / MoE-group plan from topology names.

        In EP mode this plan is consumed by the runtime coordinator. Legacy
        execution remains unchanged when EP mode is disabled.
        """
        plan = []
        layer_names = self.topo.get_layer_names()
        idx = 0

        while idx < len(layer_names):
            parsed = self._parse_moe_layer_name(layer_names[idx])
            if parsed is None:
                plan.append({
                    'type': 'layer',
                    'layer_id': idx,
                    'layer_name': layer_names[idx],
                })
                idx += 1
                continue

            experts = {}
            group_start = idx
            explicit_moe_layer_id = parsed['moe_layer_id']
            while idx < len(layer_names):
                parsed = self._parse_moe_layer_name(layer_names[idx])
                if parsed is None:
                    break
                if parsed['moe_layer_id'] != explicit_moe_layer_id:
                    break

                expert_id = int(parsed['expert_id'])
                ffn_part = int(parsed['ffn_part'])
                gpu_id = expert_id // self.conf.get_experts_per_gpu()
                local_expert_id = expert_id % self.conf.get_experts_per_gpu()
                experts.setdefault(expert_id, {
                    'expert_id': expert_id,
                    'gpu_id': gpu_id,
                    'local_expert_id': local_expert_id,
                    'layers': [],
                })
                experts[expert_id]['layers'].append({
                    'layer_id': idx,
                    'layer_name': layer_names[idx],
                    'ffn_part': ffn_part,
                })
                idx += 1

            group_experts = [experts[k] for k in sorted(experts.keys())]
            group_experts = self._annotate_active_experts(group_experts)

            plan.append({
                'type': 'moe_group',
                'group_id': len([p for p in plan if p['type'] == 'moe_group']),
                'moe_layer_id': (
                    int(explicit_moe_layer_id)
                    if explicit_moe_layer_id is not None
                    else len([p for p in plan if p['type'] == 'moe_group'])
                ),
                'start_layer_id': group_start,
                'end_layer_id': idx - 1,
                'experts': group_experts,
            })

        return self._apply_moe_routing(plan)

    def _validate_ep_moe_execution_plan(self, plan):
        """Fail before simulation when EP topology metadata is inconsistent."""
        groups = [item for item in plan if item['type'] == 'moe_group']
        if not groups:
            raise ValueError(
                "EnableEPMoE=True requires at least one layer named MoE-E<id>-FF<part>"
            )

        num_experts = int(self.conf.get_num_experts())
        expected_ids = set(range(num_experts))
        moe_layer_ids = [int(group['moe_layer_id']) for group in groups]
        if len(moe_layer_ids) != len(set(moe_layer_ids)):
            raise ValueError("Explicit MoELayerID values must be unique within a topology")
        for group in groups:
            seen_ids = set()
            for expert in group['experts']:
                expert_id = int(expert['expert_id'])
                if expert_id in seen_ids:
                    raise ValueError(f"Duplicate ExpertID {expert_id} in MoE group {group['group_id']}")
                seen_ids.add(expert_id)
                if expert_id < 0 or expert_id >= num_experts:
                    raise ValueError(
                        f"ExpertID {expert_id} is outside configured range [0, {num_experts - 1}]"
                    )

                ffn_parts = [int(layer['ffn_part']) for layer in expert['layers']]
                if len(ffn_parts) != len(set(ffn_parts)):
                    raise ValueError(
                        f"ExpertID {expert_id} has duplicate FFN parts in MoE group {group['group_id']}"
                    )
                if sorted(ffn_parts) != [1, 2]:
                    raise ValueError(
                        f"ExpertID {expert_id} must contain exactly FF1 and FF2; got {sorted(ffn_parts)}"
                    )

            if seen_ids != expected_ids:
                missing = sorted(expected_ids - seen_ids)
                raise ValueError(
                    f"MoE group {group['group_id']} does not define every configured expert; missing {missing}"
                )

    def _routing_from_counts(self, counts, top_k):
        """Construct deterministic token assignments matching expert counts."""
        remaining = {int(eid): int(count) for eid, count in counts.items()}
        total_assignments = sum(remaining.values())
        if total_assignments % top_k != 0:
            raise ValueError(
                f"Expert token counts sum to {total_assignments}, which is not divisible by TopK={top_k}"
            )

        assignments = []
        for token_id in range(total_assignments // top_k):
            available = sorted(
                (eid for eid, count in remaining.items() if count > 0),
                key=lambda eid: (-remaining[eid], eid),
            )
            if len(available) < top_k:
                raise ValueError(
                    "Expert token counts cannot form Top-K routes without assigning one token "
                    "to the same expert more than once"
                )
            selected = available[:top_k]
            for expert_id in selected:
                remaining[expert_id] -= 1
            assignments.append((token_id, selected))
        return assignments

    def _load_explicit_routing(self):
        routing_file = self.conf.get_routing_file()
        if not routing_file or not os.path.isfile(routing_file):
            raise ValueError(f"RoutingFile does not exist: {routing_file}")

        by_layer = {}
        with open(routing_file, newline='', encoding='utf-8') as route_file:
            reader = csv.DictReader(route_file)
            required = {'MoELayerID', 'TokenID', 'ExpertIDs'}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    "RoutingFile requires columns: MoELayerID, TokenID, ExpertIDs"
                )
            for row in reader:
                moe_layer_id = int(row['MoELayerID'])
                token_id = int(row['TokenID'])
                raw_experts = str(row['ExpertIDs']).replace(',', '|')
                expert_ids = [int(item.strip()) for item in raw_experts.split('|') if item.strip()]
                by_layer.setdefault(moe_layer_id, []).append((token_id, expert_ids))

        for moe_layer_id, assignments in by_layer.items():
            token_ids = [token_id for token_id, _ in assignments]
            if len(token_ids) != len(set(token_ids)):
                raise ValueError(f"RoutingFile contains duplicate TokenID in MoELayerID {moe_layer_id}")
            assignments.sort(key=lambda item: item[0])
        return by_layer

    def _build_group_routing(self, group, explicit_routes=None):
        expert_ids = [int(expert['expert_id']) for expert in group['experts']]
        num_experts = len(expert_ids)
        top_k = int(self.conf.get_top_k())
        routing_mode = self.conf.get_moe_routing_mode()

        if routing_mode == 'topology_counts':
            counts = {}
            for expert in group['experts']:
                first_layer_id = int(expert['layers'][0]['layer_id'])
                counts[int(expert['expert_id'])] = int(
                    self._get_layer_analytical_stats(first_layer_id)['tokens']
                )
            assignments = self._routing_from_counts(counts, top_k)
        elif routing_mode == 'balanced':
            num_tokens = int(self.conf.get_moe_tokens())
            assignments = []
            for token_id in range(num_tokens):
                start = (token_id * top_k) % num_experts
                selected = [expert_ids[(start + offset) % num_experts] for offset in range(top_k)]
                assignments.append((token_id, selected))
        elif routing_mode == 'seeded_skewed':
            num_tokens = int(self.conf.get_moe_tokens())
            rng = np.random.default_rng(
                int(self.conf.get_routing_seed()) + int(group['moe_layer_id'])
            )
            ranks = np.arange(1, num_experts + 1, dtype=float)
            weights = 1.0 / np.power(ranks, float(self.conf.get_routing_skew_factor()))
            weights = weights / np.sum(weights)
            assignments = []
            for token_id in range(num_tokens):
                selected_idx = rng.choice(num_experts, size=top_k, replace=False, p=weights)
                assignments.append((token_id, [expert_ids[int(idx)] for idx in selected_idx]))
        else:
            moe_layer_id = int(group['moe_layer_id'])
            assignments = list((explicit_routes or {}).get(moe_layer_id, []))
            if not assignments:
                raise ValueError(f"RoutingFile has no routes for MoELayerID {moe_layer_id}")
            configured_tokens = int(self.conf.get_moe_tokens())
            if configured_tokens > 0 and len(assignments) != configured_tokens:
                raise ValueError(
                    f"MoELayerID {moe_layer_id} has {len(assignments)} routed tokens; "
                    f"expected MoETokens={configured_tokens}"
                )

        valid_expert_ids = set(expert_ids)
        counts = {expert_id: 0 for expert_id in expert_ids}
        routing_rows = []
        for token_id, selected in assignments:
            if len(selected) != top_k or len(selected) != len(set(selected)):
                raise ValueError(
                    f"Token {token_id} in MoELayerID {group['moe_layer_id']} must route to "
                    f"exactly {top_k} distinct experts"
                )
            invalid = [expert_id for expert_id in selected if expert_id not in valid_expert_ids]
            if invalid:
                raise ValueError(
                    f"Token {token_id} in MoELayerID {group['moe_layer_id']} routes to invalid experts {invalid}"
                )
            for expert_id in selected:
                counts[expert_id] += 1
            routing_rows.append({
                'MoEGroupID': int(group['group_id']),
                'MoELayerID': int(group['moe_layer_id']),
                'TokenID': int(token_id),
                'ExpertIDs': '|'.join(str(expert_id) for expert_id in selected),
                'TopK': int(top_k),
                'RoutingMode': str(routing_mode),
            })
        return counts, routing_rows

    def _apply_moe_routing(self, plan):
        explicit_routes = None
        if self.conf.get_moe_routing_mode() == 'explicit':
            explicit_routes = self._load_explicit_routing()
            plan_layer_ids = {
                int(item['moe_layer_id']) for item in plan if item['type'] == 'moe_group'
            }
            route_layer_ids = set(int(layer_id) for layer_id in explicit_routes.keys())
            if route_layer_ids != plan_layer_ids:
                raise ValueError(
                    "RoutingFile MoELayerID set does not match topology; "
                    f"topology={sorted(plan_layer_ids)}, routing={sorted(route_layer_ids)}"
                )

        all_routing_rows = []
        for item in plan:
            if item['type'] != 'moe_group':
                continue
            counts, rows = self._build_group_routing(item, explicit_routes=explicit_routes)
            item['experts'] = self._annotate_active_experts(item['experts'], token_counts=counts)
            item['num_routed_tokens'] = len(rows)
            all_routing_rows.extend(rows)
        self.ep_moe_routing_rows = all_routing_rows
        return plan

    def _annotate_active_experts(self, experts, token_counts=None):
        """Mark experts active exactly when routing assigns them tokens."""
        sorted_experts = sorted(experts, key=lambda exp: int(exp['expert_id']))
        if token_counts is None:
            token_counts = {int(exp['expert_id']): 1 for exp in sorted_experts}
        active_ids = {
            int(exp['expert_id'])
            for exp in sorted_experts
            if int(token_counts.get(int(exp['expert_id']), 0)) > 0
        }

        routing_rank = 0
        annotated = []
        for exp in sorted_experts:
            item = dict(exp)
            is_active = int(item['expert_id']) in active_ids
            item['is_active'] = bool(is_active)
            item['tokens_per_expert'] = int(token_counts.get(int(item['expert_id']), 0)) if is_active else 0
            item['routing_rank'] = routing_rank if is_active else -1
            item['routing_policy'] = (
                str(self.conf.get_moe_routing_mode())
            )
            if is_active:
                routing_rank += 1
            annotated.append(item)

        return annotated

    def _build_expert_chunk_plan(self, expert, layer_stats=None):
        """Create the per-expert compute-tile / weight-chunk plan.

        Analytical chunks are retained for black-box GPUs. Detailed-GPU chunks
        are replaced by trace-derived records after layer simulation.
        """
        if layer_stats is None:
            layer_stats = [
                self._get_layer_analytical_stats(int(layer['layer_id']))
                for layer in expert['layers']
            ]

        chunks = []
        global_chunk_id = 0
        blackbox_bw = max(1, int(self.conf.get_blackbox_bandwidth_bytes_per_cycle()))

        for layer, stats in zip(expert['layers'], layer_stats):
            num_tiles = max(1, int(stats['compute_tiles']))
            for tile_id in range(num_tiles):
                weight_bytes = int(stats['weight_chunk_bytes'])
                chunks.append({
                    'chunk_id': int(global_chunk_id),
                    'layer_id': int(layer['layer_id']),
                    'layer_name': str(layer['layer_name']),
                    'ffn_part': int(layer['ffn_part']),
                    'tile_id_in_layer': int(tile_id),
                    'current_weight_chunk': int(global_chunk_id),
                    'weight_bytes': int(weight_bytes),
                    'weight_load_cycles': max(1, int(np.ceil(float(weight_bytes) / float(blackbox_bw)))),
                    'compute_cycles': int(stats['compute_tile_cycles']),
                    'chunk_source': 'analytical',
                    'loaded': False,
                    'prefetched': False,
                    'consumed': False,
                })
                global_chunk_id += 1

        return chunks

    def _get_layer_logical_weight_base(self, layer_id):
        """Return a topology-wide, non-overlapping weight address base."""
        _, filter_offset, _ = self.conf.get_offsets()
        prefix = sum(
            int(self._get_layer_analytical_stats(previous_id)['weight_elems'])
            for previous_id in range(int(layer_id))
        )
        return int(filter_offset) + int(prefix)

    @staticmethod
    def _nonempty_row_runs(matrix):
        """Return half-open runs containing at least one valid demand address."""
        if matrix is None or matrix.size == 0:
            return []
        rows = np.flatnonzero(np.any(matrix != -1, axis=1)).tolist()
        if not rows:
            return []
        runs = []
        start = previous = int(rows[0])
        for row in rows[1:]:
            row = int(row)
            if row != previous + 1:
                runs.append((start, previous + 1))
                start = row
            previous = row
        runs.append((start, previous + 1))
        return runs

    def _extract_detailed_layer_chunk_plan(self, layer_id):
        """Derive compute tiles and weight chunks from a completed layer trace."""
        layer_id = int(layer_id)
        layer_obj = self.single_layer_sim_object_list[layer_id]
        ifmap, filters, ofmap = layer_obj.get_cached_demand_matrices()
        if ifmap is None or filters is None or ofmap is None:
            raise RuntimeError('Demand matrices are unavailable for detailed layer ' + str(layer_id))

        runs = self._nonempty_row_runs(filters)
        if not runs:
            raise RuntimeError('Filter demand trace contains no weight tile for layer ' + str(layer_id))

        layer_name = str(self.topo.get_layer_names()[layer_id])
        parsed = self._parse_moe_layer_name(layer_name) or {}
        _, raw_filter_base, _ = self.conf.get_offsets()
        logical_base = self._get_layer_logical_weight_base(layer_id)
        trace_end = max(ifmap.shape[0], filters.shape[0], ofmap.shape[0])
        precision_bytes = max(1, int(self.conf.get_precision_bytes()))
        bandwidth = max(1, int(self.conf.get_blackbox_bandwidth_bytes_per_cycle()))
        chunks = []

        for tile_id, (weight_start, weight_end) in enumerate(runs):
            tile_end = int(runs[tile_id + 1][0]) if tile_id + 1 < len(runs) else int(trace_end)
            raw_values = filters[weight_start:weight_end]
            raw_addresses = sorted({int(value) for value in raw_values.flat if value != -1})
            layer_weight_elements = int(self._get_layer_analytical_stats(layer_id)['weight_elems'])
            relative_addresses = [address - int(raw_filter_base) for address in raw_addresses]
            if min(relative_addresses) < 0 or max(relative_addresses) >= layer_weight_elements:
                raise RuntimeError(
                    'Detailed filter trace escapes the layer weight address range for layer '
                    + str(layer_id)
                )
            logical_addresses = [logical_base + relative for relative in relative_addresses]
            weight_elements = len(raw_addresses)

            def request_count(matrix):
                return int(np.count_nonzero(matrix[weight_start:min(tile_end, matrix.shape[0])] != -1))

            chunks.append({
                'chunk_id': int(tile_id),
                'layer_id': layer_id,
                'layer_name': layer_name,
                'ffn_part': int(parsed.get('ffn_part', 0)),
                'tile_id_in_layer': int(tile_id),
                'current_weight_chunk': int(tile_id),
                'weight_bytes': int(weight_elements * precision_bytes),
                'weight_load_cycles': max(1, int(np.ceil(float(weight_elements * precision_bytes) / bandwidth))),
                'compute_cycles': max(1, int(tile_end - weight_start)),
                'trace_start_cycle': int(weight_start),
                'trace_end_cycle': int(tile_end),
                'weight_trace_end_cycle': int(weight_end),
                'weight_elements': int(weight_elements),
                'raw_weight_address_min': min(raw_addresses),
                'raw_weight_address_max': max(raw_addresses),
                'logical_weight_address_min': min(logical_addresses),
                'logical_weight_address_max': max(logical_addresses),
                'ifmap_requests': request_count(ifmap),
                'filter_requests': request_count(filters),
                'ofmap_requests': request_count(ofmap),
                'chunk_source': 'detailed_demand_trace',
                'loaded': False,
                'prefetched': False,
                'consumed': False,
            })
        return chunks

    def _refresh_detailed_ep_moe_chunk_plans(self):
        """Replace provisional analytical chunks for detailed-GPU experts."""
        detailed_gpu_id = int(self.conf.get_detailed_gpu_id())
        rows = []
        for group in self.ep_moe_groups:
            group_id = int(group['group_id'])
            for expert in group['experts']:
                if int(expert['gpu_id']) != detailed_gpu_id:
                    continue
                chunks = []
                for layer in expert['layers']:
                    layer_chunks = self._extract_detailed_layer_chunk_plan(int(layer['layer_id']))
                    for chunk in layer_chunks:
                        chunk['chunk_id'] = len(chunks)
                        chunk['current_weight_chunk'] = len(chunks)
                        chunks.append(chunk)
                state = self.ep_moe_runtime_states[group_id][int(expert['expert_id'])]
                initial_count = min(max(1, int(self.conf.get_initial_chunk())), len(chunks))
                state['chunks'] = chunks
                state['chunk_count'] = len(chunks)
                state['initial_chunk_count'] = initial_count
                state['loaded_weight_chunks'] = set(range(initial_count))
                for chunk in chunks[:initial_count]:
                    chunk['loaded'] = True
        rows = []
        for group in self.ep_moe_groups:
            group_id = int(group['group_id'])
            for expert in group['experts']:
                state = self.ep_moe_runtime_states[group_id][int(expert['expert_id'])]
                for chunk in state.get('chunks', []):
                    rows.append(dict(chunk, moe_group_id=group_id,
                                     expert_id=int(expert['expert_id']), gpu_id=int(expert['gpu_id'])))
        self.ep_moe_chunk_rows = rows
        return rows

    def _init_ep_moe_expert_runtime_state(self, group, expert, analytical_stats=None):
        """Initialize one expert state for EP-MoE execution.

        The state is the source of truth for EP runtime timing and reports.
        """
        if analytical_stats is None:
            analytical_stats = self._estimate_blackbox_expert_stats(expert)

        layer_stats = [
            self._get_layer_analytical_stats(int(layer['layer_id']))
            for layer in expert['layers']
        ]
        chunks = self._build_expert_chunk_plan(expert, layer_stats=layer_stats)
        initial_chunk = min(max(1, int(self.conf.get_initial_chunk())), len(chunks))
        loaded_weight_chunks = set(range(initial_chunk))

        for chunk in chunks:
            if int(chunk['chunk_id']) in loaded_weight_chunks:
                chunk['loaded'] = True

        return {
            'moe_group_id': int(group['group_id']),
            'expert_id': int(expert['expert_id']),
            'gpu_id': int(expert['gpu_id']),
            'local_expert_id': int(expert['local_expert_id']),
            'is_active': bool(expert.get('is_active', True)),
            'routing_rank': int(expert.get('routing_rank', 0)),
            'routing_policy': str(expert.get('routing_policy', 'all')),
            'tokens_per_expert': int(analytical_stats['tokens']),
            'hidden_dim': int(analytical_stats['hidden_dim']),
            'current_tile': 0,
            'current_weight_chunk': 0,
            'loaded_weight_chunks': loaded_weight_chunks,
            'prefetched_weight_chunks': set(),
            'consumed_weight_chunks': set(),
            'expert_state': 'ready' if bool(expert.get('is_active', True)) and len(chunks) > 0 else 'inactive',
            'expert_start_time': 0,
            'expert_finish_time': 0,
            'expert_waiting_time': 0,
            'base_expert_cycles': 0,
            'runtime_prefetch_bank_interference_stall': 0,
            'runtime_prefetch_bank_interference_cycles': 0,
            'runtime_prefetch_bank_requests': 0,
            'runtime_blackbox_background_pressure_bytes': 0,
            'runtime_blackbox_background_pressure_cycles': 0,
            'runtime_blackbox_background_pressure_stall': 0,
            'runtime_expert_cycles': 0,
            'runtime_prefetch_hit': 0,
            'runtime_prefetch_hit_rate': 0.0,
            'runtime_prefetch_miss': 0,
            'runtime_prefetch_miss_stall': 0,
            'runtime_weight_loading_stall': 0,
            'runtime_prefetch_bandwidth_overhead': 0,
            'runtime_useful_prefetch_traffic': 0,
            'runtime_useless_prefetch_traffic': 0,
            'runtime_initial_weight_stall': 0,
            'runtime_dispatch_cycles': 0,
            'runtime_combine_cycles': 0,
            'runtime_communication_overlap_cycles': 0,
            'runtime_compute_service_cycles': 0,
            'runtime_dispatch_queue_wait': 0,
            'runtime_combine_queue_wait': 0,
            'chunk_count': int(len(chunks)),
            'initial_chunk_count': int(initial_chunk),
            'prefetch_window': int(self.conf.get_chunk_prefetch_window()),
            'chunks': chunks,
        }

    def _initialize_ep_moe_runtime_states(self):
        """Build runtime-state records for every detected MoE expert."""
        runtime_states = {}
        for group in self.ep_moe_groups:
            group_id = int(group['group_id'])
            runtime_states[group_id] = {}
            for expert in group['experts']:
                analytical_stats = self._estimate_blackbox_expert_stats(expert)
                state = self._init_ep_moe_expert_runtime_state(
                    group=group,
                    expert=expert,
                    analytical_stats=analytical_stats,
                )
                runtime_states[group_id][int(expert['expert_id'])] = state

        self.ep_moe_runtime_states = runtime_states
        return runtime_states

    def _get_layer_total_cycles(self, layer_id):
        if layer_id < 0 or layer_id >= len(self.single_layer_sim_object_list):
            return 0
        comp_items = self.single_layer_sim_object_list[layer_id].get_compute_report_items()
        if not comp_items:
            return 0
        return int(comp_items[0])

    def _get_expert_detailed_cycles(self, expert):
        total_cycles = 0
        for layer in expert['layers']:
            total_cycles += self._get_layer_total_cycles(int(layer['layer_id']))
        return int(total_cycles)

    def _mark_expert_runtime_done(self, state, expert_start, expert_cycles, runtime_components=None):
        """Advance one initialized expert state to completion for the EP timeline."""
        if runtime_components is None:
            runtime_components = {}
        chunk_count = int(state.get('chunk_count', 0))
        consumed_chunks = set(range(chunk_count))
        prefetched_chunks = set(state.get('prefetched_weight_chunks', set()))

        state['current_tile'] = int(chunk_count)
        state['current_weight_chunk'] = int(chunk_count)
        state['loaded_weight_chunks'] = set(range(chunk_count))
        state['prefetched_weight_chunks'] = prefetched_chunks
        state['consumed_weight_chunks'] = consumed_chunks
        state['expert_state'] = 'done'
        state['expert_start_time'] = int(expert_start)
        state['expert_finish_time'] = int(expert_start + expert_cycles)
        state['expert_waiting_time'] = int(state.get('expert_waiting_time', 0))
        state['base_expert_cycles'] = int(runtime_components.get('base_expert_cycles', expert_cycles))
        state['runtime_prefetch_bank_interference_stall'] = int(runtime_components.get('prefetch_bank_interference_stall', 0))
        state['runtime_prefetch_bank_interference_cycles'] = int(runtime_components.get('prefetch_bank_interference_cycles', 0))
        state['runtime_prefetch_bank_requests'] = int(runtime_components.get('prefetch_bank_requests', 0))
        state['runtime_blackbox_background_pressure_bytes'] = int(runtime_components.get('blackbox_background_pressure_bytes', 0))
        state['runtime_blackbox_background_pressure_cycles'] = int(runtime_components.get('blackbox_background_pressure_cycles', 0))
        state['runtime_blackbox_background_pressure_stall'] = int(runtime_components.get('blackbox_background_pressure_stall', 0))
        state['runtime_expert_cycles'] = int(expert_cycles)

        for chunk in state.get('chunks', []):
            chunk_id = int(chunk['chunk_id'])
            chunk['loaded'] = True
            chunk['consumed'] = True
            chunk['prefetched'] = chunk_id in prefetched_chunks

    def _mark_expert_runtime_inactive(self, state):
        state['tokens_per_expert'] = 0
        state['current_tile'] = 0
        state['current_weight_chunk'] = 0
        state['loaded_weight_chunks'] = set()
        state['prefetched_weight_chunks'] = set()
        state['consumed_weight_chunks'] = set()
        state['expert_state'] = 'inactive'
        state['expert_start_time'] = 0
        state['expert_finish_time'] = 0
        state['expert_waiting_time'] = 0
        state['base_expert_cycles'] = 0
        state['runtime_prefetch_bank_interference_stall'] = 0
        state['runtime_prefetch_bank_interference_cycles'] = 0
        state['runtime_prefetch_bank_requests'] = 0
        state['runtime_blackbox_background_pressure_bytes'] = 0
        state['runtime_blackbox_background_pressure_cycles'] = 0
        state['runtime_blackbox_background_pressure_stall'] = 0
        state['runtime_expert_cycles'] = 0
        state['runtime_prefetch_hit'] = 0
        state['runtime_prefetch_hit_rate'] = 0.0
        state['runtime_prefetch_miss'] = 0
        state['runtime_prefetch_miss_stall'] = 0
        state['runtime_weight_loading_stall'] = 0
        state['runtime_prefetch_bandwidth_overhead'] = 0
        state['runtime_useful_prefetch_traffic'] = 0
        state['runtime_useless_prefetch_traffic'] = 0
        state['runtime_initial_weight_stall'] = 0
        state['runtime_dispatch_cycles'] = 0
        state['runtime_combine_cycles'] = 0
        state['runtime_communication_overlap_cycles'] = 0
        state['runtime_compute_service_cycles'] = 0
        state['runtime_dispatch_queue_wait'] = 0
        state['runtime_combine_queue_wait'] = 0

    def _estimate_group_blackbox_background_pressure(self, group):
        if not self.conf.get_enable_blackbox_background_pressure():
            return {
                'blackbox_background_pressure_bytes': 0,
                'blackbox_background_pressure_cycles': 0,
                'blackbox_background_pressure_stall': 0,
            }

        detailed_gpu_id = int(self.conf.get_detailed_gpu_id())
        pressure_bytes = 0
        for expert in group['experts']:
            if not bool(expert.get('is_active', True)):
                continue
            if int(expert['gpu_id']) == detailed_gpu_id:
                continue
            stats = self._estimate_blackbox_expert_stats(expert)
            pressure_bytes += int(stats.get('weight_bytes', 0))
            pressure_bytes += int(stats.get('communication_bytes', 0))

        global_bw = max(1, int(self.conf.get_global_memory_bandwidth_bytes_per_cycle()))
        pressure_cycles = int(np.ceil(float(pressure_bytes) / float(global_bw))) if pressure_bytes > 0 else 0
        return {
            'blackbox_background_pressure_bytes': int(pressure_bytes),
            'blackbox_background_pressure_cycles': int(pressure_cycles),
            'blackbox_background_pressure_stall': int(pressure_cycles),
        }

    def _simulate_ep_moe_group_runtime(self, group, group_start_time):
        """Advance one MoE group with an event-driven compute-engine scheduler."""
        detailed_gpu_id = int(self.conf.get_detailed_gpu_id())
        group_id = int(group['group_id'])
        background_pressure = self._estimate_group_blackbox_background_pressure(group)
        tasks_by_resource = {}
        event_rows = []
        event_sequence = 0
        dispatch_link_ready = int(group_start_time)
        combine_link_ready = int(group_start_time)

        def record_event(cycle, event, expert, state, tile_id='', engine_id='', detail=''):
            nonlocal event_sequence
            event_rows.append({
                'Sequence': int(event_sequence),
                'Cycle': int(cycle),
                'Event': str(event),
                'MoEGroupID': int(group_id),
                'MoELayerID': int(group['moe_layer_id']),
                'ExpertID': int(expert['expert_id']),
                'GPUId': int(expert['gpu_id']),
                'EngineID': engine_id,
                'TileID': tile_id,
                'ExpertState': str(state),
                'Detail': str(detail),
            })
            event_sequence += 1

        for expert in group['experts']:
            expert_id = int(expert['expert_id'])
            is_active = bool(expert.get('is_active', True))
            state = self.ep_moe_runtime_states.get(group_id, {}).get(expert_id)
            if not is_active:
                if state is not None:
                    self._mark_expert_runtime_inactive(state)
                record_event(group_start_time, 'inactive', expert, 'inactive')
                continue

            gpu_id = int(expert['gpu_id'])
            is_detailed = gpu_id == detailed_gpu_id
            analytical_stats = self._estimate_blackbox_expert_stats(expert)
            if state is not None:
                state['expert_state'] = 'dispatch_wait'
            runtime_components = {
                'base_expert_cycles': 0,
                'prefetch_bank_interference_stall': 0,
                'prefetch_bank_interference_cycles': 0,
                'prefetch_bank_requests': 0,
                'blackbox_background_pressure_bytes': 0,
                'blackbox_background_pressure_cycles': 0,
                'blackbox_background_pressure_stall': 0,
            }

            if is_detailed:
                base_cycles = self._get_expert_detailed_cycles(expert)
                detailed_prefetch_bank_stats = self._estimate_detailed_prefetch_bank_stats(expert)
                runtime_components['base_expert_cycles'] = int(base_cycles)
                runtime_components['prefetch_bank_interference_stall'] = int(
                    detailed_prefetch_bank_stats.get('prefetch_bank_interference_stall', 0)
                )
                runtime_components['prefetch_bank_interference_cycles'] = int(
                    detailed_prefetch_bank_stats.get('prefetch_bank_interference_cycles', 0)
                )
                runtime_components['prefetch_bank_requests'] = int(
                    detailed_prefetch_bank_stats.get('prefetch_bank_requests', 0)
                )
                runtime_components['blackbox_background_pressure_bytes'] = int(
                    background_pressure.get('blackbox_background_pressure_bytes', 0)
                )
                runtime_components['blackbox_background_pressure_cycles'] = int(
                    background_pressure.get('blackbox_background_pressure_cycles', 0)
                )
                runtime_components['blackbox_background_pressure_stall'] = int(
                    background_pressure.get('blackbox_background_pressure_stall', 0)
                )
                expert_cycles = int(
                    base_cycles
                    + runtime_components['prefetch_bank_interference_stall']
                    + runtime_components['blackbox_background_pressure_stall']
                )
            else:
                local_work_cycles = int(analytical_stats.get('local_work_cycles', analytical_stats['blackbox_cycles']))
                runtime_components['base_expert_cycles'] = local_work_cycles
                expert_cycles = local_work_cycles

            is_remote = gpu_id != detailed_gpu_id
            dispatch_cycles = int(analytical_stats.get('dispatch_cycles', 0)) if is_remote else 0
            combine_cycles = int(analytical_stats.get('combine_cycles', 0)) if is_remote else 0
            allow_overlap = bool(getattr(self.conf, 'get_allow_comm_prefetch_overlap', lambda: False)())
            dispatch_start = max(int(group_start_time), int(dispatch_link_ready))
            dispatch_ready = int(dispatch_start + dispatch_cycles)
            if dispatch_cycles > 0:
                dispatch_link_ready = dispatch_ready
                record_event(dispatch_start, 'token_dispatch_start', expert, 'dispatch_wait',
                             detail='bytes=' + str(analytical_stats.get('dispatch_bytes', 0)))
                record_event(dispatch_ready, 'token_dispatch_complete', expert, 'initial_weight_load')
            else:
                record_event(group_start_time, 'dispatch_complete', expert, 'initial_weight_load')
            if state is not None:
                state['runtime_dispatch_cycles'] = dispatch_cycles
                state['runtime_combine_cycles'] = combine_cycles
                state['runtime_dispatch_queue_wait'] = int(dispatch_start - group_start_time)

            chunk_count = max(1, int(state.get('chunk_count', 1) if state is not None else 1))
            trace_chunks = state.get('chunks', []) if state is not None else []
            trace_weights = [max(1, int(chunk.get('compute_cycles', 1))) for chunk in trace_chunks]
            if is_detailed and len(trace_weights) == chunk_count and sum(trace_weights) > 0:
                exact = [float(expert_cycles) * weight / sum(trace_weights) for weight in trace_weights]
                tile_durations = [int(value) for value in exact]
                remaining = int(expert_cycles) - sum(tile_durations)
                order = sorted(range(chunk_count), key=lambda idx: (-(exact[idx] - tile_durations[idx]), idx))
                for idx in order[:remaining]:
                    tile_durations[idx] += 1
            else:
                base_duration, extra_cycles = divmod(max(0, int(expert_cycles)), chunk_count)
                tile_durations = [
                    int(base_duration + (1 if tile_id < extra_cycles else 0))
                    for tile_id in range(chunk_count)
                ]
            resource_key = gpu_id if self.conf.get_enable_parallel_moe() else 'global'
            tasks_by_resource.setdefault(resource_key, []).append({
                'expert': expert,
                'state': state,
                'analytical_stats': analytical_stats,
                'runtime_components': runtime_components,
                'tile_durations': tile_durations,
                'first_start': None,
                'finish': int(group_start_time),
                'waiting': 0,
                'weight_waiting': 0,
                'chunk_ready': {},
                'prefetch_requested': set(),
                'prefetch_consumed': set(),
                'dispatch_ready': dispatch_ready,
                'dispatch_start': dispatch_start,
                'initial_load_start': int(group_start_time if allow_overlap else dispatch_ready),
                'communication_overlap': 0,
                'combine_cycles': combine_cycles,
            })

        expert_finish_times = []
        for resource_key in sorted(tasks_by_resource.keys(), key=str):
            tasks = tasks_by_resource[resource_key]
            memory_ready = int(group_start_time)
            for task in sorted(tasks, key=lambda item: int(item['expert']['expert_id'])):
                state = task['state']
                chunks = state.get('chunks', []) if state is not None else []
                if not chunks:
                    task['chunk_ready'] = {
                        tile_id: int(group_start_time)
                        for tile_id in range(len(task['tile_durations']))
                    }
                    record_event(group_start_time, 'initial_weight_ready', task['expert'], 'ready',
                                 detail='no_chunk_metadata')
                    continue
                initial_count = min(int(state.get('initial_chunk_count', 1)), len(chunks)) if state is not None else 0
                for chunk_id in range(initial_count):
                    load_cycles = max(1, int(chunks[chunk_id].get('weight_load_cycles', 1)))
                    request_start = max(int(memory_ready), int(task['initial_load_start']))
                    memory_ready = request_start + load_cycles
                    task['chunk_ready'][chunk_id] = int(memory_ready)
                    record_event(request_start, 'initial_weight_request', task['expert'],
                                 'initial_weight_load', chunk_id, detail='bytes=' + str(chunks[chunk_id].get('weight_bytes', 0)))
                    record_event(memory_ready, 'initial_weight_ready', task['expert'], 'ready', chunk_id)
                    if state is not None:
                        state['runtime_initial_weight_stall'] += load_cycles
                    task['communication_overlap'] += max(
                        0,
                        min(int(memory_ready), int(task['dispatch_ready']))
                        - max(int(request_start), int(task['dispatch_start'])),
                    )

            engine_count = (
                int(self.conf.get_compute_engines_per_gpu())
                if self.conf.get_enable_parallel_moe()
                else 1
            )
            engine_count = min(engine_count, max(1, len(tasks)))
            engines = [(int(group_start_time), engine_id) for engine_id in range(engine_count)]
            heapq.heapify(engines)
            ready_tasks = []
            for task in tasks:
                expert_id = int(task['expert']['expert_id'])
                first_ready = max(
                    int(task['chunk_ready'].get(0, group_start_time)),
                    int(task['dispatch_ready']),
                )
                heapq.heappush(ready_tasks, (first_ready, expert_id, 0, task))

            while ready_tasks:
                ready_cycle, expert_id, tile_id, task = heapq.heappop(ready_tasks)
                engine_ready, engine_id = heapq.heappop(engines)
                state = task['state']
                chunks = state.get('chunks', []) if state is not None else []
                if tile_id not in task['chunk_ready']:
                    chunk = chunks[tile_id] if tile_id < len(chunks) else {}
                    request_start = max(int(ready_cycle), int(memory_ready))
                    load_cycles = max(1, int(chunk.get('weight_load_cycles', 1)))
                    memory_ready = request_start + load_cycles
                    task['chunk_ready'][tile_id] = int(memory_ready)
                    task['weight_waiting'] += max(0, int(memory_ready) - int(ready_cycle))
                    if state is not None:
                        demand_stall = max(0, int(memory_ready) - int(ready_cycle))
                        if int(state.get('prefetch_window', 0)) > 0:
                            state['runtime_prefetch_miss'] += 1
                            state['runtime_prefetch_miss_stall'] += demand_stall
                        else:
                            state['runtime_weight_loading_stall'] += demand_stall
                    record_event(request_start, 'weight_demand_request', task['expert'],
                                 'weight_wait', tile_id, detail='bytes=' + str(chunk.get('weight_bytes', 0)))
                    record_event(memory_ready, 'weight_demand_ready', task['expert'], 'ready', tile_id)
                elif tile_id in task['prefetch_requested']:
                    if task['chunk_ready'][tile_id] <= max(int(ready_cycle), int(engine_ready)):
                        if state is not None:
                            state['runtime_prefetch_hit'] += 1
                    else:
                        stall = task['chunk_ready'][tile_id] - max(int(ready_cycle), int(engine_ready))
                        task['weight_waiting'] += stall
                        if state is not None:
                            state['runtime_prefetch_miss'] += 1
                            state['runtime_prefetch_miss_stall'] += stall
                    task['prefetch_consumed'].add(tile_id)

                start_cycle = max(int(ready_cycle), int(engine_ready), int(task['chunk_ready'].get(tile_id, ready_cycle)))
                duration = int(task['tile_durations'][tile_id])
                finish_cycle = int(start_cycle + duration)
                task['waiting'] += int(start_cycle - ready_cycle)
                if task['first_start'] is None:
                    task['first_start'] = int(start_cycle)

                if state is not None:
                    state['expert_state'] = 'computing'
                    state['current_tile'] = int(tile_id)
                    state['current_weight_chunk'] = int(tile_id)
                record_event(start_cycle, 'compute_tile_start', task['expert'], 'computing', tile_id, engine_id)
                record_event(finish_cycle, 'compute_tile_complete', task['expert'], 'ready', tile_id, engine_id)

                prefetch_window = int(state.get('prefetch_window', 0)) if state is not None else 0
                target_id = tile_id + prefetch_window
                if prefetch_window > 0 and target_id < len(chunks) and target_id not in task['chunk_ready']:
                    target = chunks[target_id]
                    request_start = max(int(start_cycle), int(memory_ready))
                    load_cycles = max(1, int(target.get('weight_load_cycles', 1)))
                    memory_ready = request_start + load_cycles
                    task['chunk_ready'][target_id] = int(memory_ready)
                    task['prefetch_requested'].add(target_id)
                    if state is not None:
                        state['prefetched_weight_chunks'].add(target_id)
                        state['runtime_prefetch_bandwidth_overhead'] += int(target.get('weight_bytes', 0))
                    record_event(request_start, 'weight_prefetch_request', task['expert'],
                                 'prefetching', target_id, detail='bytes=' + str(target.get('weight_bytes', 0)))
                    record_event(memory_ready, 'weight_prefetch_ready', task['expert'], 'ready', target_id)

                heapq.heappush(engines, (finish_cycle, engine_id))
                next_tile = tile_id + 1
                if next_tile < len(task['tile_durations']):
                    heapq.heappush(ready_tasks, (finish_cycle, expert_id, next_tile, task))
                else:
                    task['finish'] = int(finish_cycle)

            for task in tasks:
                expert_start = int(task['first_start'] if task['first_start'] is not None else group_start_time)
                compute_finish = int(task['finish'])
                combine_start = max(compute_finish, int(combine_link_ready))
                expert_finish = int(combine_start + int(task.get('combine_cycles', 0)))
                if int(task.get('combine_cycles', 0)) > 0:
                    combine_link_ready = expert_finish
                runtime_cycles = int(expert_finish - expert_start)
                state = task['state']
                if state is not None:
                    self._mark_expert_runtime_done(
                        state=state,
                        expert_start=expert_start,
                        expert_cycles=runtime_cycles,
                        runtime_components=task['runtime_components'],
                    )
                    state['expert_finish_time'] = expert_finish
                    state['expert_waiting_time'] = int(task['waiting'])
                    state['expert_waiting_time'] += int(task['weight_waiting'])
                    state['runtime_expert_cycles'] = runtime_cycles
                    state['runtime_compute_service_cycles'] = int(sum(task['tile_durations']))
                    state['runtime_combine_queue_wait'] = int(combine_start - compute_finish)
                    state['runtime_communication_overlap_cycles'] = int(task['communication_overlap'])
                    state['runtime_useful_prefetch_traffic'] = sum(
                        int(state['chunks'][chunk_id].get('weight_bytes', 0))
                        for chunk_id in task['prefetch_consumed']
                    )
                    unused = task['prefetch_requested'] - task['prefetch_consumed']
                    state['runtime_useless_prefetch_traffic'] = sum(
                        int(state['chunks'][chunk_id].get('weight_bytes', 0))
                        for chunk_id in unused
                    )
                    prefetch_hits = int(state.get('runtime_prefetch_hit', 0))
                    prefetch_misses = int(state.get('runtime_prefetch_miss', 0))
                    prefetch_lookups = prefetch_hits + prefetch_misses
                    state['runtime_prefetch_hit_rate'] = (
                        float(prefetch_hits) / float(prefetch_lookups)
                        if prefetch_lookups > 0 else 0.0
                    )
                if int(task.get('combine_cycles', 0)) > 0:
                    record_event(combine_start, 'output_combine_start', task['expert'], 'combine',
                                 detail='bytes=' + str(task['analytical_stats'].get('combine_bytes', 0)))
                    record_event(expert_finish, 'output_combine_complete', task['expert'], 'done')
                record_event(expert_finish, 'expert_done', task['expert'], 'done')
                expert_finish_times.append(expert_finish)

        event_rows.sort(key=lambda row: (int(row['Cycle']), int(row['Sequence'])))
        self.ep_moe_event_rows.extend(event_rows)
        group_finish_time = max(expert_finish_times) if expert_finish_times else int(group_start_time)
        return int(group_finish_time)

    def _simulate_ep_moe_runtime_timeline(self):
        """Build the mixed normal-layer / MoE-group timeline.

        Normal layers advance sequentially. A MoE group advances once using
        parallel expert completion time: group_finish = max(expert_finish).
        """
        current_time = 0
        timeline_rows = []
        self.ep_moe_event_rows = []

        for item in self.ep_moe_execution_plan:
            if item['type'] == 'layer':
                layer_id = int(item['layer_id'])
                start_time = int(current_time)
                cycles = self._get_layer_total_cycles(layer_id)
                finish_time = int(start_time + cycles)
                timeline_rows.append({
                    'TimelineType': 'layer',
                    'MoEGroupID': '',
                    'MoELayerID': '',
                    'LayerID': layer_id,
                    'LayerName': str(item['layer_name']),
                    'StartCycle': start_time,
                    'FinishCycle': finish_time,
                    'DurationCycles': int(cycles),
                    'NumExperts': 0,
                    'NumActiveExperts': 0,
                    'ParallelExecution': False,
                })
                current_time = finish_time
                continue

            group = item
            start_time = int(current_time)
            finish_time = self._simulate_ep_moe_group_runtime(group, start_time)
            cycles = int(finish_time - start_time)
            active_experts = [exp for exp in group['experts'] if bool(exp.get('is_active', True))]
            timeline_rows.append({
                'TimelineType': 'moe_group',
                'MoEGroupID': int(group['group_id']),
                'MoELayerID': int(group['moe_layer_id']),
                'LayerID': str(group['start_layer_id']) + '-' + str(group['end_layer_id']),
                'LayerName': 'MoEGroup' + str(group['group_id']),
                'StartCycle': start_time,
                'FinishCycle': finish_time,
                'DurationCycles': cycles,
                'NumExperts': int(len(group['experts'])),
                'NumActiveExperts': int(len(active_experts)),
                'ParallelExecution': bool(self.conf.get_enable_parallel_moe()),
            })
            current_time = finish_time

        self.ep_moe_timeline_rows = timeline_rows
        return timeline_rows

    #
    def run(self):
        """
        Method to run scalesim simulation for all layers. This method first runs compute and memory
        simulations for each layer and gathers the required stats. Once the simulation runs are
        done, it gathers the stats from single_layer_sim objects and calls generate_report() method
        to create the report files. If save_trace flag is set, then layer wise traces are saved as
        well.
        """
        assert self.params_set_flag, 'Simulator parameters are not set'

        if self.conf.get_enable_ep_moe():
            self.ep_moe_execution_plan = self._build_ep_moe_execution_plan()
            self._validate_ep_moe_execution_plan(self.ep_moe_execution_plan)
            self.ep_moe_groups = [
                item for item in self.ep_moe_execution_plan
                if item['type'] == 'moe_group'
            ]
            detailed_gpu_id = int(self.conf.get_detailed_gpu_id())
            self.ep_moe_blackbox_layer_ids = {
                int(layer['layer_id'])
                for group in self.ep_moe_groups
                for expert in group['experts']
                if int(expert['gpu_id']) != detailed_gpu_id
                for layer in expert['layers']
            }
            self._initialize_ep_moe_runtime_states()
            if self.verbose:
                print('EP-MoE mode enabled')
                print('EP-MoE groups detected: ' + str(len(self.ep_moe_groups)))
                for group in self.ep_moe_groups:
                    expert_ids = [str(exp['expert_id']) for exp in group['experts']]
                    active_ids = [str(exp['expert_id']) for exp in group['experts'] if bool(exp.get('is_active', True))]
                    print('  MoE group ' + str(group['group_id']) + ': experts ' + ','.join(expert_ids)
                          + ' active ' + ','.join(active_ids))

        # 1. Create the layer runners for each layer
        for i in range(self.num_layers):
            this_layer_sim = layer_sim()
            this_layer_sim.set_params(layer_id=i,
                                 config_obj=self.conf,
                                 topology_obj=self.topo,
                                 layout_obj=self.layout,
                                 verbose=self.verbose)

            self.single_layer_sim_object_list.append(this_layer_sim)

        if not os.path.isdir(self.top_path):
            os.mkdir(self.top_path)

        report_path = self.top_path + '/' + self.conf.get_run_name()

        if not os.path.isdir(report_path):
            os.mkdir(report_path)

        self.top_path = report_path

        # 2. Run each layer
        # TODO: This is parallelizable
        for single_layer_obj in self.single_layer_sim_object_list:
            layer_id = single_layer_obj.get_layer_id()
            if layer_id in self.ep_moe_blackbox_layer_ids:
                if self.verbose:
                    print('\nSkipping analytical black-box Layer ' + str(layer_id))
                continue

            if self.verbose:
                print('\nRunning Layer ' + str(layer_id))

            single_layer_obj.run()

            if self.verbose:
                comp_items = single_layer_obj.get_compute_report_items()
                total_cycles = comp_items[0]
                comp_cycles = comp_items[1]
                stall_cycles = comp_items[2]
                util = comp_items[3]
                mapping_eff = comp_items[4]
                print('Total cycles: ' + str(total_cycles))
                print('Compute cycles: ' + str(comp_cycles))
                print('Stall cycles: ' + str(stall_cycles))
                print('Overall utilization: ' + "{:.2f}".format(util) +'%')
                print('Mapping efficiency: ' + "{:.2f}".format(mapping_eff) +'%')

                avg_bw_items = single_layer_obj.get_bandwidth_report_items()
                if self.conf.sparsity_support is True:
                    avg_ifmap_sram_bw = avg_bw_items[0]
                    avg_filter_sram_bw = avg_bw_items[1]
                    avg_filter_metadata_sram_bw = avg_bw_items[2]
                    avg_ofmap_sram_bw = avg_bw_items[3]
                    avg_ifmap_dram_bw = avg_bw_items[4]
                    avg_filter_dram_bw = avg_bw_items[5]
                    avg_ofmap_dram_bw = avg_bw_items[6]
                else:
                    avg_ifmap_sram_bw = avg_bw_items[0]
                    avg_filter_sram_bw = avg_bw_items[1]
                    avg_ofmap_sram_bw = avg_bw_items[2]
                    avg_ifmap_dram_bw = avg_bw_items[3]
                    avg_filter_dram_bw = avg_bw_items[4]
                    avg_ofmap_dram_bw = avg_bw_items[5]

                print('Average IFMAP SRAM BW: ' + "{:.3f}".format(avg_ifmap_sram_bw) + \
                      ' words/cycle')
                print('Average Filter SRAM BW: ' + "{:.3f}".format(avg_filter_sram_bw) + \
                      ' words/cycle')
                if self.conf.sparsity_support is True:
                    print('Average Filter Metadata SRAM BW: ' + \
                          "{:.3f}".format(avg_filter_metadata_sram_bw) + ' words/cycle')
                print('Average OFMAP SRAM BW: ' + "{:.3f}".format(avg_ofmap_sram_bw) + \
                      ' words/cycle')
                print('Average IFMAP DRAM BW: ' + "{:.3f}".format(avg_ifmap_dram_bw) + \
                      ' words/cycle')
                print('Average Filter DRAM BW: ' + "{:.3f}".format(avg_filter_dram_bw) + \
                      ' words/cycle')
                print('Average OFMAP DRAM BW: ' + "{:.3f}".format(avg_ofmap_dram_bw) + \
                      ' words/cycle')

            if self.save_trace:
                if self.verbose:
                    print('Saving traces: ', end='')
                single_layer_obj.save_traces(self.top_path)
                if self.verbose:
                    print('Done!')

        self.all_layer_run_done = True

        # Apply lightweight next-layer prefetch experiment model (bank-conflict SRAM only).
        # This is a cycle-level abstraction for prefetch-bank allocation co-design, not a
        # detailed DRAM/SRAM prefetcher implementation.
        if not self.conf.get_enable_ep_moe():
            self._apply_prefetch_experiment_model()

        if self.conf.get_enable_ep_moe():
            self._refresh_detailed_ep_moe_chunk_plans()
            self._simulate_ep_moe_runtime_timeline()
            self._compute_ep_moe_report_rows()

        self.generate_reports()

    def _get_layer_analytical_stats(self, layer_id):
        """Estimate layer-level analytical workload quantities from topology."""
        arr_h, arr_w = self.conf.get_array_dims()
        mac_units = max(1, int(arr_h) * int(arr_w))
        mac_ops = int(self.topo.get_layer_mac_ops(layer_id=layer_id))
        num_filters = max(1, int(self.topo.get_layer_num_filters(layer_id=layer_id)))
        tokens = max(1, int(self.topo.get_layer_num_ofmap_px(layer_id=layer_id) / num_filters))
        hidden_dim = max(1, int(self.topo.get_layer_window_size(layer_id=layer_id)))
        output_dim = num_filters
        weight_elems = int(hidden_dim * output_dim)
        precision_bytes = max(1, int(self.conf.get_precision_bytes()))
        weight_bytes = int(weight_elems * precision_bytes)
        blackbox_bw = max(1, int(self.conf.get_blackbox_bandwidth_bytes_per_cycle()))
        compute_tiles = max(
            1,
            int(np.ceil(float(tokens) / float(max(1, arr_h))))
            * int(np.ceil(float(output_dim) / float(max(1, arr_w))))
        )
        compute_cycles = max(1, int(np.ceil(float(mac_ops) / float(mac_units))))

        return {
            'tokens': int(tokens),
            'hidden_dim': int(hidden_dim),
            'output_dim': int(output_dim),
            'mac_ops': int(mac_ops),
            'weight_elems': int(weight_elems),
            'weight_bytes': int(weight_bytes),
            'compute_tiles': int(compute_tiles),
            'compute_tile_cycles': max(1, int(np.ceil(float(compute_cycles) / float(compute_tiles)))),
            'weight_chunk_bytes': max(1, int(np.ceil(float(weight_bytes) / float(compute_tiles)))),
            'compute_cycles': int(compute_cycles),
            'weight_bw_cycles': max(1, int(np.ceil(float(weight_bytes) / float(blackbox_bw)))),
        }

    def _estimate_communication_breakdown(self, tokens, hidden_dim, overlap_budget_cycles=0):
        input_bytes_per_elem = int(self.conf.get_communication_input_bytes_per_elem())
        output_bytes_per_elem = int(self.conf.get_communication_output_bytes_per_elem())
        comm_latency = int(self.conf.get_communication_latency_cycles())
        comm_bw = max(1, int(self.conf.get_communication_bandwidth_bytes_per_cycle()))

        dispatch_bytes = int(tokens * hidden_dim * input_bytes_per_elem)
        combine_bytes = int(tokens * hidden_dim * output_bytes_per_elem)
        dispatch_cycles = int(comm_latency + np.ceil(float(dispatch_bytes) / float(comm_bw))) if dispatch_bytes > 0 else 0
        combine_cycles = int(comm_latency + np.ceil(float(combine_bytes) / float(comm_bw))) if combine_bytes > 0 else 0
        total_cycles = int(dispatch_cycles + combine_cycles)

        overlap_mode = str(self.conf.get_communication_overlap_mode())
        if not self.conf.get_enable_communication_overlap():
            overlap_mode = 'none'

        overlap_budget = max(0, int(overlap_budget_cycles))
        if overlap_mode == 'none':
            overlap_cycles = 0
        elif overlap_mode == 'full':
            overlap_cycles = min(total_cycles, overlap_budget)
        else:
            overlap_cycles = min(combine_cycles, overlap_budget)

        exposed_cycles = max(0, total_cycles - overlap_cycles)

        return {
            'dispatch_bytes': int(dispatch_bytes),
            'combine_bytes': int(combine_bytes),
            'communication_bytes': int(dispatch_bytes + combine_bytes),
            'dispatch_cycles': int(dispatch_cycles),
            'combine_cycles': int(combine_cycles),
            'communication_cycles': int(total_cycles),
            'communication_overlap_cycles': int(overlap_cycles),
            'exposed_communication_cycles': int(exposed_cycles),
            'communication_overlap_mode': str(overlap_mode),
        }

    def _estimate_blackbox_expert_stats(self, expert):
        """Analytical black-box estimate for non-detailed GPU experts.

        Black-box GPUs do not participate in GPU0's detailed bank-conflict simulation.
        The estimate is intentionally coarse and exposes compute, weight traffic,
        and communication as separate runtime inputs.
        """
        layer_stats = [
            self._get_layer_analytical_stats(int(layer['layer_id']))
            for layer in expert['layers']
        ]

        has_routed_tokens = 'tokens_per_expert' in expert
        routed_tokens = int(expert.get('tokens_per_expert', 0))
        if has_routed_tokens and routed_tokens > 0:
            arr_h, arr_w = self.conf.get_array_dims()
            mac_units = max(1, int(arr_h) * int(arr_w))
            adjusted_stats = []
            for stats in layer_stats:
                item = dict(stats)
                item['tokens'] = routed_tokens
                item['mac_ops'] = int(routed_tokens * item['hidden_dim'] * item['output_dim'])
                item['compute_tiles'] = max(
                    1,
                    int(np.ceil(float(routed_tokens) / float(max(1, arr_h))))
                    * int(np.ceil(float(item['output_dim']) / float(max(1, arr_w)))),
                )
                item['compute_cycles'] = max(
                    1, int(np.ceil(float(item['mac_ops']) / float(mac_units)))
                )
                item['compute_tile_cycles'] = max(
                    1,
                    int(np.ceil(float(item['compute_cycles']) / float(item['compute_tiles']))),
                )
                item['weight_chunk_bytes'] = max(
                    1, int(np.ceil(float(item['weight_bytes']) / float(item['compute_tiles'])))
                )
                adjusted_stats.append(item)
            layer_stats = adjusted_stats

        tokens = routed_tokens if has_routed_tokens else max([s['tokens'] for s in layer_stats] or [0])
        dims = []
        for s in layer_stats:
            dims.append(int(s['hidden_dim']))
            dims.append(int(s['output_dim']))
        hidden_dim = min(dims) if dims else 0

        mac_ops = sum(int(s['mac_ops']) for s in layer_stats)
        weight_bytes = sum(int(s['weight_bytes']) for s in layer_stats)
        compute_cycles = sum(int(s['compute_cycles']) for s in layer_stats)
        weight_bw_cycles = sum(int(s['weight_bw_cycles']) for s in layer_stats)
        communication = self._estimate_communication_breakdown(
            tokens=tokens,
            hidden_dim=hidden_dim,
            overlap_budget_cycles=int(compute_cycles),
        )

        # Weight load and prefetch timing is scheduled exactly once by the
        # runtime chunk-memory timeline. Keep analytical local work compute-only.
        local_work_cycles = int(compute_cycles)
        total_cycles = int(local_work_cycles + communication['exposed_communication_cycles'])

        return {
            'tokens': int(tokens),
            'hidden_dim': int(hidden_dim),
            'mac_ops': int(mac_ops),
            'weight_bytes': int(weight_bytes),
            'compute_cycles': int(compute_cycles),
            'weight_bw_cycles': int(weight_bw_cycles),
            'communication_bytes': int(communication['communication_bytes']),
            'dispatch_bytes': int(communication['dispatch_bytes']),
            'combine_bytes': int(communication['combine_bytes']),
            'dispatch_cycles': int(communication['dispatch_cycles']),
            'combine_cycles': int(communication['combine_cycles']),
            'communication_cycles': int(communication['communication_cycles']),
            'communication_overlap_cycles': int(communication['communication_overlap_cycles']),
            'exposed_communication_cycles': int(communication['exposed_communication_cycles']),
            'communication_overlap_mode': str(communication['communication_overlap_mode']),
            'local_work_cycles': int(local_work_cycles),
            'blackbox_cycles': int(max(1, total_cycles)),
        }

    def _estimate_detailed_prefetch_bank_stats(self, expert):
        """Run detailed GPU prefetch traffic through the existing bank model.

        This is a filter-only prefetch estimate. It reuses the per-layer bank split
        selected by the normal detailed run and does not mutate the main memory model.
        Normal and prefetch filter demands are also combined on the same request
        line to estimate prefetch interference with regular IA/W/OA traffic.
        """
        if (not self.conf.get_enable_bank_model()) or self.conf.get_chunk_prefetch_window() <= 0:
            return {
                'prefetch_bank_model_cycles': 0,
                'prefetch_bank_conflict_cycles': 0,
                'prefetch_bank_requests': 0,
                'combined_bank_model_cycles': 0,
                'combined_bank_conflict_cycles': 0,
                'prefetch_bank_interference_stall': 0,
                'prefetch_bank_interference_cycles': 0,
            }

        total_cycles = 0
        total_stall = 0
        total_requests = 0
        total_combined_cycles = 0
        total_combined_stall = 0
        total_interference_stall = 0
        total_interference_cycles = 0
        prefetch_window = max(1, int(self.conf.get_chunk_prefetch_window()))

        for layer in expert['layers']:
            layer_id = int(layer['layer_id'])
            layer_obj = self.single_layer_sim_object_list[layer_id]
            if not hasattr(layer_obj, 'get_cached_demand_matrices'):
                continue
            ifmap_demand, filter_demand, ofmap_demand = layer_obj.get_cached_demand_matrices()
            if ifmap_demand is None or filter_demand is None or ofmap_demand is None:
                continue

            bank_items = layer_obj.get_bank_report_items() if hasattr(layer_obj, 'get_bank_report_items') else {}
            counts = {
                'ifmap': int(bank_items.get('ifmap_banknum', getattr(self.conf, 'ifmap_sram_bank_num', 1))),
                'filter': int(bank_items.get('filter_banknum', getattr(self.conf, 'filter_sram_bank_num', 1))),
                'ofmap': int(bank_items.get('ofmap_banknum', getattr(self.conf, 'ofmap_sram_bank_num', 1))),
            }

            ifmap_pf = np.full_like(ifmap_demand, -1)
            filter_pf = np.full_like(filter_demand, -1)
            if prefetch_window < filter_demand.shape[0]:
                filter_pf[:-prefetch_window, :] = filter_demand[prefetch_window:, :]
            ofmap_pf = np.full_like(ofmap_demand, -1)
            total_requests += int(np.count_nonzero(filter_pf != -1))

            sim_pf = layer_obj.memory_system.simulate_with_explicit_counts(
                counts=counts,
                ifmap_demand_mat=ifmap_pf,
                filter_demand_mat=filter_pf,
                ofmap_demand_mat=ofmap_pf,
                use_allocation_bases=True,
            )
            total_cycles += int(sim_pf.get('total_cycles', 0))
            total_stall += int(sim_pf.get('stall_cycles', 0))

            normal_sim = layer_obj.memory_system.simulate_with_explicit_counts(
                counts=counts,
                ifmap_demand_mat=ifmap_demand,
                filter_demand_mat=filter_demand,
                ofmap_demand_mat=ofmap_demand,
                use_allocation_bases=True,
            )
            if hasattr(layer_obj.memory_system, 'simulate_with_filter_prefetch_counts'):
                combined_sim = layer_obj.memory_system.simulate_with_filter_prefetch_counts(
                    counts=counts,
                    ifmap_demand_mat=ifmap_demand,
                    filter_demand_mat=filter_demand,
                    ofmap_demand_mat=ofmap_demand,
                    filter_prefetch_demand_mat=filter_pf,
                    use_allocation_bases=True,
                    prefetch_priority='low',
                )
            else:
                combined_filter = np.concatenate((filter_demand, filter_pf), axis=1)
                combined_sim = layer_obj.memory_system.simulate_with_explicit_counts(
                    counts=counts,
                    ifmap_demand_mat=ifmap_demand,
                    filter_demand_mat=combined_filter,
                    ofmap_demand_mat=ofmap_demand,
                    use_allocation_bases=True,
                )
            normal_cycles = int(normal_sim.get('total_cycles', 0))
            normal_stall = int(normal_sim.get('stall_cycles', 0))
            combined_cycles = int(combined_sim.get('total_cycles', 0))
            combined_stall = int(combined_sim.get('stall_cycles', 0))

            total_combined_cycles += combined_cycles
            total_combined_stall += combined_stall
            total_interference_cycles += max(0, combined_cycles - normal_cycles)
            total_interference_stall += max(
                0,
                int(combined_sim.get('prefetch_interference_stall', combined_stall - normal_stall))
            )

        return {
            'prefetch_bank_model_cycles': int(total_cycles),
            'prefetch_bank_conflict_cycles': int(total_stall),
            'prefetch_bank_requests': int(total_requests),
            'combined_bank_model_cycles': int(total_combined_cycles),
            'combined_bank_conflict_cycles': int(total_combined_stall),
            'prefetch_bank_interference_stall': int(total_interference_stall),
            'prefetch_bank_interference_cycles': int(total_interference_cycles),
        }

    def _build_ep_moe_bank_allocation_rows(self):
        """Collect per-layer bank-allocation rows for detailed EP-MoE experts."""
        rows = []
        detailed_gpu_id = int(self.conf.get_detailed_gpu_id())
        prefetch_window = max(0, int(self.conf.get_chunk_prefetch_window()))

        for group in self.ep_moe_groups:
            group_id = int(group['group_id'])
            for expert in group['experts']:
                expert_id = int(expert['expert_id'])
                gpu_id = int(expert['gpu_id'])
                is_active = bool(expert.get('is_active', True))
                is_detailed = gpu_id == detailed_gpu_id

                if not is_detailed:
                    continue

                for layer in expert['layers']:
                    layer_id = int(layer['layer_id'])
                    layer_obj = self.single_layer_sim_object_list[layer_id]
                    bank_items = layer_obj.get_bank_report_items() if hasattr(layer_obj, 'get_bank_report_items') else {}
                    comp_items = layer_obj.get_compute_report_items()

                    counts = {
                        'ifmap': int(bank_items.get('ifmap_banknum', getattr(self.conf, 'ifmap_sram_bank_num', 1))),
                        'filter': int(bank_items.get('filter_banknum', getattr(self.conf, 'filter_sram_bank_num', 1))),
                        'ofmap': int(bank_items.get('ofmap_banknum', getattr(self.conf, 'ofmap_sram_bank_num', 1))),
                    }

                    prefetch_requests = 0
                    normal_stall = 0
                    prefetch_interference_stall = 0
                    prefetch_interference_cycles = 0
                    combined_cycles = 0
                    combined_stall = 0

                    if is_active and self.conf.get_enable_bank_model() and prefetch_window > 0 and hasattr(layer_obj, 'get_cached_demand_matrices'):
                        ifmap_demand, filter_demand, ofmap_demand = layer_obj.get_cached_demand_matrices()
                        if ifmap_demand is not None and filter_demand is not None and ofmap_demand is not None:
                            filter_pf = np.full_like(filter_demand, -1)
                            if prefetch_window < filter_demand.shape[0]:
                                filter_pf[:-prefetch_window, :] = filter_demand[prefetch_window:, :]
                            prefetch_requests = int(np.count_nonzero(filter_pf != -1))

                            normal_sim = layer_obj.memory_system.simulate_with_explicit_counts(
                                counts=counts,
                                ifmap_demand_mat=ifmap_demand,
                                filter_demand_mat=filter_demand,
                                ofmap_demand_mat=ofmap_demand,
                                use_allocation_bases=True,
                            )
                            combined_sim = layer_obj.memory_system.simulate_with_filter_prefetch_counts(
                                counts=counts,
                                ifmap_demand_mat=ifmap_demand,
                                filter_demand_mat=filter_demand,
                                ofmap_demand_mat=ofmap_demand,
                                filter_prefetch_demand_mat=filter_pf,
                                use_allocation_bases=True,
                                prefetch_priority='low',
                            )
                            normal_stall = int(normal_sim.get('stall_cycles', 0))
                            combined_cycles = int(combined_sim.get('total_cycles', 0))
                            combined_stall = int(combined_sim.get('stall_cycles', 0))
                            prefetch_interference_stall = int(combined_sim.get('prefetch_interference_stall', 0))
                            prefetch_interference_cycles = max(0, combined_cycles - int(normal_sim.get('total_cycles', 0)))

                    rows.append({
                        'MoEGroupID': group_id,
                        'ExpertID': expert_id,
                        'GPUId': gpu_id,
                        'LocalExpertID': int(expert['local_expert_id']),
                        'IsActiveExpert': bool(is_active),
                        'LayerID': layer_id,
                        'LayerName': str(layer['layer_name']),
                        'BankAllocationMode': 'dynamic' if self.conf.get_enable_dynamic() else 'static',
                        'EnableDynamic': bool(bank_items.get('EnableDynamic', self.conf.get_enable_dynamic())),
                        'DynamicFallbackToStatic': bool(bank_items.get('EnableDynamic', self.conf.get_enable_dynamic()))
                                                   and bool(bank_items.get('dynamic_fallback_to_static', False)),
                        'AllocationRatio': str(bank_items.get('allocation_ratio', '')),
                        'IfmapBankNum': int(counts['ifmap']),
                        'FilterBankNum': int(counts['filter']),
                        'OfmapBankNum': int(counts['ofmap']),
                        'StaticIfmapBankNum': int(getattr(self.conf, 'ifmap_sram_bank_num', counts['ifmap'])),
                        'StaticFilterBankNum': int(getattr(self.conf, 'filter_sram_bank_num', counts['filter'])),
                        'StaticOfmapBankNum': int(getattr(self.conf, 'ofmap_sram_bank_num', counts['ofmap'])),
                        'BitCapacityTargetIfmapBankNum': int(bank_items.get('bit_capacity_target_ifmap_banknum', -1)),
                        'BitCapacityTargetFilterBankNum': int(bank_items.get('bit_capacity_target_filter_banknum', -1)),
                        'BitCapacityTargetOfmapBankNum': int(bank_items.get('bit_capacity_target_ofmap_banknum', -1)),
                        'EffectivePerBankBandwidth': float(bank_items.get('effective_per_bank_bandwidth', 0)),
                        'LayerTotalCycles': int(comp_items[0]) if comp_items else 0,
                        'LayerStallCycles': int(comp_items[2]) if comp_items and len(comp_items) > 2 else 0,
                        'LayerBankConflictStall': int(bank_items.get('stall_cycles_due_to_bank_conflict', 0)),
                        'RuntimePrefetchBankRequests': int(prefetch_requests),
                        'RuntimePrefetchBankInterferenceStall': int(prefetch_interference_stall),
                        'RuntimePrefetchBankInterferenceCycles': int(prefetch_interference_cycles),
                        'PrefetchAwareCombinedBankCycles': int(combined_cycles),
                        'PrefetchAwareCombinedBankStall': int(combined_stall),
                        'NormalBankStall': int(normal_stall),
                    })

        self.ep_moe_bank_allocation_rows = rows
        return rows

    def _collect_expert_bank_allocation_stats(self, expert, is_detailed):
        """Collect static/dynamic bank allocation metadata for EP-MoE reports."""
        if not is_detailed:
            return {
                'bank_allocation_mode': 'blackbox_no_onchip_banks',
                'bank_allocation_ratios': '',
                'dynamic_fallback_count': 0,
                'dynamic_bank_overhead_model': str(self.conf.get_dynamic_bank_overhead()),
                'effective_per_bank_bandwidth': 0,
            }

        ratios = []
        fallback_count = 0
        effective_bw_values = []
        mode = 'dynamic' if self.conf.get_enable_dynamic() else 'static'

        for layer in expert['layers']:
            layer_id = int(layer['layer_id'])
            layer_obj = self.single_layer_sim_object_list[layer_id]
            if not hasattr(layer_obj, 'get_bank_report_items'):
                continue
            bank_items = layer_obj.get_bank_report_items()
            ratios.append(str(bank_items.get('allocation_ratio', '')))
            if bool(bank_items.get('EnableDynamic', self.conf.get_enable_dynamic())) and bool(bank_items.get('dynamic_fallback_to_static', False)):
                fallback_count += 1
            if 'effective_per_bank_bandwidth' in bank_items:
                effective_bw_values.append(float(bank_items.get('effective_per_bank_bandwidth', 0)))

        effective_bw = 0.0
        if effective_bw_values:
            effective_bw = sum(effective_bw_values) / float(len(effective_bw_values))

        return {
            'bank_allocation_mode': mode,
            'bank_allocation_ratios': '|'.join(ratios),
            'dynamic_fallback_count': int(fallback_count),
            'dynamic_bank_overhead_model': str(self.conf.get_dynamic_bank_overhead()),
            'effective_per_bank_bandwidth': float(effective_bw),
        }

    def _compute_ep_moe_report_rows(self):
        """Build EP-MoE expert/group timing rows from existing layer results.

        First implementation:
        - detailed GPU experts use per-layer SCALE-Sim results
        - black-box GPU experts use analytical MAC/PE estimate
        - MoE group time is max expert finish time
        """
        detailed_gpu_id = int(self.conf.get_detailed_gpu_id())
        self.ep_moe_report_rows = []

        for group in self.ep_moe_groups:
            pending_rows = []
            timeline_row = next(
                (row for row in self.ep_moe_timeline_rows
                 if row.get('TimelineType') == 'moe_group'
                 and int(row.get('MoEGroupID', -1)) == int(group['group_id'])),
                None,
            )
            group_start_time = int(timeline_row['StartCycle']) if timeline_row is not None else None
            group_finish_time = int(timeline_row['FinishCycle']) if timeline_row is not None else 0

            for expert in group['experts']:
                gpu_id = int(expert['gpu_id'])
                is_active = bool(expert.get('is_active', True))
                is_detailed = gpu_id == detailed_gpu_id
                expert_cycles = 0
                layer_ids = []
                layer_names = []
                analytical_stats = self._estimate_blackbox_expert_stats(expert)
                detailed_prefetch_bank_stats = self._estimate_detailed_prefetch_bank_stats(expert) if (is_active and is_detailed) else {
                    'prefetch_bank_model_cycles': 0,
                    'prefetch_bank_conflict_cycles': 0,
                    'prefetch_bank_requests': 0,
                    'combined_bank_model_cycles': 0,
                    'combined_bank_conflict_cycles': 0,
                    'prefetch_bank_interference_stall': 0,
                    'prefetch_bank_interference_cycles': 0,
                }
                bank_allocation_stats = self._collect_expert_bank_allocation_stats(expert, is_detailed)

                for layer in expert['layers']:
                    layer_id = int(layer['layer_id'])
                    layer_ids.append(str(layer_id))
                    layer_names.append(str(layer['layer_name']))

                    if is_detailed:
                        comp_items = self.single_layer_sim_object_list[layer_id].get_compute_report_items()
                        expert_cycles += int(comp_items[0])

                if not is_active:
                    expert_cycles = 0
                elif not is_detailed:
                    expert_cycles = int(analytical_stats['blackbox_cycles'])

                runtime_state = self.ep_moe_runtime_states.get(int(group['group_id']), {}).get(int(expert['expert_id']))
                runtime_base_expert_cycles = int(expert_cycles)
                runtime_prefetch_bank_interference_stall = 0
                runtime_prefetch_bank_interference_cycles = 0
                runtime_prefetch_bank_requests = 0
                runtime_blackbox_background_pressure_bytes = 0
                runtime_blackbox_background_pressure_cycles = 0
                runtime_blackbox_background_pressure_stall = 0
                if runtime_state is not None:
                    expert_start = int(runtime_state.get('expert_start_time', 0))
                    expert_finish = int(runtime_state.get('expert_finish_time', expert_start + expert_cycles))
                    expert_cycles = int(expert_finish - expert_start)
                    runtime_base_expert_cycles = int(runtime_state.get('base_expert_cycles', runtime_base_expert_cycles))
                    runtime_prefetch_bank_interference_stall = int(runtime_state.get('runtime_prefetch_bank_interference_stall', 0))
                    runtime_prefetch_bank_interference_cycles = int(runtime_state.get('runtime_prefetch_bank_interference_cycles', 0))
                    runtime_prefetch_bank_requests = int(runtime_state.get('runtime_prefetch_bank_requests', 0))
                    runtime_blackbox_background_pressure_bytes = int(runtime_state.get('runtime_blackbox_background_pressure_bytes', 0))
                    runtime_blackbox_background_pressure_cycles = int(runtime_state.get('runtime_blackbox_background_pressure_cycles', 0))
                    runtime_blackbox_background_pressure_stall = int(runtime_state.get('runtime_blackbox_background_pressure_stall', 0))
                else:
                    expert_start = 0
                    expert_finish = int(expert_start + expert_cycles)
                if group_start_time is None:
                    group_start_time = int(expert_start)
                    group_finish_time = max(group_finish_time, int(expert_finish))
                elif timeline_row is None:
                    group_start_time = min(group_start_time, int(expert_start))
                    group_finish_time = max(group_finish_time, int(expert_finish))

                row_stats = analytical_stats
                if not is_active:
                    row_stats = {
                        'tokens': 0,
                        'hidden_dim': 0,
                        'mac_ops': 0,
                        'weight_bytes': 0,
                        'compute_cycles': 0,
                        'weight_bw_cycles': 0,
                        'communication_bytes': 0,
                        'communication_cycles': 0,
                        'dispatch_bytes': 0,
                        'combine_bytes': 0,
                        'dispatch_cycles': 0,
                        'combine_cycles': 0,
                        'communication_overlap_cycles': 0,
                        'exposed_communication_cycles': 0,
                        'communication_overlap_mode': str(self.conf.get_communication_overlap_mode()),
                    }

                pending_rows.append({
                    'MoEGroupID': int(group['group_id']),
                    'ExpertID': int(expert['expert_id']),
                    'GPUId': gpu_id,
                    'LocalExpertID': int(expert['local_expert_id']),
                    'IsActiveExpert': bool(is_active),
                    'RoutingRank': int(expert.get('routing_rank', -1)),
                    'RoutingPolicy': str(expert.get('routing_policy', 'all')),
                    'IsDetailedGPU': bool(is_detailed),
                    'LayerIDs': '|'.join(layer_ids),
                    'LayerNames': '|'.join(layer_names),
                    'ExpertStartCycle': int(expert_start),
                    'ExpertFinishCycle': int(expert_finish),
                    'ExpertCycles': int(expert_cycles),
                    'EstimationMode': 'inactive_routed_out' if not is_active else ('detailed_scalesim' if is_detailed else 'analytical_blackbox'),
                    'TokensPerExpert': int(row_stats['tokens']),
                    'HiddenDim': int(row_stats['hidden_dim']),
                    'ExpertMACs': int(row_stats['mac_ops']),
                    'ExpertWeightBytes': int(row_stats['weight_bytes']),
                    'BlackBoxComputeCycles': int(row_stats['compute_cycles']),
                    'BlackBoxWeightBWCycles': int(row_stats['weight_bw_cycles']),
                    'BlackBoxCommunicationBytes': 0 if is_detailed else int(row_stats['communication_bytes']),
                    'BlackBoxCommunicationCycles': 0 if is_detailed else int(row_stats['communication_cycles']),
                    'TokenDispatchBytes': 0 if is_detailed else int(row_stats['dispatch_bytes']),
                    'OutputCombineBytes': 0 if is_detailed else int(row_stats['combine_bytes']),
                    'TokenDispatchCycles': 0 if is_detailed else int(row_stats['dispatch_cycles']),
                    'OutputCombineCycles': 0 if is_detailed else int(row_stats['combine_cycles']),
                    'CommunicationOverlapCycles': int(runtime_state.get('runtime_communication_overlap_cycles', 0)) if runtime_state is not None else 0,
                    'ExposedCommunicationCycles': 0 if is_detailed else int(row_stats['exposed_communication_cycles']),
                    'CommunicationOverlapMode': str(row_stats['communication_overlap_mode']),
                    'WeightChunkCount': int(runtime_state.get('chunk_count', 0)) if runtime_state is not None else 0,
                    'InitialWeightStall': int(runtime_state.get('runtime_initial_weight_stall', 0)) if runtime_state is not None else 0,
                    'WeightLoadingStall': int(runtime_state.get('runtime_weight_loading_stall', 0)) if runtime_state is not None else 0,
                    'PrefetchHit': int(runtime_state.get('runtime_prefetch_hit', 0)) if runtime_state is not None else 0,
                    'PrefetchMiss': int(runtime_state.get('runtime_prefetch_miss', 0)) if runtime_state is not None else 0,
                    'PrefetchHitRate': float(runtime_state.get('runtime_prefetch_hit_rate', 0.0)) if runtime_state is not None else 0.0,
                    'PrefetchMissStall': int(runtime_state.get('runtime_prefetch_miss_stall', 0)) if runtime_state is not None else 0,
                    'PrefetchBandwidthOverhead': int(runtime_state.get('runtime_prefetch_bandwidth_overhead', 0)) if runtime_state is not None else 0,
                    'PrefetchInterferenceStall': int(runtime_state.get('runtime_prefetch_bank_interference_stall', 0)) if runtime_state is not None else 0,
                    'UsefulPrefetchTraffic': int(runtime_state.get('runtime_useful_prefetch_traffic', 0)) if runtime_state is not None else 0,
                    'UselessPrefetchTraffic': int(runtime_state.get('runtime_useless_prefetch_traffic', 0)) if runtime_state is not None else 0,
                    'ComputeWithPrefetchCycles': int(runtime_state.get('runtime_expert_cycles', 0)) if runtime_state is not None else 0,
                    'PrefetchBankModelCycles': int(detailed_prefetch_bank_stats['prefetch_bank_model_cycles']),
                    'PrefetchBankConflictCycles': int(detailed_prefetch_bank_stats['prefetch_bank_conflict_cycles']),
                    'PrefetchBankRequests': int(detailed_prefetch_bank_stats['prefetch_bank_requests']),
                    'CombinedBankModelCycles': int(detailed_prefetch_bank_stats['combined_bank_model_cycles']),
                    'CombinedBankConflictCycles': int(detailed_prefetch_bank_stats['combined_bank_conflict_cycles']),
                    'PrefetchBankInterferenceStall': int(detailed_prefetch_bank_stats['prefetch_bank_interference_stall']),
                    'PrefetchBankInterferenceCycles': int(detailed_prefetch_bank_stats['prefetch_bank_interference_cycles']),
                    'BankAllocationMode': str(bank_allocation_stats['bank_allocation_mode']),
                    'BankAllocationRatios': str(bank_allocation_stats['bank_allocation_ratios']),
                    'DynamicFallbackCount': int(bank_allocation_stats['dynamic_fallback_count']),
                    'DynamicBankOverheadModel': str(bank_allocation_stats['dynamic_bank_overhead_model']),
                    'EffectivePerBankBandwidth': float(bank_allocation_stats['effective_per_bank_bandwidth']),
                    'BaseExpertCycles': int(runtime_base_expert_cycles),
                    'RuntimePrefetchBankInterferenceStall': int(runtime_prefetch_bank_interference_stall),
                    'RuntimePrefetchBankInterferenceCycles': int(runtime_prefetch_bank_interference_cycles),
                    'RuntimePrefetchBankRequests': int(runtime_prefetch_bank_requests),
                    'RuntimeBlackBoxBackgroundPressureBytes': int(runtime_blackbox_background_pressure_bytes),
                    'RuntimeBlackBoxBackgroundPressureCycles': int(runtime_blackbox_background_pressure_cycles),
                    'RuntimeBlackBoxBackgroundPressureStall': int(runtime_blackbox_background_pressure_stall),
                })

            for row in pending_rows:
                row['MoEGroupTime'] = int(group_finish_time - (group_start_time or 0))
                self.ep_moe_report_rows.append(row)

    def _build_ep_moe_summary_rows(self):
        summary_rows = []
        group_ids = sorted(set(int(row.get('MoEGroupID', 0)) for row in self.ep_moe_report_rows))

        for group_id in group_ids:
            rows = [row for row in self.ep_moe_report_rows if int(row.get('MoEGroupID', 0)) == group_id]
            if not rows:
                continue

            num_experts = len(rows)
            detailed_rows = [row for row in rows if bool(row.get('IsDetailedGPU', False))]
            blackbox_rows = [row for row in rows if not bool(row.get('IsDetailedGPU', False))]
            active_rows = [row for row in rows if bool(row.get('IsActiveExpert', True))]
            inactive_rows = [row for row in rows if not bool(row.get('IsActiveExpert', True))]
            runtime_states = self.ep_moe_runtime_states.get(group_id, {})
            active_runtime_states = [
                state for state in runtime_states.values()
                if bool(state.get('is_active', True))
            ]
            total_hit = sum(int(state.get('runtime_prefetch_hit', 0)) for state in active_runtime_states)
            total_miss = sum(int(state.get('runtime_prefetch_miss', 0)) for state in active_runtime_states)
            total_prefetch_lookups = total_hit + total_miss
            avg_hit_rate = 0.0
            if total_prefetch_lookups > 0:
                avg_hit_rate = float(total_hit) / float(total_prefetch_lookups)
            active_cycles = [int(row.get('ExpertCycles', 0)) for row in active_rows]
            active_tokens = [int(row.get('TokensPerExpert', 0)) for row in active_rows]
            active_waiting = [
                int(state.get('expert_waiting_time', 0))
                for state in runtime_states.values()
                if bool(state.get('is_active', True))
            ]
            group_time = int(max(int(row.get('MoEGroupTime', 0)) for row in rows))
            gpu_service = {}
            for state in active_runtime_states:
                gpu_id = int(state.get('gpu_id', 0))
                gpu_service[gpu_id] = gpu_service.get(gpu_id, 0) + int(
                    state.get('runtime_compute_service_cycles', 0)
                )
            gpu_load_values = list(gpu_service.values())
            engine_count = max(1, int(self.conf.get_compute_engines_per_gpu()))
            gpu_utilizations = [
                min(1.0, float(load) / float(max(1, group_time * engine_count)))
                for load in gpu_load_values
            ]

            detailed_layer_ids = set()
            for row in detailed_rows:
                if bool(row.get('IsActiveExpert', True)):
                    detailed_layer_ids.update(
                        int(layer_id) for layer_id in str(row.get('LayerIDs', '')).split('|')
                        if layer_id != ''
                    )
            bank_utilizations = []
            detailed_dram_elements = 0
            for layer_id in sorted(detailed_layer_ids):
                layer_obj = self.single_layer_sim_object_list[layer_id]
                bank_items = layer_obj.get_bank_report_items()
                bank_utilizations.extend([
                    float(bank_items.get('ifmap_capacity_utilization', 0)),
                    float(bank_items.get('filter_capacity_utilization', 0)),
                    float(bank_items.get('ofmap_capacity_utilization', 0)),
                ])
                detail_items = layer_obj.get_detail_report_items()
                if len(detail_items) >= 18:
                    detailed_dram_elements += sum(int(detail_items[index]) for index in (11, 14, 17))
            precision_bytes = max(1, int(self.conf.get_precision_bytes()))
            blackbox_dram_bytes = sum(
                int(row.get('ExpertWeightBytes', 0))
                for row in blackbox_rows if bool(row.get('IsActiveExpert', True))
            )

            summary_rows.append({
                'MoEGroupID': int(group_id),
                'NumExperts': int(num_experts),
                'NumActiveExperts': int(len(active_rows)),
                'NumInactiveExperts': int(len(inactive_rows)),
                'NumDetailedExperts': int(len(detailed_rows)),
                'NumBlackBoxExperts': int(len(blackbox_rows)),
                'MoEGroupTime': group_time,
                'TotalExpertCycles': int(sum(int(row.get('ExpertCycles', 0)) for row in rows)),
                'MaxExpertCycles': int(max(int(row.get('ExpertCycles', 0)) for row in rows)),
                'ExpertCycleImbalance': int(max(active_cycles) - min(active_cycles)) if active_cycles else 0,
                'ExpertTokenImbalance': int(max(active_tokens) - min(active_tokens)) if active_tokens else 0,
                'GPULoadImbalanceCycles': int(max(gpu_load_values) - min(gpu_load_values)) if gpu_load_values else 0,
                'AverageGPUUtilization': float(sum(gpu_utilizations) / len(gpu_utilizations)) if gpu_utilizations else 0.0,
                'MinimumGPUUtilization': float(min(gpu_utilizations)) if gpu_utilizations else 0.0,
                'AverageDetailedBankCapacityUtilization': float(sum(bank_utilizations) / len(bank_utilizations)) if bank_utilizations else 0.0,
                'MaximumDetailedBankCapacityUtilization': float(max(bank_utilizations)) if bank_utilizations else 0.0,
                'DetailedDRAMTrafficBytes': int(detailed_dram_elements * precision_bytes),
                'EstimatedBlackBoxDRAMTrafficBytes': int(blackbox_dram_bytes),
                'TotalInterconnectQueueWait': int(sum(
                    int(state.get('runtime_dispatch_queue_wait', 0))
                    + int(state.get('runtime_combine_queue_wait', 0))
                    for state in active_runtime_states
                )),
                'TotalExpertWaitingCycles': int(sum(active_waiting)),
                'MaxExpertWaitingCycles': int(max(active_waiting)) if active_waiting else 0,
                'TotalPrefetchHit': int(total_hit),
                'TotalPrefetchMiss': int(total_miss),
                'AvgPrefetchHitRate': float(avg_hit_rate),
                'TotalPrefetchMissStall': int(sum(int(state.get('runtime_prefetch_miss_stall', 0)) for state in active_runtime_states)),
                'TotalWeightLoadingStall': int(sum(int(state.get('runtime_weight_loading_stall', 0)) for state in active_runtime_states)),
                'TotalPrefetchBankInterferenceStall': int(sum(int(row.get('PrefetchBankInterferenceStall', 0)) for row in rows)),
                'TotalUsefulPrefetchTraffic': int(sum(int(state.get('runtime_useful_prefetch_traffic', 0)) for state in active_runtime_states)),
                'TotalUselessPrefetchTraffic': int(sum(int(state.get('runtime_useless_prefetch_traffic', 0)) for state in active_runtime_states)),
                'TotalPrefetchBandwidthOverhead': int(sum(int(state.get('runtime_prefetch_bandwidth_overhead', 0)) for state in active_runtime_states)),
                'TotalCommunicationBytes': int(sum(int(row.get('BlackBoxCommunicationBytes', 0)) for row in rows)),
                'MaxCommunicationCycles': int(max(int(row.get('BlackBoxCommunicationCycles', 0)) for row in rows)),
                'TotalTokenDispatchBytes': int(sum(int(row.get('TokenDispatchBytes', 0)) for row in rows)),
                'TotalOutputCombineBytes': int(sum(int(row.get('OutputCombineBytes', 0)) for row in rows)),
                'MaxTokenDispatchCycles': int(max(int(row.get('TokenDispatchCycles', 0)) for row in rows)),
                'MaxOutputCombineCycles': int(max(int(row.get('OutputCombineCycles', 0)) for row in rows)),
                'TotalCommunicationOverlapCycles': int(sum(int(state.get('runtime_communication_overlap_cycles', 0)) for state in active_runtime_states)),
                'MaxExposedCommunicationCycles': int(max(int(row.get('ExposedCommunicationCycles', 0)) for row in rows)),
                'BankAllocationModes': '|'.join(sorted(set(str(row.get('BankAllocationMode', '')) for row in rows))),
                'TotalDynamicFallbackCount': int(sum(int(row.get('DynamicFallbackCount', 0)) for row in rows)),
                'TotalRuntimePrefetchBankInterferenceStall': int(sum(int(row.get('RuntimePrefetchBankInterferenceStall', 0)) for row in rows)),
                'GroupRuntimeBlackBoxBackgroundPressureBytes': int(max(int(row.get('RuntimeBlackBoxBackgroundPressureBytes', 0)) for row in rows)),
                'MaxRuntimeBlackBoxBackgroundPressureCycles': int(max(int(row.get('RuntimeBlackBoxBackgroundPressureCycles', 0)) for row in rows)),
                'GroupRuntimeBlackBoxBackgroundPressureStall': int(max(int(row.get('RuntimeBlackBoxBackgroundPressureStall', 0)) for row in rows)),
                'DynamicBankOverheadModel': str(self.conf.get_dynamic_bank_overhead()),
            })

        return summary_rows

    def _write_ep_moe_config_report(self):
        config_report_name = self.top_path + '/EP_MOE_CONFIG.csv'
        items = [
            ('EnableEPMoE', self.conf.get_enable_ep_moe()),
            ('EnableParallelMoE', self.conf.get_enable_parallel_moe()),
            ('NumGPUs', self.conf.get_num_gpus()),
            ('DetailedGPUId', self.conf.get_detailed_gpu_id()),
            ('BlackBoxGPUIds', '|'.join([str(x) for x in self.conf.get_blackbox_gpu_ids()])),
            ('ExpertsPerGPU', self.conf.get_experts_per_gpu()),
            ('ComputeEnginesPerGPU', self.conf.get_compute_engines_per_gpu()),
            ('NumExperts', self.conf.get_num_experts()),
            ('TopK', self.conf.get_top_k()),
            ('MoERoutingMode', self.conf.get_moe_routing_mode()),
            ('MoETokens', self.conf.get_moe_tokens()),
            ('RoutingFile', self.conf.get_routing_file()),
            ('RoutingSeed', self.conf.get_routing_seed()),
            ('RoutingSkewFactor', self.conf.get_routing_skew_factor()),
            ('MoEActiveExpertMode', self.conf.get_moe_active_expert_mode()),
            ('ActiveExpertIds', '|'.join([str(x) for x in self.conf.get_active_expert_ids()])),
            ('EnableChunkPrefetch', self.conf.get_enable_chunk_prefetch()),
            ('InitialChunk', self.conf.get_initial_chunk()),
            ('ChunkPrefetchWindow', self.conf.get_chunk_prefetch_window()),
            ('BlackBoxWorkloadMode', self.conf.get_blackbox_workload_mode()),
            ('BlackBoxBandwidthBytesPerCycle', self.conf.get_blackbox_bandwidth_bytes_per_cycle()),
            ('EnableBlackBoxBackgroundPressure', self.conf.get_enable_blackbox_background_pressure()),
            ('GlobalMemoryBandwidthBytesPerCycle', self.conf.get_global_memory_bandwidth_bytes_per_cycle()),
            ('DynamicBankOverhead', self.conf.get_dynamic_bank_overhead()),
            ('CommunicationModel', self.conf.get_communication_model()),
            ('PrecisionBytes', self.conf.get_precision_bytes()),
            ('CommunicationLatencyCycles', self.conf.get_communication_latency_cycles()),
            ('CommunicationBandwidthBytesPerCycle', self.conf.get_communication_bandwidth_bytes_per_cycle()),
            ('CommunicationOverlapMode', self.conf.get_communication_overlap_mode()),
            ('AllowCommPrefetchOverlap', self.conf.get_allow_comm_prefetch_overlap()),
            ('EnableBankModel', self.conf.get_enable_bank_model()),
            ('EnableDynamic', self.conf.get_enable_dynamic()),
            ('BankConflictPenalty', self.conf.get_bank_conflict_penalty()),
            ('EPBankAllocationReport', 'EP_MOE_BANK_ALLOCATION.csv'),
        ]

        with open(config_report_name, 'w', encoding='utf-8') as config_report:
            config_report.write('Key, Value,\n')
            for key, value in items:
                config_report.write(str(key) + ', ' + str(value) + ',\n')

    def _write_ep_moe_runtime_state_report(self):
        runtime_report_name = self.top_path + '/EP_MOE_RUNTIME_STATE.csv'
        with open(runtime_report_name, 'w', encoding='utf-8') as runtime_report:
            runtime_report.write(
                'MoEGroupID, ExpertID, GPUId, LocalExpertID, IsActiveExpert, '
                'RoutingRank, RoutingPolicy, ExpertState, '
                'TokensPerExpert, CurrentTile, CurrentWeightChunk, '
                'LoadedWeightChunks, PrefetchedWeightChunks, ConsumedWeightChunks, '
                'ChunkCount, InitialChunkCount, PrefetchWindow, '
                'ExpertStartTime, ExpertFinishTime, ExpertWaitingTime, '
                'BaseExpertCycles, RuntimePrefetchBankInterferenceStall, '
                'RuntimePrefetchBankInterferenceCycles, RuntimePrefetchBankRequests, '
                'RuntimeBlackBoxBackgroundPressureBytes, '
                'RuntimeBlackBoxBackgroundPressureCycles, '
                'RuntimeBlackBoxBackgroundPressureStall, '
                'RuntimeExpertCycles, RuntimePrefetchHit, RuntimePrefetchMiss, RuntimePrefetchHitRate, '
                'RuntimePrefetchMissStall, RuntimeWeightLoadingStall, RuntimePrefetchBandwidthOverhead, '
                'RuntimeUsefulPrefetchTraffic, RuntimeUselessPrefetchTraffic, '
                'RuntimeInitialWeightStall, RuntimeDispatchCycles, RuntimeCombineCycles, '
                'RuntimeCommunicationOverlapCycles, RuntimeComputeServiceCycles, '
                'RuntimeDispatchQueueWait, RuntimeCombineQueueWait,\n'
            )
            for group_id in sorted(self.ep_moe_runtime_states.keys()):
                expert_states = self.ep_moe_runtime_states[group_id]
                for expert_id in sorted(expert_states.keys()):
                    state = expert_states[expert_id]
                    runtime_report.write(', '.join([
                        str(state.get('moe_group_id', group_id)),
                        str(state.get('expert_id', expert_id)),
                        str(state.get('gpu_id', 0)),
                        str(state.get('local_expert_id', 0)),
                        str(state.get('is_active', True)),
                        str(state.get('routing_rank', -1)),
                        str(state.get('routing_policy', 'all')),
                        str(state.get('expert_state', 'unknown')),
                        str(state.get('tokens_per_expert', 0)),
                        str(state.get('current_tile', 0)),
                        str(state.get('current_weight_chunk', 0)),
                        '|'.join([str(x) for x in sorted(state.get('loaded_weight_chunks', set()))]),
                        '|'.join([str(x) for x in sorted(state.get('prefetched_weight_chunks', set()))]),
                        '|'.join([str(x) for x in sorted(state.get('consumed_weight_chunks', set()))]),
                        str(state.get('chunk_count', 0)),
                        str(state.get('initial_chunk_count', 0)),
                        str(state.get('prefetch_window', 0)),
                        str(state.get('expert_start_time', 0)),
                        str(state.get('expert_finish_time', 0)),
                        str(state.get('expert_waiting_time', 0)),
                        str(state.get('base_expert_cycles', 0)),
                        str(state.get('runtime_prefetch_bank_interference_stall', 0)),
                        str(state.get('runtime_prefetch_bank_interference_cycles', 0)),
                        str(state.get('runtime_prefetch_bank_requests', 0)),
                        str(state.get('runtime_blackbox_background_pressure_bytes', 0)),
                        str(state.get('runtime_blackbox_background_pressure_cycles', 0)),
                        str(state.get('runtime_blackbox_background_pressure_stall', 0)),
                        str(state.get('runtime_expert_cycles', 0)),
                        str(state.get('runtime_prefetch_hit', 0)),
                        str(state.get('runtime_prefetch_miss', 0)),
                        str(state.get('runtime_prefetch_hit_rate', 0.0)),
                        str(state.get('runtime_prefetch_miss_stall', 0)),
                        str(state.get('runtime_weight_loading_stall', 0)),
                        str(state.get('runtime_prefetch_bandwidth_overhead', 0)),
                        str(state.get('runtime_useful_prefetch_traffic', 0)),
                        str(state.get('runtime_useless_prefetch_traffic', 0)),
                        str(state.get('runtime_initial_weight_stall', 0)),
                        str(state.get('runtime_dispatch_cycles', 0)),
                        str(state.get('runtime_combine_cycles', 0)),
                        str(state.get('runtime_communication_overlap_cycles', 0)),
                        str(state.get('runtime_compute_service_cycles', 0)),
                        str(state.get('runtime_dispatch_queue_wait', 0)),
                        str(state.get('runtime_combine_queue_wait', 0)),
                    ]) + ',\n')

    def _write_ep_moe_timeline_report(self):
        timeline_report_name = self.top_path + '/EP_MOE_TIMELINE.csv'
        with open(timeline_report_name, 'w', encoding='utf-8') as timeline_report:
            timeline_report.write(
                'TimelineType, MoEGroupID, MoELayerID, LayerID, LayerName, StartCycle, '
                'FinishCycle, DurationCycles, NumExperts, NumActiveExperts, ParallelExecution,\n'
            )
            for row in self.ep_moe_timeline_rows:
                timeline_report.write(', '.join([
                    str(row.get('TimelineType', '')),
                    str(row.get('MoEGroupID', '')),
                    str(row.get('MoELayerID', '')),
                    str(row.get('LayerID', '')),
                    str(row.get('LayerName', '')),
                    str(row.get('StartCycle', 0)),
                    str(row.get('FinishCycle', 0)),
                    str(row.get('DurationCycles', 0)),
                    str(row.get('NumExperts', 0)),
                    str(row.get('NumActiveExperts', 0)),
                    str(row.get('ParallelExecution', False)),
                ]) + ',\n')

    def _write_ep_moe_routing_report(self):
        routing_report_name = self.top_path + '/EP_MOE_ROUTING.csv'
        with open(routing_report_name, 'w', encoding='utf-8', newline='') as routing_report:
            writer = csv.DictWriter(
                routing_report,
                fieldnames=['MoEGroupID', 'MoELayerID', 'TokenID', 'ExpertIDs', 'TopK', 'RoutingMode'],
            )
            writer.writeheader()
            writer.writerows(self.ep_moe_routing_rows)

    def _write_ep_moe_event_report(self):
        event_report_name = self.top_path + '/EP_MOE_EVENTS.csv'
        fieldnames = [
            'Sequence', 'Cycle', 'Event', 'MoEGroupID', 'MoELayerID',
            'ExpertID', 'GPUId', 'EngineID', 'TileID', 'ExpertState', 'Detail',
        ]
        with open(event_report_name, 'w', encoding='utf-8', newline='') as event_report:
            writer = csv.DictWriter(event_report, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.ep_moe_event_rows)

    def _write_ep_moe_chunk_report(self):
        chunk_report_name = self.top_path + '/EP_MOE_CHUNKS.csv'
        fieldnames = [
            'MoEGroupID', 'ExpertID', 'GPUId', 'ChunkID', 'LayerID', 'LayerName',
            'FFNPart', 'TileIDInLayer', 'ChunkSource', 'TraceStartCycle',
            'TraceEndCycle', 'WeightTraceEndCycle', 'ComputeCycles',
            'WeightElements', 'WeightBytes', 'WeightLoadCycles',
            'RawWeightAddressMin', 'RawWeightAddressMax',
            'LogicalWeightAddressMin', 'LogicalWeightAddressMax',
            'IfmapRequests', 'FilterRequests', 'OfmapRequests',
        ]
        with open(chunk_report_name, 'w', encoding='utf-8', newline='') as chunk_report:
            writer = csv.DictWriter(chunk_report, fieldnames=fieldnames)
            writer.writeheader()
            for chunk in self.ep_moe_chunk_rows:
                writer.writerow({
                    'MoEGroupID': chunk.get('moe_group_id', ''),
                    'ExpertID': chunk.get('expert_id', ''),
                    'GPUId': chunk.get('gpu_id', ''),
                    'ChunkID': chunk.get('chunk_id', ''),
                    'LayerID': chunk.get('layer_id', ''),
                    'LayerName': chunk.get('layer_name', ''),
                    'FFNPart': chunk.get('ffn_part', ''),
                    'TileIDInLayer': chunk.get('tile_id_in_layer', ''),
                    'ChunkSource': chunk.get('chunk_source', 'analytical'),
                    'TraceStartCycle': chunk.get('trace_start_cycle', ''),
                    'TraceEndCycle': chunk.get('trace_end_cycle', ''),
                    'WeightTraceEndCycle': chunk.get('weight_trace_end_cycle', ''),
                    'ComputeCycles': chunk.get('compute_cycles', 0),
                    'WeightElements': chunk.get('weight_elements', ''),
                    'WeightBytes': chunk.get('weight_bytes', 0),
                    'WeightLoadCycles': chunk.get('weight_load_cycles', 0),
                    'RawWeightAddressMin': chunk.get('raw_weight_address_min', ''),
                    'RawWeightAddressMax': chunk.get('raw_weight_address_max', ''),
                    'LogicalWeightAddressMin': chunk.get('logical_weight_address_min', ''),
                    'LogicalWeightAddressMax': chunk.get('logical_weight_address_max', ''),
                    'IfmapRequests': chunk.get('ifmap_requests', ''),
                    'FilterRequests': chunk.get('filter_requests', ''),
                    'OfmapRequests': chunk.get('ofmap_requests', ''),
                })

    def _write_ep_moe_run_manifest(self):
        manifest_name = self.top_path + '/EP_MOE_RUN_MANIFEST.csv'
        fieldnames = ['InputKind', 'Path', 'SHA256', 'SizeBytes']
        model_root = os.path.dirname(os.path.abspath(__file__))
        sources = {
            'config': self.input_sources.get('config', ''),
            'topology': self.input_sources.get('topology', ''),
            'layout': self.input_sources.get('layout', ''),
            'model_simulator': os.path.join(model_root, 'simulator.py'),
            'model_config': os.path.join(model_root, 'scale_config.py'),
            'model_banked_memory': os.path.join(model_root, 'memory', 'banked_memory_system.py'),
        }
        with open(manifest_name, 'w', encoding='utf-8', newline='') as manifest:
            writer = csv.DictWriter(manifest, fieldnames=fieldnames)
            writer.writeheader()
            for kind, source in sources.items():
                if source == '' or not os.path.isfile(source):
                    writer.writerow({'InputKind': kind, 'Path': source, 'SHA256': '', 'SizeBytes': 0})
                    continue
                digest = hashlib.sha256()
                with open(source, 'rb') as input_file:
                    for block in iter(lambda: input_file.read(1024 * 1024), b''):
                        digest.update(block)
                writer.writerow({
                    'InputKind': kind,
                    'Path': os.path.abspath(source),
                    'SHA256': digest.hexdigest(),
                    'SizeBytes': os.path.getsize(source),
                })

    def _write_ep_moe_layer_execution_report(self):
        report_name = self.top_path + '/EP_MOE_LAYER_EXECUTION.csv'
        with open(report_name, 'w', encoding='utf-8', newline='') as report:
            writer = csv.DictWriter(
                report,
                fieldnames=['LayerID', 'LayerName', 'ExecutionMode', 'DetailedSimulationExecuted'],
            )
            writer.writeheader()
            for layer_id, layer_name in enumerate(self.topo.get_layer_names()):
                is_blackbox = layer_id in self.ep_moe_blackbox_layer_ids
                writer.writerow({
                    'LayerID': layer_id,
                    'LayerName': layer_name,
                    'ExecutionMode': 'analytical_blackbox' if is_blackbox else 'detailed_scalesim',
                    'DetailedSimulationExecuted': not is_blackbox,
                })

    def _write_ep_moe_bank_allocation_report(self):
        bank_alloc_report_name = self.top_path + '/EP_MOE_BANK_ALLOCATION.csv'
        rows = self._build_ep_moe_bank_allocation_rows()
        with open(bank_alloc_report_name, 'w', encoding='utf-8') as bank_alloc_report:
            bank_alloc_report.write(
                'MoEGroupID, ExpertID, GPUId, LocalExpertID, IsActiveExpert, '
                'LayerID, LayerName, BankAllocationMode, EnableDynamic, '
                'DynamicFallbackToStatic, AllocationRatio, '
                'IfmapBankNum, FilterBankNum, OfmapBankNum, '
                'StaticIfmapBankNum, StaticFilterBankNum, StaticOfmapBankNum, '
                'BitCapacityTargetIfmapBankNum, BitCapacityTargetFilterBankNum, '
                'BitCapacityTargetOfmapBankNum, EffectivePerBankBandwidth, '
                'LayerTotalCycles, LayerStallCycles, LayerBankConflictStall, '
                'RuntimePrefetchBankRequests, RuntimePrefetchBankInterferenceStall, '
                'RuntimePrefetchBankInterferenceCycles, PrefetchAwareCombinedBankCycles, '
                'PrefetchAwareCombinedBankStall, NormalBankStall,\n'
            )
            for row in rows:
                bank_alloc_report.write(', '.join([
                    str(row.get('MoEGroupID', 0)),
                    str(row.get('ExpertID', 0)),
                    str(row.get('GPUId', 0)),
                    str(row.get('LocalExpertID', 0)),
                    str(row.get('IsActiveExpert', True)),
                    str(row.get('LayerID', 0)),
                    str(row.get('LayerName', '')),
                    str(row.get('BankAllocationMode', '')),
                    str(row.get('EnableDynamic', False)),
                    str(row.get('DynamicFallbackToStatic', False)),
                    str(row.get('AllocationRatio', '')),
                    str(row.get('IfmapBankNum', 0)),
                    str(row.get('FilterBankNum', 0)),
                    str(row.get('OfmapBankNum', 0)),
                    str(row.get('StaticIfmapBankNum', 0)),
                    str(row.get('StaticFilterBankNum', 0)),
                    str(row.get('StaticOfmapBankNum', 0)),
                    str(row.get('BitCapacityTargetIfmapBankNum', -1)),
                    str(row.get('BitCapacityTargetFilterBankNum', -1)),
                    str(row.get('BitCapacityTargetOfmapBankNum', -1)),
                    str(row.get('EffectivePerBankBandwidth', 0)),
                    str(row.get('LayerTotalCycles', 0)),
                    str(row.get('LayerStallCycles', 0)),
                    str(row.get('LayerBankConflictStall', 0)),
                    str(row.get('RuntimePrefetchBankRequests', 0)),
                    str(row.get('RuntimePrefetchBankInterferenceStall', 0)),
                    str(row.get('RuntimePrefetchBankInterferenceCycles', 0)),
                    str(row.get('PrefetchAwareCombinedBankCycles', 0)),
                    str(row.get('PrefetchAwareCombinedBankStall', 0)),
                    str(row.get('NormalBankStall', 0)),
                ]) + ',\n')

    def _apply_prefetch_experiment_model(self):
        """Compute per-layer prefetch experiment stats and attach them to layer objects.

        Constraints enforced:
        - EnablePrefetch=False or PrefetchWindow=0 must be identical to baseline (no changes).
        - PrefetchHiddenCycles <= OriginalMemoryLatency
        - PrefetchResidualStall >= 0
        - Dynamic+Prefetch reuses the existing EnableDynamic allocation (already selected per-layer).
        """
        assert self.params_set_flag, 'Simulator parameters are not set'

        enable_bank_model = bool(self.conf.get_enable_bank_model())
        enable_prefetch = bool(getattr(self.conf, 'get_enable_prefetch', lambda: False)())
        prefetch_window = int(getattr(self.conf, 'get_prefetch_window', lambda: 0)())
        prefetch_targets = []
        if hasattr(self.conf, 'get_prefetch_target_list'):
            prefetch_targets = list(self.conf.get_prefetch_target_list())
        prefetch_target_raw = str(getattr(self.conf, 'get_prefetch_target', lambda: 'ifmap,filter')())

        num_layers = len(self.single_layer_sim_object_list)

        # Default: stamp baseline items only.
        if (not enable_bank_model) or (not enable_prefetch) or prefetch_window <= 0 or num_layers <= 0:
            for lid, layer_obj in enumerate(self.single_layer_sim_object_list):
                comp_items = layer_obj.get_compute_report_items()
                base_total = int(comp_items[1])
                base_stall = int(comp_items[2])
                items = {
                    'PrefetchEnabled': bool(enable_prefetch and prefetch_window > 0 and enable_bank_model),
                    'PrefetchWindow': int(prefetch_window) if enable_prefetch else 0,
                    'PrefetchTarget': prefetch_target_raw,
                    'PrefetchIssuedCycles': 0,
                    'PrefetchHiddenCycles': 0,
                    'PrefetchResidualStall': int(base_stall),
                    'PrefetchBankConflictCycles': 0,
                    'EffectiveMemoryLatency': int(base_stall),
                    'OriginalMemoryLatency': int(base_stall),
                    'TotalCyclesWithPrefetch': int(base_total),
                    'TotalCyclesNoPrefetch': int(base_total),
                }
                layer_obj.set_prefetch_report_items(items)
            return

        # --- Baseline per-layer stats and prefetch demand characteristics ---
        base_total_cycles = [0 for _ in range(num_layers)]
        base_stall_cycles = [0 for _ in range(num_layers)]
        compute_overlap_cycles = [0 for _ in range(num_layers)]
        prefetch_required_cycles = [0 for _ in range(num_layers)]
        prefetch_conflict_cycles = [0 for _ in range(num_layers)]
        max_hideable_latency = [0 for _ in range(num_layers)]

        # Precompute the "prefetch-only" cost per target layer using the already-selected
        # bank split (static or dynamic) from the baseline simulation.
        for lid, layer_obj in enumerate(self.single_layer_sim_object_list):
            comp_items = layer_obj.get_compute_report_items()
            base_total = int(comp_items[1])
            base_stall = int(comp_items[2])
            base_total_cycles[lid] = base_total
            base_stall_cycles[lid] = base_stall
            compute_overlap_cycles[lid] = max(0, base_total - base_stall)

            # For lid==0, it can still be a prefetch target if window>1 from earlier, but none.
            if lid == 0:
                continue

            ifmap_demand, filter_demand, ofmap_demand = layer_obj.get_cached_demand_matrices()
            if ifmap_demand is None or filter_demand is None or ofmap_demand is None:
                continue

            # Use the per-layer chosen bank split (EnableDynamic already applied in baseline).
            bank_items = layer_obj.get_bank_report_items() if hasattr(layer_obj, 'get_bank_report_items') else {}
            counts = {
                'ifmap': int(bank_items.get('ifmap_banknum', getattr(self.conf, 'ifmap_sram_bank_num', 1))),
                'filter': int(bank_items.get('filter_banknum', getattr(self.conf, 'filter_sram_bank_num', 1))),
                'ofmap': int(bank_items.get('ofmap_banknum', getattr(self.conf, 'ofmap_sram_bank_num', 1))),
            }

            # Build subset demands: only prefetch requested targets; never prefetch ofmap.
            ifmap_pf = ifmap_demand if ('ifmap' in prefetch_targets) else np.full_like(ifmap_demand, -1)
            filter_pf = filter_demand if ('filter' in prefetch_targets) else np.full_like(filter_demand, -1)
            ofmap_pf = np.full_like(ofmap_demand, -1)

            sim_pf = layer_obj.memory_system.simulate_with_explicit_counts(
                counts=counts,
                ifmap_demand_mat=ifmap_pf,
                filter_demand_mat=filter_pf,
                ofmap_demand_mat=ofmap_pf,
                use_allocation_bases=True,
            )
            pf_total = int(sim_pf.get('total_cycles', 0))
            pf_stall = int(sim_pf.get('stall_cycles', 0))

            prefetch_required_cycles[lid] = max(0, pf_total)
            prefetch_conflict_cycles[lid] = max(0, pf_stall)
            # Upper bound on hideable latency cannot exceed baseline stall.
            max_hideable_latency[lid] = min(int(base_stall), max(0, pf_stall))

        # --- Prefetch scheduling across layers (low priority, uses overlap compute cycles) ---
        prefetch_progress = [0 for _ in range(num_layers)]
        for issuer in range(num_layers):
            budget = int(compute_overlap_cycles[issuer])
            if budget <= 0:
                continue
            for target in range(issuer + 1, min(num_layers, issuer + prefetch_window + 1)):
                need = int(prefetch_required_cycles[target])
                if need <= 0:
                    continue
                remaining = need - int(prefetch_progress[target])
                if remaining <= 0:
                    continue
                consume = min(remaining, budget)
                prefetch_progress[target] += int(consume)
                budget -= int(consume)
                if budget <= 0:
                    break

        # --- Attach per-layer report items and validate ---
        for lid, layer_obj in enumerate(self.single_layer_sim_object_list):
            original_latency = int(base_stall_cycles[lid])
            issued = int(prefetch_progress[lid])
            hidden = min(int(max_hideable_latency[lid]), issued)
            hidden = min(hidden, original_latency)
            residual = max(0, original_latency - hidden)

            # Estimate how many of the prefetch conflict cycles were actually incurred for the
            # issued portion (linear scaling; conservative abstraction).
            pf_need = int(prefetch_required_cycles[lid])
            pf_conf_need = int(prefetch_conflict_cycles[lid])
            if pf_need > 0 and issued > 0:
                pf_conf_issued = int(round(float(pf_conf_need) * (float(issued) / float(pf_need))))
            else:
                pf_conf_issued = 0
            pf_conf_issued = max(0, min(pf_conf_need, pf_conf_issued))

            total_no_pf = int(base_total_cycles[lid])
            total_with_pf = max(0, int(total_no_pf - hidden))

            # Simple validations
            if hidden > original_latency:
                raise ValueError(f"PrefetchHiddenCycles ({hidden}) exceeds OriginalMemoryLatency ({original_latency}) at layer {lid}")
            if residual < 0:
                raise ValueError(f"PrefetchResidualStall became negative at layer {lid}")
            if (not enable_prefetch) or prefetch_window <= 0:
                if total_with_pf != total_no_pf:
                    raise ValueError(f"Prefetch disabled but TotalCycles changed at layer {lid}")

            items = {
                'PrefetchEnabled': True,
                'PrefetchWindow': int(prefetch_window),
                'PrefetchTarget': prefetch_target_raw,
                'PrefetchIssuedCycles': int(issued),
                'PrefetchHiddenCycles': int(hidden),
                'PrefetchResidualStall': int(residual),
                'PrefetchBankConflictCycles': int(pf_conf_issued),
                'EffectiveMemoryLatency': int(residual),
                'OriginalMemoryLatency': int(original_latency),
                'TotalCyclesWithPrefetch': int(total_with_pf),
                'TotalCyclesNoPrefetch': int(total_no_pf),
            }
            layer_obj.set_prefetch_report_items(items)

    #
    def generate_reports(self):
        """
        Method to generate the report files for scalesim run if the runs are already completed. For
        each layer, this method collects the report data from single_layer_sim objects and then
        prints them out into COMPUTE_REPORT.csv, BANDWIDTH_REPORT.csv, DETAILED_ACCESS_REPORT.csv
        and SPARSE_REPORT.csv files.
        """
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        compute_report_name = self.top_path + '/COMPUTE_REPORT.csv'
        compute_report = open(compute_report_name, 'w')
        header = ('LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %,'
              ' Compute Util %,'
              ' PrefetchEnabled, PrefetchWindow, PrefetchTarget, PrefetchIssuedCycles, PrefetchHiddenCycles,'
              ' PrefetchResidualStall, PrefetchBankConflictCycles, EffectiveMemoryLatency, OriginalMemoryLatency,'
              ' TotalCyclesWithPrefetch, TotalCyclesNoPrefetch,\n')
        compute_report.write(header)
        
        # Create TIME_REPORT.csv for linear model time conversion
        time_report_name = self.top_path + '/TIME_REPORT.csv'
        time_report = open(time_report_name, 'w')
        time_report.write('LayerID, Time (us),\n')

        bandwidth_report_name = self.top_path + '/BANDWIDTH_REPORT.csv'
        bandwidth_report = open(bandwidth_report_name, 'w')
        if self.conf.sparsity_support is True:
            header = ('LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg FILTER Metadata SRAM BW,'
                      ' Avg OFMAP SRAM BW, ')
        else:
            header = 'LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, '
        header += 'Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW,\n'
        bandwidth_report.write(header)

        detail_report_name = self.top_path + '/DETAILED_ACCESS_REPORT.csv'
        detail_report = open(detail_report_name, 'w')
        header = 'LayerID, '
        header += 'SRAM IFMAP Start Cycle, SRAM IFMAP Stop Cycle, SRAM IFMAP Reads, '
        header += 'SRAM Filter Start Cycle, SRAM Filter Stop Cycle, SRAM Filter Reads, '
        header += 'SRAM OFMAP Start Cycle, SRAM OFMAP Stop Cycle, SRAM OFMAP Writes, '
        header += 'DRAM IFMAP Start Cycle, DRAM IFMAP Stop Cycle, DRAM IFMAP Reads, '
        header += 'DRAM Filter Start Cycle, DRAM Filter Stop Cycle, DRAM Filter Reads, '
        header += 'DRAM OFMAP Start Cycle, DRAM OFMAP Stop Cycle, DRAM OFMAP Writes,\n'
        detail_report.write(header)

        bank_model_report = None
        if self.conf.get_enable_bank_model():
            bank_model_report_name = self.top_path + '/BANK_MODEL_REPORT.csv'
            bank_model_report = open(bank_model_report_name, 'w')
            header = 'LayerID, EnableBankModel, EnableDynamic, EnableCapacityPenalty, bank_conflict_penalty, DRAMPenaltyScale, '
            header += 'total_banknum, ifmap_banknum, filter_banknum, ofmap_banknum, allocation_ratio, '
            header += 'bank_capacity_kb, ifmap_total_capacity_kb, filter_total_capacity_kb, ofmap_total_capacity_kb, '
            header += 'ifmap_elements, filter_elements, ofmap_elements, '
            header += 'ifmap_capacity_utilization, filter_capacity_utilization, ofmap_capacity_utilization, '
            header += 'ifmap_overflow_to_dram, filter_overflow_to_dram, ofmap_overflow_to_dram, '
            header += 'ifmap_dram_penalty_cycles_per_request, filter_dram_penalty_cycles_per_request, ofmap_dram_penalty_cycles_per_request, '
            header += 'ifmap_bank_conflict_delay, filter_bank_conflict_delay, ofmap_bank_conflict_delay, total_bank_conflict_delay, '
            header += 'total_cycles, stall_cycles_due_to_bank_conflict,\n'
            header = header.rstrip('\n')
            header += (' PrefetchEnabled, PrefetchWindow, PrefetchTarget, PrefetchIssuedCycles, PrefetchHiddenCycles,'
                       ' PrefetchResidualStall, PrefetchBankConflictCycles, EffectiveMemoryLatency, OriginalMemoryLatency,'
                       ' TotalCyclesWithPrefetch, TotalCyclesNoPrefetch,\n')
            bank_model_report.write(header)

        # Prefetch experiment report (always generated; zeros when disabled)
        prefetch_report_name = self.top_path + '/PREFETCH_REPORT.csv'
        prefetch_report = open(prefetch_report_name, 'w')
        prefetch_header = ('LayerID, PrefetchEnabled, PrefetchWindow, PrefetchTarget, PrefetchIssuedCycles, PrefetchHiddenCycles,'
                           ' PrefetchResidualStall, PrefetchBankConflictCycles, EffectiveMemoryLatency, OriginalMemoryLatency,'
                           ' TotalCyclesWithPrefetch, TotalCyclesNoPrefetch,\n')
        prefetch_report.write(prefetch_header)

        if self.conf.sparsity_support is True:
            sparse_report_name = self.top_path + '/SPARSE_REPORT.csv'
            sparse_report = open(sparse_report_name, 'w')
            header = 'LayerID, '
            header += 'Sparsity Representation, '
            header += ('Original Filter Storage, New Storage (Filter+Metadata),'
                       ' Filter Metadata Storage, ')
            header += 'Avg FILTER Metadata SRAM BW, '
            header += '\n'
            sparse_report.write(header)

        for lid in range(len(self.single_layer_sim_object_list)):
            single_layer_obj = self.single_layer_sim_object_list[lid]
            is_blackbox_layer = lid in self.ep_moe_blackbox_layer_ids
            if is_blackbox_layer:
                analytical = self._get_layer_analytical_stats(lid)
                analytical_cycles = int(analytical['compute_cycles'])
                compute_report_items_this_layer = [analytical_cycles, analytical_cycles, 0, 0, 0, 0]
                prefetch_items = {
                    'PrefetchEnabled': False,
                    'PrefetchWindow': 0,
                    'PrefetchTarget': 'blackbox_analytical',
                    'TotalCyclesWithPrefetch': analytical_cycles,
                    'TotalCyclesNoPrefetch': analytical_cycles,
                }
            else:
                compute_report_items_this_layer = single_layer_obj.get_compute_report_items()
                prefetch_items = single_layer_obj.get_prefetch_report_items() if hasattr(single_layer_obj, 'get_prefetch_report_items') else {}
            log = str(lid) +', '
            log += ', '.join([str(x) for x in compute_report_items_this_layer])
            log += ', ' + ', '.join([
                str(prefetch_items.get('PrefetchEnabled', False)),
                str(prefetch_items.get('PrefetchWindow', 0)),
                str(prefetch_items.get('PrefetchTarget', '')),
                str(prefetch_items.get('PrefetchIssuedCycles', 0)),
                str(prefetch_items.get('PrefetchHiddenCycles', 0)),
                str(prefetch_items.get('PrefetchResidualStall', 0)),
                str(prefetch_items.get('PrefetchBankConflictCycles', 0)),
                str(prefetch_items.get('EffectiveMemoryLatency', 0)),
                str(prefetch_items.get('OriginalMemoryLatency', 0)),
                str(prefetch_items.get('TotalCyclesWithPrefetch', compute_report_items_this_layer[0] if compute_report_items_this_layer else 0)),
                str(prefetch_items.get('TotalCyclesNoPrefetch', compute_report_items_this_layer[1] if compute_report_items_this_layer else 0)),
            ])
            log += ',\n'
            compute_report.write(log)
            
            # Generate TIME_REPORT entry using linear model
            total_cycles = compute_report_items_this_layer[1]  # Total Cycles (not including prefetch)
            time_linear_model = self.conf.get_time_linear_model()
            
            # Get spatiotemporal dimensions for this layer
            dataflow = self.conf.get_dataflow()
            s_row, s_col, t_time = self.topo.get_spatiotemporal_dims(layer_id=lid, df=dataflow)
            
            
            # Apply the appropriate linear model based on config
            if time_linear_model == 'TPUv4':
                time_us = tpuv4_linear_model(total_cycles, s_row, s_col, t_time)
            elif time_linear_model == 'TPUv5e':
                time_us = tpuv5e_linear_model(total_cycles, s_row, s_col, t_time)
            elif time_linear_model == 'TPUv6e':
                time_us = tpuv6e_linear_model(total_cycles, s_row, s_col, t_time)
            else:
                # Default: no conversion, just use cycles as time
                time_us = total_cycles
            
            time_log = str(lid) + ', ' + str(time_us) + ',\n'
            time_report.write(time_log)

            bandwidth_count = 7 if self.conf.sparsity_support is True else 6
            bandwidth_report_items_this_layer = (
                [0] * bandwidth_count if is_blackbox_layer
                else single_layer_obj.get_bandwidth_report_items()
            )
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in bandwidth_report_items_this_layer])
            log += ',\n'
            bandwidth_report.write(log)

            detail_report_items_this_layer = (
                [0] * 18 if is_blackbox_layer
                else single_layer_obj.get_detail_report_items()
            )
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in detail_report_items_this_layer])
            log += ',\n'
            detail_report.write(log)

            if self.conf.sparsity_support is True:
                sparse_report_items_this_layer = (
                    [0] * 4 if is_blackbox_layer
                    else single_layer_obj.get_sparse_report_items()
                )
                log = str(lid) + ', ' + self.conf.sparsity_representation + ', '
                log += ', '.join([str(x) for x in sparse_report_items_this_layer])
                log += ',\n'
                sparse_report.write(log)

            if self.conf.get_enable_bank_model() and bank_model_report is not None and not is_blackbox_layer:
                bank_items = single_layer_obj.get_bank_report_items()
                log = str(lid) + ', '
                log += ', '.join([
                    str(bank_items.get('EnableBankModel', False)),
                    str(bank_items.get('EnableDynamic', False)),
                    str(bank_items.get('EnableCapacityPenalty', True)),
                    str(bank_items.get('bank_conflict_penalty', 1)),
                    str(bank_items.get('DRAMPenaltyScale', 8)),
                    str(bank_items.get('total_banknum', 0)),
                    str(bank_items.get('ifmap_banknum', 0)),
                    str(bank_items.get('filter_banknum', 0)),
                    str(bank_items.get('ofmap_banknum', 0)),
                    str(bank_items.get('allocation_ratio', '0:0:0')),
                    str(bank_items.get('bank_capacity_kb', 0)),
                    str(bank_items.get('ifmap_total_capacity_kb', 0)),
                    str(bank_items.get('filter_total_capacity_kb', 0)),
                    str(bank_items.get('ofmap_total_capacity_kb', 0)),
                    str(bank_items.get('ifmap_elements', 0)),
                    str(bank_items.get('filter_elements', 0)),
                    str(bank_items.get('ofmap_elements', 0)),
                    str(bank_items.get('ifmap_capacity_utilization', 0)),
                    str(bank_items.get('filter_capacity_utilization', 0)),
                    str(bank_items.get('ofmap_capacity_utilization', 0)),
                    str(bank_items.get('ifmap_overflow_to_dram', False)),
                    str(bank_items.get('filter_overflow_to_dram', False)),
                    str(bank_items.get('ofmap_overflow_to_dram', False)),
                    str(bank_items.get('ifmap_dram_penalty_cycles_per_request', 0)),
                    str(bank_items.get('filter_dram_penalty_cycles_per_request', 0)),
                    str(bank_items.get('ofmap_dram_penalty_cycles_per_request', 0)),
                    str(bank_items.get('ifmap_bank_conflict_delay', 0)),
                    str(bank_items.get('filter_bank_conflict_delay', 0)),
                    str(bank_items.get('ofmap_bank_conflict_delay', 0)),
                    str(bank_items.get('total_bank_conflict_delay', 0)),
                    str(bank_items.get('total_cycles', 0)),
                    str(bank_items.get('stall_cycles_due_to_bank_conflict', 0)),
                ])
                log += ', ' + ', '.join([
                    str(prefetch_items.get('PrefetchEnabled', False)),
                    str(prefetch_items.get('PrefetchWindow', 0)),
                    str(prefetch_items.get('PrefetchTarget', '')),
                    str(prefetch_items.get('PrefetchIssuedCycles', 0)),
                    str(prefetch_items.get('PrefetchHiddenCycles', 0)),
                    str(prefetch_items.get('PrefetchResidualStall', 0)),
                    str(prefetch_items.get('PrefetchBankConflictCycles', 0)),
                    str(prefetch_items.get('EffectiveMemoryLatency', 0)),
                    str(prefetch_items.get('OriginalMemoryLatency', 0)),
                    str(prefetch_items.get('TotalCyclesWithPrefetch', 0)),
                    str(prefetch_items.get('TotalCyclesNoPrefetch', 0)),
                ])
                log += ',\n'
                bank_model_report.write(log)

            # Prefetch report row
            pf_log = str(lid) + ', ' + ', '.join([
                str(prefetch_items.get('PrefetchEnabled', False)),
                str(prefetch_items.get('PrefetchWindow', 0)),
                str(prefetch_items.get('PrefetchTarget', '')),
                str(prefetch_items.get('PrefetchIssuedCycles', 0)),
                str(prefetch_items.get('PrefetchHiddenCycles', 0)),
                str(prefetch_items.get('PrefetchResidualStall', 0)),
                str(prefetch_items.get('PrefetchBankConflictCycles', 0)),
                str(prefetch_items.get('EffectiveMemoryLatency', 0)),
                str(prefetch_items.get('OriginalMemoryLatency', 0)),
                str(prefetch_items.get('TotalCyclesWithPrefetch', 0)),
                str(prefetch_items.get('TotalCyclesNoPrefetch', 0)),
            ]) + ',\n'
            prefetch_report.write(pf_log)

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()
        time_report.close()
        prefetch_report.close()
        if self.conf.sparsity_support is True:
            sparse_report.close()
        if bank_model_report is not None:
            bank_model_report.close()

        if self.conf.get_enable_ep_moe():
            self._write_ep_moe_config_report()
            self._write_ep_moe_runtime_state_report()
            self._write_ep_moe_timeline_report()
            self._write_ep_moe_bank_allocation_report()
            self._write_ep_moe_routing_report()
            self._write_ep_moe_event_report()
            self._write_ep_moe_chunk_report()
            self._write_ep_moe_run_manifest()
            self._write_ep_moe_layer_execution_report()

            ep_moe_report_name = self.top_path + '/EP_MOE_REPORT.csv'
            with open(ep_moe_report_name, 'w', encoding='utf-8') as ep_report:
                ep_report.write(
                    'MoEGroupID, ExpertID, GPUId, LocalExpertID, IsDetailedGPU, '
                    'IsActiveExpert, RoutingRank, RoutingPolicy, '
                    'LayerIDs, LayerNames, ExpertStartCycle, ExpertFinishCycle, '
                    'ExpertCycles, EstimationMode, TokensPerExpert, HiddenDim, '
                    'ExpertMACs, ExpertWeightBytes, BlackBoxComputeCycles, '
                    'BlackBoxWeightBWCycles, BlackBoxCommunicationBytes, '
                    'BlackBoxCommunicationCycles, TokenDispatchBytes, OutputCombineBytes, '
                    'TokenDispatchCycles, OutputCombineCycles, CommunicationOverlapCycles, '
                    'ExposedCommunicationCycles, CommunicationOverlapMode, '
                    'WeightChunkCount, InitialWeightStall, WeightLoadingStall, '
                    'PrefetchHit, PrefetchMiss, PrefetchHitRate, PrefetchMissStall, '
                    'PrefetchBandwidthOverhead, PrefetchInterferenceStall, '
                    'UsefulPrefetchTraffic, UselessPrefetchTraffic, '
                    'ComputeWithPrefetchCycles, PrefetchBankModelCycles, '
                    'PrefetchBankConflictCycles, PrefetchBankRequests, '
                    'CombinedBankModelCycles, CombinedBankConflictCycles, '
                    'PrefetchBankInterferenceStall, PrefetchBankInterferenceCycles, '
                    'BankAllocationMode, BankAllocationRatios, DynamicFallbackCount, '
                    'DynamicBankOverheadModel, EffectivePerBankBandwidth, '
                    'BaseExpertCycles, RuntimePrefetchBankInterferenceStall, '
                    'RuntimePrefetchBankInterferenceCycles, RuntimePrefetchBankRequests, '
                    'RuntimeBlackBoxBackgroundPressureBytes, '
                    'RuntimeBlackBoxBackgroundPressureCycles, '
                    'RuntimeBlackBoxBackgroundPressureStall, '
                    'MoEGroupTime,\n'
                )
                for row in self.ep_moe_report_rows:
                    ep_report.write(', '.join([
                        str(row.get('MoEGroupID', 0)),
                        str(row.get('ExpertID', 0)),
                        str(row.get('GPUId', 0)),
                        str(row.get('LocalExpertID', 0)),
                        str(row.get('IsDetailedGPU', False)),
                        str(row.get('IsActiveExpert', True)),
                        str(row.get('RoutingRank', -1)),
                        str(row.get('RoutingPolicy', 'all')),
                        str(row.get('LayerIDs', '')),
                        str(row.get('LayerNames', '')),
                        str(row.get('ExpertStartCycle', 0)),
                        str(row.get('ExpertFinishCycle', 0)),
                        str(row.get('ExpertCycles', 0)),
                        str(row.get('EstimationMode', '')),
                        str(row.get('TokensPerExpert', 0)),
                        str(row.get('HiddenDim', 0)),
                        str(row.get('ExpertMACs', 0)),
                        str(row.get('ExpertWeightBytes', 0)),
                        str(row.get('BlackBoxComputeCycles', 0)),
                        str(row.get('BlackBoxWeightBWCycles', 0)),
                        str(row.get('BlackBoxCommunicationBytes', 0)),
                        str(row.get('BlackBoxCommunicationCycles', 0)),
                        str(row.get('TokenDispatchBytes', 0)),
                        str(row.get('OutputCombineBytes', 0)),
                        str(row.get('TokenDispatchCycles', 0)),
                        str(row.get('OutputCombineCycles', 0)),
                        str(row.get('CommunicationOverlapCycles', 0)),
                        str(row.get('ExposedCommunicationCycles', 0)),
                        str(row.get('CommunicationOverlapMode', '')),
                        str(row.get('WeightChunkCount', 0)),
                        str(row.get('InitialWeightStall', 0)),
                        str(row.get('WeightLoadingStall', 0)),
                        str(row.get('PrefetchHit', 0)),
                        str(row.get('PrefetchMiss', 0)),
                        str(row.get('PrefetchHitRate', 0)),
                        str(row.get('PrefetchMissStall', 0)),
                        str(row.get('PrefetchBandwidthOverhead', 0)),
                        str(row.get('PrefetchInterferenceStall', 0)),
                        str(row.get('UsefulPrefetchTraffic', 0)),
                        str(row.get('UselessPrefetchTraffic', 0)),
                        str(row.get('ComputeWithPrefetchCycles', 0)),
                        str(row.get('PrefetchBankModelCycles', 0)),
                        str(row.get('PrefetchBankConflictCycles', 0)),
                        str(row.get('PrefetchBankRequests', 0)),
                        str(row.get('CombinedBankModelCycles', 0)),
                        str(row.get('CombinedBankConflictCycles', 0)),
                        str(row.get('PrefetchBankInterferenceStall', 0)),
                        str(row.get('PrefetchBankInterferenceCycles', 0)),
                        str(row.get('BankAllocationMode', '')),
                        str(row.get('BankAllocationRatios', '')),
                        str(row.get('DynamicFallbackCount', 0)),
                        str(row.get('DynamicBankOverheadModel', '')),
                        str(row.get('EffectivePerBankBandwidth', 0)),
                        str(row.get('BaseExpertCycles', 0)),
                        str(row.get('RuntimePrefetchBankInterferenceStall', 0)),
                        str(row.get('RuntimePrefetchBankInterferenceCycles', 0)),
                        str(row.get('RuntimePrefetchBankRequests', 0)),
                        str(row.get('RuntimeBlackBoxBackgroundPressureBytes', 0)),
                        str(row.get('RuntimeBlackBoxBackgroundPressureCycles', 0)),
                        str(row.get('RuntimeBlackBoxBackgroundPressureStall', 0)),
                        str(row.get('MoEGroupTime', 0)),
                    ]) + ',\n')

            ep_moe_summary_name = self.top_path + '/EP_MOE_SUMMARY.csv'
            with open(ep_moe_summary_name, 'w', encoding='utf-8') as ep_summary:
                ep_summary.write(
                    'MoEGroupID, NumExperts, NumActiveExperts, NumInactiveExperts, '
                    'NumDetailedExperts, NumBlackBoxExperts, '
                    'NumGPUs, DetailedGPUId, ExpertsPerGPU, TopK, '
                    'InitialChunk, ChunkPrefetchWindow, MoEGroupTime, '
                    'TotalExpertCycles, MaxExpertCycles, ExpertCycleImbalance, '
                    'ExpertTokenImbalance, GPULoadImbalanceCycles, AverageGPUUtilization, '
                    'MinimumGPUUtilization, AverageDetailedBankCapacityUtilization, '
                    'MaximumDetailedBankCapacityUtilization, DetailedDRAMTrafficBytes, '
                    'EstimatedBlackBoxDRAMTrafficBytes, TotalInterconnectQueueWait, '
                    'TotalExpertWaitingCycles, MaxExpertWaitingCycles, '
                    'TotalPrefetchHit, TotalPrefetchMiss, AvgPrefetchHitRate, '
                    'TotalPrefetchMissStall, TotalWeightLoadingStall, TotalPrefetchBankInterferenceStall, '
                    'TotalUsefulPrefetchTraffic, TotalUselessPrefetchTraffic, '
                    'TotalPrefetchBandwidthOverhead, TotalCommunicationBytes, '
                    'MaxCommunicationCycles, TotalTokenDispatchBytes, TotalOutputCombineBytes, '
                    'MaxTokenDispatchCycles, MaxOutputCombineCycles, '
                    'TotalCommunicationOverlapCycles, MaxExposedCommunicationCycles, '
                    'BankAllocationModes, '
                    'TotalDynamicFallbackCount, TotalRuntimePrefetchBankInterferenceStall, '
                    'GroupRuntimeBlackBoxBackgroundPressureBytes, '
                    'MaxRuntimeBlackBoxBackgroundPressureCycles, '
                    'GroupRuntimeBlackBoxBackgroundPressureStall, '
                    'DynamicBankOverheadModel,\n'
                )
                for row in self._build_ep_moe_summary_rows():
                    ep_summary.write(', '.join([
                        str(row.get('MoEGroupID', 0)),
                        str(row.get('NumExperts', 0)),
                        str(row.get('NumActiveExperts', 0)),
                        str(row.get('NumInactiveExperts', 0)),
                        str(row.get('NumDetailedExperts', 0)),
                        str(row.get('NumBlackBoxExperts', 0)),
                        str(self.conf.get_num_gpus()),
                        str(self.conf.get_detailed_gpu_id()),
                        str(self.conf.get_experts_per_gpu()),
                        str(self.conf.get_top_k()),
                        str(self.conf.get_initial_chunk()),
                        str(self.conf.get_chunk_prefetch_window()),
                        str(row.get('MoEGroupTime', 0)),
                        str(row.get('TotalExpertCycles', 0)),
                        str(row.get('MaxExpertCycles', 0)),
                        str(row.get('ExpertCycleImbalance', 0)),
                        str(row.get('ExpertTokenImbalance', 0)),
                        str(row.get('GPULoadImbalanceCycles', 0)),
                        str(row.get('AverageGPUUtilization', 0)),
                        str(row.get('MinimumGPUUtilization', 0)),
                        str(row.get('AverageDetailedBankCapacityUtilization', 0)),
                        str(row.get('MaximumDetailedBankCapacityUtilization', 0)),
                        str(row.get('DetailedDRAMTrafficBytes', 0)),
                        str(row.get('EstimatedBlackBoxDRAMTrafficBytes', 0)),
                        str(row.get('TotalInterconnectQueueWait', 0)),
                        str(row.get('TotalExpertWaitingCycles', 0)),
                        str(row.get('MaxExpertWaitingCycles', 0)),
                        str(row.get('TotalPrefetchHit', 0)),
                        str(row.get('TotalPrefetchMiss', 0)),
                        str(row.get('AvgPrefetchHitRate', 0)),
                        str(row.get('TotalPrefetchMissStall', 0)),
                        str(row.get('TotalWeightLoadingStall', 0)),
                        str(row.get('TotalPrefetchBankInterferenceStall', 0)),
                        str(row.get('TotalUsefulPrefetchTraffic', 0)),
                        str(row.get('TotalUselessPrefetchTraffic', 0)),
                        str(row.get('TotalPrefetchBandwidthOverhead', 0)),
                        str(row.get('TotalCommunicationBytes', 0)),
                        str(row.get('MaxCommunicationCycles', 0)),
                        str(row.get('TotalTokenDispatchBytes', 0)),
                        str(row.get('TotalOutputCombineBytes', 0)),
                        str(row.get('MaxTokenDispatchCycles', 0)),
                        str(row.get('MaxOutputCombineCycles', 0)),
                        str(row.get('TotalCommunicationOverlapCycles', 0)),
                        str(row.get('MaxExposedCommunicationCycles', 0)),
                        str(row.get('BankAllocationModes', '')),
                        str(row.get('TotalDynamicFallbackCount', 0)),
                        str(row.get('TotalRuntimePrefetchBankInterferenceStall', 0)),
                        str(row.get('GroupRuntimeBlackBoxBackgroundPressureBytes', 0)),
                        str(row.get('MaxRuntimeBlackBoxBackgroundPressureCycles', 0)),
                        str(row.get('GroupRuntimeBlackBoxBackgroundPressureStall', 0)),
                        str(row.get('DynamicBankOverheadModel', '')),
                    ]) + ',\n')

        # Also write a one-line summary CSV for quick comparisons
        summary_name = self.top_path + '/PREFETCH_SUMMARY.csv'
        with open(summary_name, 'w', encoding='utf-8') as fsum:
            fsum.write('TotalCyclesWithPrefetch, TotalCyclesNoPrefetch, TotalPrefetchHiddenCycles, TotalPrefetchIssuedCycles,\n')
            tot_with = 0
            tot_no = 0
            tot_hidden = 0
            tot_issued = 0
            for layer_id, layer_obj in enumerate(self.single_layer_sim_object_list):
                if layer_id in self.ep_moe_blackbox_layer_ids:
                    analytical_cycles = int(self._get_layer_analytical_stats(layer_id)['compute_cycles'])
                    pf = {
                        'TotalCyclesWithPrefetch': analytical_cycles,
                        'TotalCyclesNoPrefetch': analytical_cycles,
                    }
                else:
                    pf = layer_obj.get_prefetch_report_items() if hasattr(layer_obj, 'get_prefetch_report_items') else {}
                tot_with += int(pf.get('TotalCyclesWithPrefetch', 0))
                tot_no += int(pf.get('TotalCyclesNoPrefetch', 0))
                tot_hidden += int(pf.get('PrefetchHiddenCycles', 0))
                tot_issued += int(pf.get('PrefetchIssuedCycles', 0))
            fsum.write(f'{tot_with}, {tot_no}, {tot_hidden}, {tot_issued},\n')

    #
    def get_total_cycles(self):
        """
        Return the workload completion time in cycles.

        Legacy runs execute layers sequentially, so their completion time is the
        sum of each layer's overall cycle count (including any enabled legacy
        prefetch adjustment). EP-MoE runs have an explicit mixed layer/group
        timeline; its final finish cycle is the authoritative completion time.
        """
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        if self.conf.get_enable_ep_moe() and self.ep_moe_timeline_rows:
            return int(max(
                int(row.get('FinishCycle', 0))
                for row in self.ep_moe_timeline_rows
            ))

        return int(sum(
            int(layer_obj.get_compute_report_items()[0])
            for layer_obj in self.single_layer_sim_object_list
        ))
