"""
This file contains the 'simulator' class that simulates the entire model using the class
'single_layer_sim' and generates the reports (.csv files).
"""

import os
import re

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
        self.ep_moe_report_rows = []

        self.params_set_flag = False
        self.all_layer_run_done = False

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   layout_obj=layout(),
                   top_path="./",
                   verbosity=True,
                   save_trace=True
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

        # Calculate inferrable parameters here
        self.num_layers = self.topo.get_num_layers()
        self.ep_moe_execution_plan = []
        self.ep_moe_groups = []
        self.ep_moe_report_rows = []

        self.params_set_flag = True

    @staticmethod
    def _parse_moe_layer_name(layer_name):
        """Parse names like MoE-E3-FF2 into expert metadata."""
        match = re.match(r'^MoE-E(\d+)-FF(\d+)$', str(layer_name).strip())
        if match is None:
            return None
        return {
            'expert_id': int(match.group(1)),
            'ffn_part': int(match.group(2)),
        }

    def _build_ep_moe_execution_plan(self):
        """Build a mixed normal-layer / MoE-group plan from topology names.

        This is currently scaffolding for EP-MoE mode. Legacy execution remains
        unchanged unless the EP-MoE runner is explicitly wired in later.
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
            while idx < len(layer_names):
                parsed = self._parse_moe_layer_name(layer_names[idx])
                if parsed is None:
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

            plan.append({
                'type': 'moe_group',
                'group_id': len([p for p in plan if p['type'] == 'moe_group']),
                'start_layer_id': group_start,
                'end_layer_id': idx - 1,
                'experts': [experts[k] for k in sorted(experts.keys())],
            })

        return plan

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
            self.ep_moe_groups = [
                item for item in self.ep_moe_execution_plan
                if item['type'] == 'moe_group'
            ]
            if self.verbose:
                print('EP-MoE mode enabled')
                print('EP-MoE groups detected: ' + str(len(self.ep_moe_groups)))
                for group in self.ep_moe_groups:
                    expert_ids = [str(exp['expert_id']) for exp in group['experts']]
                    print('  MoE group ' + str(group['group_id']) + ': experts ' + ','.join(expert_ids))

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

            if self.verbose:
                layer_id = single_layer_obj.get_layer_id()
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
        self._apply_prefetch_experiment_model()

        if self.conf.get_enable_ep_moe():
            self._compute_ep_moe_report_rows()

        self.generate_reports()

    def _estimate_blackbox_layer_cycles(self, layer_id):
        """Analytical black-box estimate for non-detailed GPUs.

        This intentionally does not use GPU0's bank-conflict result, because black-box GPUs
        should not participate in detailed on-chip bank competition.
        """
        arr_h, arr_w = self.conf.get_array_dims()
        mac_units = max(1, int(arr_h) * int(arr_w))
        mac_ops = int(self.topo.get_layer_mac_ops(layer_id=layer_id))
        return max(1, int(np.ceil(float(mac_ops) / float(mac_units))))

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
            moe_group_time = 0

            for expert in group['experts']:
                gpu_id = int(expert['gpu_id'])
                is_detailed = gpu_id == detailed_gpu_id
                expert_cycles = 0
                layer_ids = []
                layer_names = []

                for layer in expert['layers']:
                    layer_id = int(layer['layer_id'])
                    layer_ids.append(str(layer_id))
                    layer_names.append(str(layer['layer_name']))

                    if is_detailed:
                        comp_items = self.single_layer_sim_object_list[layer_id].get_compute_report_items()
                        expert_cycles += int(comp_items[0])
                    else:
                        expert_cycles += self._estimate_blackbox_layer_cycles(layer_id)

                expert_start = 0
                expert_finish = int(expert_start + expert_cycles)
                moe_group_time = max(moe_group_time, expert_finish)

                pending_rows.append({
                    'MoEGroupID': int(group['group_id']),
                    'ExpertID': int(expert['expert_id']),
                    'GPUId': gpu_id,
                    'LocalExpertID': int(expert['local_expert_id']),
                    'IsDetailedGPU': bool(is_detailed),
                    'LayerIDs': '|'.join(layer_ids),
                    'LayerNames': '|'.join(layer_names),
                    'ExpertStartCycle': int(expert_start),
                    'ExpertFinishCycle': int(expert_finish),
                    'ExpertCycles': int(expert_cycles),
                    'EstimationMode': 'detailed_scalesim' if is_detailed else 'analytical_blackbox',
                })

            for row in pending_rows:
                row['MoEGroupTime'] = int(moe_group_time)
                self.ep_moe_report_rows.append(row)

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

            bandwidth_report_items_this_layer = single_layer_obj.get_bandwidth_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in bandwidth_report_items_this_layer])
            log += ',\n'
            bandwidth_report.write(log)

            detail_report_items_this_layer = single_layer_obj.get_detail_report_items()
            log = str(lid) + ', '
            log += ', '.join([str(x) for x in detail_report_items_this_layer])
            log += ',\n'
            detail_report.write(log)

            if self.conf.sparsity_support is True:
                sparse_report_items_this_layer = single_layer_obj.get_sparse_report_items()
                log = str(lid) + ', ' + self.conf.sparsity_representation + ', '
                log += ', '.join([str(x) for x in sparse_report_items_this_layer])
                log += ',\n'
                sparse_report.write(log)

            if self.conf.get_enable_bank_model() and bank_model_report is not None:
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
            ep_moe_report_name = self.top_path + '/EP_MOE_REPORT.csv'
            with open(ep_moe_report_name, 'w', encoding='utf-8') as ep_report:
                ep_report.write(
                    'MoEGroupID, ExpertID, GPUId, LocalExpertID, IsDetailedGPU, '
                    'LayerIDs, LayerNames, ExpertStartCycle, ExpertFinishCycle, '
                    'ExpertCycles, EstimationMode, MoEGroupTime,\n'
                )
                for row in self.ep_moe_report_rows:
                    ep_report.write(', '.join([
                        str(row.get('MoEGroupID', 0)),
                        str(row.get('ExpertID', 0)),
                        str(row.get('GPUId', 0)),
                        str(row.get('LocalExpertID', 0)),
                        str(row.get('IsDetailedGPU', False)),
                        str(row.get('LayerIDs', '')),
                        str(row.get('LayerNames', '')),
                        str(row.get('ExpertStartCycle', 0)),
                        str(row.get('ExpertFinishCycle', 0)),
                        str(row.get('ExpertCycles', 0)),
                        str(row.get('EstimationMode', '')),
                        str(row.get('MoEGroupTime', 0)),
                    ]) + ',\n')

        # Also write a one-line summary CSV for quick comparisons
        summary_name = self.top_path + '/PREFETCH_SUMMARY.csv'
        with open(summary_name, 'w', encoding='utf-8') as fsum:
            fsum.write('TotalCyclesWithPrefetch, TotalCyclesNoPrefetch, TotalPrefetchHiddenCycles, TotalPrefetchIssuedCycles,\n')
            tot_with = 0
            tot_no = 0
            tot_hidden = 0
            tot_issued = 0
            for layer_obj in self.single_layer_sim_object_list:
                pf = layer_obj.get_prefetch_report_items() if hasattr(layer_obj, 'get_prefetch_report_items') else {}
                tot_with += int(pf.get('TotalCyclesWithPrefetch', 0))
                tot_no += int(pf.get('TotalCyclesNoPrefetch', 0))
                tot_hidden += int(pf.get('PrefetchHiddenCycles', 0))
                tot_issued += int(pf.get('PrefetchIssuedCycles', 0))
            fsum.write(f'{tot_with}, {tot_no}, {tot_hidden}, {tot_issued},\n')

    #
    def get_total_cycles(self):
        """
        Method which aggregates the total cycles (both compute and stall) across all the layers for
        the given workload.
        """
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        total_cycles = 0
        for layer_obj in self.single_layer_sim_object_list:
            cycles_this_layer = int(layer_obj.get_compute_report_items[0])
            total_cycles += cycles_this_layer

        return total_cycles
