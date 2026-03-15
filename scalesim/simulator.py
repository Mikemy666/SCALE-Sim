"""
This file contains the 'simulator' class that simulates the entire model using the class
'single_layer_sim' and generates the reports (.csv files).
"""

import os
import re

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

        self.params_set_flag = False
        self.all_layer_run_done = False

    def _parse_moe_layer_name(self, layer_name):
        """
        Parse layer names like MoE-E0-FF1 into (expert_id, ff_stage).
        """
        layer_name_str = str(layer_name)
        match = re.match(r'^MoE-E(\d+)-FF(\d+)$', layer_name_str)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    def _build_serial_schedule(self, layer_infos):
        """
        Build the default serial schedule in topology order.
        """
        schedule = {}
        cursor = 0
        for item in layer_infos:
            lid = int(item['layer_id'])
            cyc = max(int(item['cycles']), 0)
            schedule[lid] = {
                'start': cursor,
                'end': cursor + cyc
            }
            cursor += cyc
        return schedule, cursor

    def _build_moe_parallel_schedule(self, layer_infos):
        """
        Build MoE-aware schedule:
        - non-MoE layers remain serial
        - contiguous MoE blocks run experts in parallel
        - each expert remains serial by FF stage order (FF1 -> FF2 -> ...)
        """
        enable_moe_parallel = self.conf.get_enable_moe_parallel_bank_arb() \
            if hasattr(self.conf, 'get_enable_moe_parallel_bank_arb') else False

        schedule = {}
        cursor = 0
        idx = 0
        num_layers = len(layer_infos)

        while idx < num_layers:
            cur_name = layer_infos[idx]['name']
            parsed = self._parse_moe_layer_name(cur_name)

            if (not enable_moe_parallel) or parsed is None:
                lid = int(layer_infos[idx]['layer_id'])
                cyc = max(int(layer_infos[idx]['cycles']), 0)
                schedule[lid] = {
                    'start': cursor,
                    'end': cursor + cyc
                }
                cursor += cyc
                idx += 1
                continue

            block_start = idx
            while idx < num_layers and self._parse_moe_layer_name(layer_infos[idx]['name']) is not None:
                idx += 1
            block_end = idx

            expert_stage_map = {}
            for block_idx in range(block_start, block_end):
                layer_info = layer_infos[block_idx]
                parse_out = self._parse_moe_layer_name(layer_info['name'])
                if parse_out is None:
                    continue
                expert_id, ff_stage = parse_out
                if expert_id not in expert_stage_map:
                    expert_stage_map[expert_id] = []
                expert_stage_map[expert_id].append((ff_stage, layer_info))

            block_finish = cursor
            for expert_id in expert_stage_map:
                expert_cursor = cursor
                stage_entries = sorted(expert_stage_map[expert_id], key=lambda x: x[0])
                for _, layer_info in stage_entries:
                    lid = int(layer_info['layer_id'])
                    cyc = max(int(layer_info['cycles']), 0)
                    schedule[lid] = {
                        'start': expert_cursor,
                        'end': expert_cursor + cyc
                    }
                    expert_cursor += cyc

                if expert_cursor > block_finish:
                    block_finish = expert_cursor

            cursor = block_finish

        return schedule, cursor

    def _generate_moe_parallel_reports(self):
        """
        Emit schedule-oriented reports that compare serial execution and MoE-parallel execution.
        """
        layer_infos = []
        for lid in range(len(self.single_layer_sim_object_list)):
            layer_obj = self.single_layer_sim_object_list[lid]
            compute_items = layer_obj.get_compute_report_items()
            layer_infos.append({
                'layer_id': lid,
                'name': self.topo.get_layer_name(lid),
                'cycles': int(compute_items[1]),
                'compute_stall': int(compute_items[2]),
                'global_bank_conflict_stall': int(layer_obj.get_global_bank_conflict_stall_cycles())
            })

        serial_sched, serial_total = self._build_serial_schedule(layer_infos)
        parallel_sched, parallel_total = self._build_moe_parallel_schedule(layer_infos)

        schedule_report_name = self.top_path + '/MOE_PARALLEL_SCHEDULE_REPORT.csv'
        schedule_report = open(schedule_report_name, 'w')
        schedule_report.write(
            'layer_id,layer_name,is_moe,expert_id,ff_stage,layer_cycles,compute_stall_cycles,'
            'global_bank_conflict_stall_cycles,serial_start,serial_end,parallel_start,parallel_end,start_shift\n'
        )

        moe_layer_ids = []
        for item in layer_infos:
            lid = int(item['layer_id'])
            name = str(item['name'])
            parse_out = self._parse_moe_layer_name(name)
            is_moe = parse_out is not None
            expert_id = parse_out[0] if is_moe else -1
            ff_stage = parse_out[1] if is_moe else -1
            if is_moe:
                moe_layer_ids.append(lid)

            s_start = int(serial_sched[lid]['start'])
            s_end = int(serial_sched[lid]['end'])
            p_start = int(parallel_sched[lid]['start'])
            p_end = int(parallel_sched[lid]['end'])
            start_shift = s_start - p_start

            schedule_report.write(
                f"{lid},{name},{is_moe},{expert_id},{ff_stage},{int(item['cycles'])},"
                f"{int(item['compute_stall'])},{int(item['global_bank_conflict_stall'])},"
                f"{s_start},{s_end},{p_start},{p_end},{start_shift}\n"
            )

        schedule_report.close()

        summary_report_name = self.top_path + '/MOE_PARALLEL_SUMMARY.csv'
        summary_report = open(summary_report_name, 'w')
        summary_report.write('metric,value\n')
        summary_report.write(f'serial_total_cycles,{int(serial_total)}\n')
        summary_report.write(f'parallel_total_cycles,{int(parallel_total)}\n')
        speedup = 0.0
        if parallel_total > 0:
            speedup = float(serial_total) / float(parallel_total)
        summary_report.write(f'estimated_speedup,{speedup}\n')

        if len(moe_layer_ids) > 0:
            moe_serial_start = min(int(serial_sched[x]['start']) for x in moe_layer_ids)
            moe_serial_end = max(int(serial_sched[x]['end']) for x in moe_layer_ids)
            moe_parallel_start = min(int(parallel_sched[x]['start']) for x in moe_layer_ids)
            moe_parallel_end = max(int(parallel_sched[x]['end']) for x in moe_layer_ids)

            summary_report.write(f'moe_serial_window_cycles,{int(moe_serial_end - moe_serial_start)}\n')
            summary_report.write(f'moe_parallel_window_cycles,{int(moe_parallel_end - moe_parallel_start)}\n')
            moe_speedup = 0.0
            if (moe_parallel_end - moe_parallel_start) > 0:
                moe_speedup = float(moe_serial_end - moe_serial_start) / float(moe_parallel_end - moe_parallel_start)
            summary_report.write(f'moe_window_speedup,{moe_speedup}\n')
        else:
            summary_report.write('moe_serial_window_cycles,0\n')
            summary_report.write('moe_parallel_window_cycles,0\n')
            summary_report.write('moe_window_speedup,0.0\n')

        summary_report.close()

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

        self.params_set_flag = True

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

        self.generate_reports()

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
                  ' Compute Util %,\n')
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

        dump_bank_util_csv = self.conf.get_dump_bank_util_csv() if hasattr(self.conf, 'get_dump_bank_util_csv') else False
        bank_report = None
        bank_stall_report = None
        bank_busy_totals = []
        bank_access_totals = []
        bank_denom_totals = []
        total_bank_conflict_stalls = 0
        total_bank_conflict_blocked_cycles = 0
        total_global_bank_conflict_stall_cycles = 0
        if dump_bank_util_csv:
            bank_report_name = self.top_path + '/bank_utilization.csv'
            bank_report = open(bank_report_name, 'w')
            bank_report.write('bank_id,busy_cycles,access_count,utilization\n')

            bank_stall_report_name = self.top_path + '/bank_stall_breakdown.csv'
            bank_stall_report = open(bank_stall_report_name, 'w')
            bank_stall_report.write(
                'layer_id,compute_stall_cycles,bank_conflict_stall_cycles,bank_conflict_blocked_cycles,global_bank_conflict_stall_cycles\n'
            )

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
            log = str(lid) +', '
            log += ', '.join([str(x) for x in compute_report_items_this_layer])
            log += ',\n'
            compute_report.write(log)

            if dump_bank_util_csv:
                bank_conflict_stalls_this_layer = int(single_layer_obj.get_bank_conflict_stall_cycles())
                bank_conflict_blocked_cycles_this_layer = int(single_layer_obj.get_bank_conflict_blocked_cycles())
                global_bank_conflict_stall_cycles_this_layer = int(single_layer_obj.get_global_bank_conflict_stall_cycles())
                total_bank_conflict_stalls += bank_conflict_stalls_this_layer
                total_bank_conflict_blocked_cycles += bank_conflict_blocked_cycles_this_layer
                total_global_bank_conflict_stall_cycles += global_bank_conflict_stall_cycles_this_layer
                bank_stall_report.write(
                    f'{lid},{int(compute_report_items_this_layer[2])},{bank_conflict_stalls_this_layer},{bank_conflict_blocked_cycles_this_layer},{global_bank_conflict_stall_cycles_this_layer}\n'
                )
            
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

            if dump_bank_util_csv:
                layer_bank_stats = single_layer_obj.get_bank_utilization_items()
                if len(layer_bank_stats) > 0:
                    if len(bank_busy_totals) < len(layer_bank_stats):
                        grow = len(layer_bank_stats) - len(bank_busy_totals)
                        bank_busy_totals.extend([0] * grow)
                        bank_access_totals.extend([0] * grow)
                        bank_denom_totals.extend([0] * grow)

                    for item in layer_bank_stats:
                        bank_id = int(item['bank_id'])
                        source_count = int(item['source_count']) if 'source_count' in item else 1
                        layer_cycles = int(compute_report_items_this_layer[1])
                        bank_busy_totals[bank_id] += int(item['busy_cycles'])
                        bank_access_totals[bank_id] += int(item['access_count'])
                        bank_denom_totals[bank_id] += layer_cycles * source_count

            if self.conf.sparsity_support is True:
                sparse_report_items_this_layer = single_layer_obj.get_sparse_report_items()
                log = str(lid) + ', ' + self.conf.sparsity_representation + ', '
                log += ', '.join([str(x) for x in sparse_report_items_this_layer])
                log += ',\n'
                sparse_report.write(log)

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()
        time_report.close()

        if dump_bank_util_csv:
            for bank_id in range(len(bank_busy_totals)):
                busy_cycles = bank_busy_totals[bank_id]
                access_count = bank_access_totals[bank_id]
                if bank_denom_totals[bank_id] > 0:
                    utilization = busy_cycles / float(bank_denom_totals[bank_id])
                else:
                    utilization = 0.0
                bank_report.write(
                    f'{bank_id},{busy_cycles},{access_count},{utilization}\n'
                )

            bank_report.close()
            bank_stall_report.write(
                f'TOTAL,NA,{total_bank_conflict_stalls},{total_bank_conflict_blocked_cycles},{total_global_bank_conflict_stall_cycles}\n'
            )
            bank_stall_report.close()

        if self.conf.sparsity_support is True:
            sparse_report.close()

        self._generate_moe_parallel_reports()

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

