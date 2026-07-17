"""
This file defines the 'scale_config' class responsible for all the configuration file related
activities such as parsing the config file, writing the parameters into a config file, updating the
parameters.
"""
import configparser as cp
import math
from pathlib import Path


class scale_config:
    """
    Class that handles the SCALE-Sim configuration files.
    """
    #
    def __init__(self):
        """
        __init__ method
        """
        self.run_name = "scale_run"
        self.config_dir = Path('.')
        # Anand: ISSUE #2. Patch
        self.use_user_bandwidth = False

        self.array_rows = 4
        self.array_cols = 4
        self.ifmap_sz_kb = 256
        self.filter_sz_kb = 256
        self.ofmap_sz_kb = 128
        self.df = 'ws'
        self.ifmap_offset = 0
        self.filter_offset = 10000000
        self.ofmap_offset = 20000000
        self.req_buf_sz_rd = 60
        self.req_buf_sz_wr = 60
        self.topofile = ""
        self.layoutfile = ""
        self.bandwidths = []
        self.valid_conf_flag = False
        self.num_bank = 1
        self.num_port = 2

        # Layout flags with default values
        self.using_ifmap_custom_layout = False
        self.ifmap_sram_bank_bandwidth = 10
        self.ifmap_sram_bank_num = 10
        self.ifmap_sram_bank_port = 2
        self.using_filter_custom_layout = False
        self.filter_sram_bank_bandwidth = 10
        self.filter_sram_bank_num = 10
        self.filter_sram_bank_port = 2
        self.ofmap_sram_bank_bandwidth = 10
        self.ofmap_sram_bank_num = 1
        self.ofmap_sram_bank_port = 2

        self.valid_df_list = ['os', 'ws', 'is']

        self.sparsity_support = False
        self.sparsity_representation = ""
        # self.sparsity_N = 4
        # self.sparsity_M = 4
        self.sparsity_optimized_mapping = False
        self.sparsity_block_size = 4
        self.sparsity_rand_seed = 40
    
    # Sarbartha: Added ramulator based DRAM trace support
        self.use_ramulator_trace = False

        # Bank-conflict-only memory model controls
        self.enable_bank_model = False
        self.enable_dynamic = False
        self.bank_conflict_penalty = 1
        self.enable_capacity_penalty = True
        self.dram_penalty_scale = 8

        # Prefetch experiment controls (cycle-level abstraction for prefetch-bank co-design)
        # NOTE: This is NOT the original SCALE-Sim read-buffer prefetch model.
        # These knobs enable a lightweight next-layer prefetch experiment on top of the
        # bank-conflict SRAM model.
        self.enable_prefetch = False
        self.prefetch_window = 0
        self.prefetch_target = 'ifmap,filter'
        self.prefetch_policy = 'next_layer'
        self.prefetch_priority = 'low'

        # EP-MoE controls. Disabled by default to preserve legacy SCALE-Sim runs.
        self.enable_ep_moe = False
        self.enable_parallel_moe = True
        self.num_gpus = 2
        self.detailed_gpu_id = 0
        self.blackbox_gpu_ids = ''
        self.experts_per_gpu = 4
        self.compute_engines_per_gpu = 4
        self.top_k = 1
        self.moe_routing_mode = 'topology_counts'
        self.moe_tokens = 0
        self.routing_file = ''
        self.routing_seed = 40
        self.routing_skew_factor = 1.0
        self.moe_active_expert_mode = 'all'
        self.active_expert_ids = ''
        self.enable_chunk_prefetch = False
        self.initial_chunk = 1
        self.chunk_prefetch_window = 0
        self.blackbox_workload_mode = 'analytical'
        self.blackbox_bandwidth_bytes_per_cycle = 128
        self.enable_blackbox_background_pressure = False
        self.global_memory_bandwidth_bytes_per_cycle = 1024
        self.dynamic_bank_overhead = 'old_model'
        self.communication_model = 'latency_plus_bandwidth'
        self.precision_bytes = 2
        self.communication_latency_cycles = 0
        self.communication_bandwidth_bytes_per_cycle = 128
        self.communication_overlap_mode = 'prefetch_only'
        self.allow_comm_prefetch_overlap = True
        
        # Time linear model parameter
        self.time_linear_model = 'None'
    #
    def read_conf_file(self, conf_file_in):
        """
        Method to read the configuration file and extract all the archietctural knobs.
        """

        me = 'scale_config.' + 'read_conf_file()'

        config = cp.ConfigParser()
        config.read(conf_file_in)
        self.config_dir = Path(conf_file_in).resolve().parent

        section = 'general'
        self.run_name = config.get(section, 'run_name')

        # Anand: ISSUE #2. Patch
        section = 'run_presets'
        bw_mode_string = config.get(section, 'InterfaceBandwidth')
        if bw_mode_string == 'USER':
            self.use_user_bandwidth = True
        elif bw_mode_string == 'CALC':
            self.use_user_bandwidth = False
        else:
            message = 'ERROR: ' + me
            message += 'Use either USER or CALC in InterfaceBandwidth feild. Aborting!'
            return
        
        # Parse UseRamulatorTrace if present
        if config.has_option(section, 'UseRamulatorTrace'):
            ramulator_on = config.get(section, 'UseRamulatorTrace')
            if ramulator_on == 'True':
                self.use_ramulator_trace = True
            else:
                self.use_ramulator_trace = False
        
        # Parse TimeLinearModel if present
        if config.has_option(section, 'TimeLinearModel'):
            self.time_linear_model = config.get(section, 'TimeLinearModel')
            assert self.time_linear_model in ['None', 'TPUv4', 'TPUv5e', 'TPUv6e'], f"ERROR: Invalid time linear model '{self.time_linear_model}'. Must be one of: None, TPUv4, TPUv5e, TPUv6e"

        # Optional bank model switches
        if config.has_option(section, 'EnableBankModel'):
            self.enable_bank_model = config.getboolean(section, 'EnableBankModel')
        if config.has_option(section, 'EnableDynamic'):
            self.enable_dynamic = config.getboolean(section, 'EnableDynamic')
        if config.has_option(section, 'BankConflictPenalty'):
            self.bank_conflict_penalty = max(1, config.getint(section, 'BankConflictPenalty'))
        if config.has_option(section, 'EnableCapacityPenalty'):
            self.enable_capacity_penalty = config.getboolean(section, 'EnableCapacityPenalty')
        if config.has_option(section, 'DRAMPenaltyScale'):
            self.dram_penalty_scale = max(1, config.getint(section, 'DRAMPenaltyScale'))

        # Optional prefetch experiment knobs (next-layer prefetch)
        if config.has_option(section, 'EnablePrefetch'):
            self.enable_prefetch = config.getboolean(section, 'EnablePrefetch')
        if config.has_option(section, 'PrefetchWindow'):
            self.prefetch_window = max(0, config.getint(section, 'PrefetchWindow'))
        if config.has_option(section, 'PrefetchTarget'):
            self.prefetch_target = str(config.get(section, 'PrefetchTarget')).strip()
        if config.has_option(section, 'PrefetchPolicy'):
            self.prefetch_policy = str(config.get(section, 'PrefetchPolicy')).strip()
        if config.has_option(section, 'PrefetchPriority'):
            self.prefetch_priority = str(config.get(section, 'PrefetchPriority')).strip()

        # Optional EP-MoE knobs. These are intentionally separate from legacy
        # next-layer prefetch knobs because chunk prefetch has different semantics.
        if config.has_option(section, 'EnableEPMoE'):
            self.enable_ep_moe = config.getboolean(section, 'EnableEPMoE')
        if config.has_option(section, 'EnableParallelMoE'):
            self.enable_parallel_moe = config.getboolean(section, 'EnableParallelMoE')
        if config.has_option(section, 'NumGPUs'):
            self.num_gpus = config.getint(section, 'NumGPUs')
        if config.has_option(section, 'DetailedGPUId'):
            self.detailed_gpu_id = config.getint(section, 'DetailedGPUId')
        if config.has_option(section, 'BlackBoxGPUIds'):
            self.blackbox_gpu_ids = str(config.get(section, 'BlackBoxGPUIds')).strip()
        if config.has_option(section, 'ExpertsPerGPU'):
            self.experts_per_gpu = config.getint(section, 'ExpertsPerGPU')
        if config.has_option(section, 'ComputeEnginesPerGPU'):
            self.compute_engines_per_gpu = config.getint(section, 'ComputeEnginesPerGPU')
        if config.has_option(section, 'TopK'):
            self.top_k = config.getint(section, 'TopK')
        if config.has_option(section, 'MoERoutingMode'):
            self.moe_routing_mode = str(config.get(section, 'MoERoutingMode')).strip()
        if config.has_option(section, 'MoETokens'):
            self.moe_tokens = config.getint(section, 'MoETokens')
        if config.has_option(section, 'RoutingFile'):
            self.routing_file = str(config.get(section, 'RoutingFile')).strip()
        if config.has_option(section, 'RoutingSeed'):
            self.routing_seed = config.getint(section, 'RoutingSeed')
        if config.has_option(section, 'RoutingSkewFactor'):
            self.routing_skew_factor = config.getfloat(section, 'RoutingSkewFactor')
        if config.has_option(section, 'MoEActiveExpertMode'):
            self.moe_active_expert_mode = str(config.get(section, 'MoEActiveExpertMode')).strip()
        if config.has_option(section, 'ActiveExpertIds'):
            self.active_expert_ids = str(config.get(section, 'ActiveExpertIds')).strip()
        if config.has_option(section, 'EnableChunkPrefetch'):
            self.enable_chunk_prefetch = config.getboolean(section, 'EnableChunkPrefetch')
        if config.has_option(section, 'InitialChunk'):
            self.initial_chunk = config.getint(section, 'InitialChunk')
        if config.has_option(section, 'ChunkPrefetchWindow'):
            self.chunk_prefetch_window = config.getint(section, 'ChunkPrefetchWindow')
        if config.has_option(section, 'BlackBoxWorkloadMode'):
            self.blackbox_workload_mode = str(config.get(section, 'BlackBoxWorkloadMode')).strip()
        if config.has_option(section, 'BlackBoxBandwidthBytesPerCycle'):
            self.blackbox_bandwidth_bytes_per_cycle = config.getint(section, 'BlackBoxBandwidthBytesPerCycle')
        if config.has_option(section, 'EnableBlackBoxBackgroundPressure'):
            self.enable_blackbox_background_pressure = config.getboolean(section, 'EnableBlackBoxBackgroundPressure')
        if config.has_option(section, 'GlobalMemoryBandwidthBytesPerCycle'):
            self.global_memory_bandwidth_bytes_per_cycle = config.getint(section, 'GlobalMemoryBandwidthBytesPerCycle')
        if config.has_option(section, 'DynamicBankOverhead'):
            self.dynamic_bank_overhead = str(config.get(section, 'DynamicBankOverhead')).strip()
        if config.has_option(section, 'CommunicationModel'):
            self.communication_model = str(config.get(section, 'CommunicationModel')).strip()
        if config.has_option(section, 'PrecisionBytes'):
            self.precision_bytes = config.getint(section, 'PrecisionBytes')
        if config.has_option(section, 'CommunicationLatencyCycles'):
            self.communication_latency_cycles = config.getint(section, 'CommunicationLatencyCycles')
        if config.has_option(section, 'CommunicationBandwidthBytesPerCycle'):
            self.communication_bandwidth_bytes_per_cycle = config.getint(section, 'CommunicationBandwidthBytesPerCycle')
        if config.has_option(section, 'CommunicationOverlapMode'):
            self.communication_overlap_mode = str(config.get(section, 'CommunicationOverlapMode')).strip()
        if config.has_option(section, 'AllowCommPrefetchOverlap'):
            self.allow_comm_prefetch_overlap = config.getboolean(section, 'AllowCommPrefetchOverlap')
        elif config.has_option(section, 'EnableCommunicationOverlap'):
            # Backward-compatible alias used by the initial EP scaffold.
            self.allow_comm_prefetch_overlap = config.getboolean(section, 'EnableCommunicationOverlap')


        # TODO Sarbartha: Should be bw
        div_factor = 1
        
        section = 'architecture_presets'
        self.array_rows = int(config.get(section, 'ArrayHeight'))
        self.array_cols = int(config.get(section, 'ArrayWidth'))
        self.ifmap_sz_kb = int(config.get(section, 'ifmapsramszkB'))
        self.filter_sz_kb = int(config.get(section, 'filtersramszkB'))
        self.ofmap_sz_kb = int(config.get(section, 'ofmapsramszkB'))
        self.ifmap_offset = int(config.get(section, 'IfmapOffset'))
        self.filter_offset = int(config.get(section, 'FilterOffset'))
        self.ofmap_offset = int(config.get(section, 'OfmapOffset'))
        self.df = config.get(section, 'Dataflow')
        
        # Make ReadRequestBuffer and WriteRequestBuffer optional
        if config.has_option(section, 'ReadRequestBuffer'):
            self.req_buf_sz_rd = int(config.get(section, 'ReadRequestBuffer')) // div_factor
        if config.has_option(section, 'WriteRequestBuffer'):
            self.req_buf_sz_wr = int(config.get(section, 'WriteRequestBuffer')) // div_factor

        layout_section = 'layout'
        if config.has_section(layout_section):
            self.using_ifmap_custom_layout = config.getboolean(layout_section, 'IfmapCustomLayout')
            self.using_filter_custom_layout = config.getboolean(layout_section, 'FilterCustomLayout')
            self.ifmap_sram_bank_bandwidth = int(config.get(layout_section, 'IfmapSRAMBankBandwidth'))
            self.ifmap_sram_bank_num = int(config.get(layout_section, 'IfmapSRAMBankNum'))
            self.ifmap_sram_bank_port = int(config.get(layout_section, 'IfmapSRAMBankPort'))
            self.filter_sram_bank_bandwidth = int(config.get(layout_section, 'FilterSRAMBankBandwidth'))
            self.filter_sram_bank_num = int(config.get(layout_section, 'FilterSRAMBankNum'))
            self.filter_sram_bank_port = int(config.get(layout_section, 'FilterSRAMBankPort'))
            if config.has_option(layout_section, 'OfmapSRAMBankBandwidth'):
                self.ofmap_sram_bank_bandwidth = int(config.get(layout_section, 'OfmapSRAMBankBandwidth'))
            else:
                self.ofmap_sram_bank_bandwidth = self.filter_sram_bank_bandwidth
            if config.has_option(layout_section, 'OfmapSRAMBankNum'):
                self.ofmap_sram_bank_num = int(config.get(layout_section, 'OfmapSRAMBankNum'))
            else:
                self.ofmap_sram_bank_num = 1
            if config.has_option(layout_section, 'OfmapSRAMBankPort'):
                self.ofmap_sram_bank_port = int(config.get(layout_section, 'OfmapSRAMBankPort'))
            else:
                self.ofmap_sram_bank_port = self.filter_sram_bank_port
        else:
            # Original SCALE-Sim configs predate the layout section. Preserve
            # their aggregate bank defaults while providing a complete internal
            # representation for config validation and round-tripping.
            legacy_bank_num = config.getint(section, 'OnChipMemoryBanks', fallback=1)
            legacy_bank_port = config.getint(section, 'OnChipMemoryBankPorts', fallback=2)
            legacy_bank_bw = config.getint(section, 'Bandwidth', fallback=10)
            self.using_ifmap_custom_layout = False
            self.using_filter_custom_layout = False
            self.ifmap_sram_bank_bandwidth = legacy_bank_bw
            self.filter_sram_bank_bandwidth = legacy_bank_bw
            self.ofmap_sram_bank_bandwidth = legacy_bank_bw
            self.ifmap_sram_bank_num = legacy_bank_num
            self.filter_sram_bank_num = legacy_bank_num
            self.ofmap_sram_bank_num = legacy_bank_num
            self.ifmap_sram_bank_port = legacy_bank_port
            self.filter_sram_bank_port = legacy_bank_port
            self.ofmap_sram_bank_port = legacy_bank_port
        
        # Anand: ISSUE #2. Patch
        if self.use_user_bandwidth:
            self.bandwidths = [int(x.strip())
                               for x in config.get(section, 'Bandwidth').strip().split(',')]

        if self.df not in self.valid_df_list:
            print("WARNING: Invalid dataflow")

        if config.has_section('network_presets'):  # Read network_presets
            self.topofile = config.get('network_presets', 'TopologyCsvLoc').strip().strip('"')

        # Sparsity - make this section optional
        if config.has_section('sparsity'):
            section = 'sparsity'
            if config.get(section, 'SparsitySupport').lower() == 'true':
                self.sparsity_support = True
            else:
                self.sparsity_support = False

            if self.sparsity_support:
                self.sparsity_representation = config.get(section, 'SparseRep')
                # self.sparsity_N = int(config.get(section, 'NonZeroElems'))
                # self.sparsity_M = int(config.get(section, 'BlockSize'))
                if config.get(section, 'OptimizedMapping').lower() == 'true':
                    self.sparsity_optimized_mapping = True
                else:
                    self.sparsity_optimized_mapping = False

                if self.sparsity_optimized_mapping:
                    self.sparsity_block_size = int(config.get(section, 'BlockSize'))
                    assert self.sparsity_block_size <= self.array_rows, "ERROR: Invalid block size"

                self.sparsity_rand_seed = int(config.get(section, 'RandomNumberGeneratorSeed'))

        self.valid_conf_flag = True

        positive_common_values = {
            'ArrayHeight': self.array_rows,
            'ArrayWidth': self.array_cols,
            'IfmapSRAMSzkB': self.ifmap_sz_kb,
            'FilterSRAMSzkB': self.filter_sz_kb,
            'OfmapSRAMSzkB': self.ofmap_sz_kb,
            'IfmapSRAMBankBandwidth': self.ifmap_sram_bank_bandwidth,
            'IfmapSRAMBankNum': self.ifmap_sram_bank_num,
            'IfmapSRAMBankPort': self.ifmap_sram_bank_port,
            'FilterSRAMBankBandwidth': self.filter_sram_bank_bandwidth,
            'FilterSRAMBankNum': self.filter_sram_bank_num,
            'FilterSRAMBankPort': self.filter_sram_bank_port,
            'OfmapSRAMBankBandwidth': self.ofmap_sram_bank_bandwidth,
            'OfmapSRAMBankNum': self.ofmap_sram_bank_num,
            'OfmapSRAMBankPort': self.ofmap_sram_bank_port,
        }
        invalid_common = [name for name, value in positive_common_values.items() if int(value) <= 0]
        if invalid_common:
            raise ValueError('ERROR: Configuration values must be positive: ' + ', '.join(invalid_common))
        if self.use_user_bandwidth and (not self.bandwidths or any(int(bw) <= 0 for bw in self.bandwidths)):
            raise ValueError('ERROR: USER InterfaceBandwidth requires positive Bandwidth values')

        # Lightweight validation for prefetch config
        if self.enable_prefetch:
            pol = str(self.prefetch_policy).lower().strip()
            prio = str(self.prefetch_priority).lower().strip()
            if pol not in ['next_layer']:
                raise ValueError(f"ERROR: Unsupported PrefetchPolicy '{self.prefetch_policy}'. Only 'next_layer' is supported.")
            if prio not in ['low']:
                raise ValueError(f"ERROR: Unsupported PrefetchPriority '{self.prefetch_priority}'. Only 'low' is supported.")

        if self.enable_ep_moe:
            if self.num_gpus <= 0:
                raise ValueError("ERROR: NumGPUs must be positive")
            if self.detailed_gpu_id < 0:
                raise ValueError("ERROR: DetailedGPUId must be non-negative")
            if self.detailed_gpu_id >= self.num_gpus:
                raise ValueError("ERROR: DetailedGPUId must be smaller than NumGPUs")
            if self.experts_per_gpu <= 0:
                raise ValueError("ERROR: ExpertsPerGPU must be positive")
            if self.compute_engines_per_gpu <= 0:
                raise ValueError("ERROR: ComputeEnginesPerGPU must be positive")
            if self.initial_chunk <= 0:
                raise ValueError("ERROR: InitialChunk must be positive")
            if self.chunk_prefetch_window < 0:
                raise ValueError("ERROR: ChunkPrefetchWindow must be non-negative")
            if self.precision_bytes <= 0:
                raise ValueError("ERROR: PrecisionBytes must be positive")
            if self.blackbox_bandwidth_bytes_per_cycle <= 0:
                raise ValueError("ERROR: BlackBoxBandwidthBytesPerCycle must be positive")
            if self.global_memory_bandwidth_bytes_per_cycle <= 0:
                raise ValueError("ERROR: GlobalMemoryBandwidthBytesPerCycle must be positive")
            if self.communication_latency_cycles < 0:
                raise ValueError("ERROR: CommunicationLatencyCycles must be non-negative")
            if self.communication_bandwidth_bytes_per_cycle <= 0:
                raise ValueError("ERROR: CommunicationBandwidthBytesPerCycle must be positive")
            if self.top_k not in [1, 2]:
                raise ValueError("ERROR: TopK currently supports only 1 or 2")
            if self.top_k > self.get_num_experts():
                raise ValueError("ERROR: TopK cannot exceed NumExperts")
            routing_mode = self.get_moe_routing_mode()
            if routing_mode not in ['topology_counts', 'balanced', 'explicit', 'seeded_skewed']:
                raise ValueError(
                    "ERROR: MoERoutingMode supports topology_counts, balanced, explicit, or seeded_skewed"
                )
            if routing_mode in ['balanced', 'seeded_skewed'] and self.moe_tokens <= 0:
                raise ValueError(f"ERROR: MoETokens must be positive for MoERoutingMode={routing_mode}")
            if routing_mode == 'explicit' and not self.routing_file:
                raise ValueError("ERROR: RoutingFile is required for MoERoutingMode=explicit")
            if self.moe_tokens < 0:
                raise ValueError("ERROR: MoETokens must be non-negative")
            if self.routing_skew_factor <= 0:
                raise ValueError("ERROR: RoutingSkewFactor must be positive")
            blackbox_gpu_ids = self.get_blackbox_gpu_ids()
            expected_blackbox_ids = [gpu_id for gpu_id in range(self.num_gpus) if gpu_id != self.detailed_gpu_id]
            if len(blackbox_gpu_ids) != len(set(blackbox_gpu_ids)):
                raise ValueError("ERROR: BlackBoxGPUIds must not contain duplicates")
            if sorted(blackbox_gpu_ids) != expected_blackbox_ids:
                raise ValueError(
                    "ERROR: BlackBoxGPUIds must contain every non-detailed GPU exactly once; "
                    f"expected {expected_blackbox_ids}, got {blackbox_gpu_ids}"
                )
            active_expert_ids = self.get_active_expert_ids()
            if len(active_expert_ids) != len(set(active_expert_ids)):
                raise ValueError("ERROR: ActiveExpertIds must not contain duplicates")
            invalid_expert_ids = [eid for eid in active_expert_ids if eid < 0 or eid >= self.get_num_experts()]
            if invalid_expert_ids:
                raise ValueError(
                    f"ERROR: ActiveExpertIds out of range [0, {self.get_num_experts() - 1}]: {invalid_expert_ids}"
                )
            if str(self.moe_active_expert_mode).lower().strip() != 'all':
                raise ValueError(
                    "ERROR: MoEActiveExpertMode=topk_prefix is removed; active experts are derived from routing"
                )
            if active_expert_ids:
                raise ValueError(
                    "ERROR: ActiveExpertIds is replaced by MoERoutingMode/RoutingFile in EP-MoE mode"
                )
            if str(self.blackbox_workload_mode).lower().strip() not in ['analytical']:
                raise ValueError("ERROR: BlackBoxWorkloadMode currently supports only 'analytical'")
            if str(self.dynamic_bank_overhead).lower().strip() not in ['old_model']:
                raise ValueError("ERROR: DynamicBankOverhead currently supports only 'old_model'")
            if str(self.communication_model).lower().strip() not in ['latency_plus_bandwidth']:
                raise ValueError("ERROR: CommunicationModel currently supports only 'latency_plus_bandwidth'")
            if str(self.communication_overlap_mode).lower().strip() not in ['none', 'prefetch_only', 'full']:
                raise ValueError("ERROR: CommunicationOverlapMode supports only 'none', 'prefetch_only', or 'full'")
            if self.enable_prefetch and self.enable_chunk_prefetch:
                raise ValueError(
                    "ERROR: legacy EnablePrefetch and EP EnableChunkPrefetch cannot be enabled together"
                )
            if self.enable_chunk_prefetch and self.chunk_prefetch_window <= 0:
                raise ValueError("ERROR: EnableChunkPrefetch=True requires ChunkPrefetchWindow > 0")

    #
    def update_from_list(self, conf_list):
        """
        Method to update the parameters through a configuration list.
        """
        if not len(conf_list) > 11:
            print("ERROR: scale_config.update_from_list: "
                  "Incompatible number of elements in the list")

        self.run_name = conf_list[0]
        self.array_rows = int(conf_list[1])
        self.array_cols = int(conf_list[2])
        self.ifmap_sz_kb = int(conf_list[3])
        self.filter_sz_kb = int(conf_list[4])
        self.ofmap_sz_kb = int(conf_list[5])
        self.ifmap_offset = int(conf_list[6])
        self.filter_offset = int(conf_list[7])
        self.ofmap_offset = int(conf_list[8])
        self.df = conf_list[9]
        bw_mode_string = str(conf_list[10])

        assert bw_mode_string in ['CALC', 'USER'], 'Invalid mode of operation'
        if bw_mode_string == "USER":
            assert not len(conf_list) < 12, 'The user bandwidth needs to be provided'
            self.bandwidths = conf_list[11]
            self.use_user_bandwidth = True
        elif bw_mode_string == 'CALC':
            self.use_user_bandwidth = False

        if len(conf_list) == 15:
            self.topofile = conf_list[14]

        self.valid_conf_flag = True

    #
    def write_conf_file(self, conf_file_out):
        """
        Method to generate a configuration file.
        """
        if not self.valid_conf_flag:
            print('ERROR: scale_config.write_conf_file: No valid config loaded')
            return

        config = cp.ConfigParser()

        section = 'general'
        config.add_section(section)
        config.set(section, 'run_name', str(self.run_name))

        section = 'architecture_presets'
        config.add_section(section)
        config.set(section, 'ArrayHeight', str(self.array_rows))
        config.set(section, 'ArrayWidth', str(self.array_cols))

        config.set(section, 'ifmapsramszkB', str(self.ifmap_sz_kb))
        config.set(section, 'filtersramszkB', str(self.filter_sz_kb))
        config.set(section, 'ofmapsramszkB', str(self.ofmap_sz_kb))

        config.set(section, 'IfmapOffset', str(self.ifmap_offset))
        config.set(section, 'FilterOffset', str(self.filter_offset))
        config.set(section, 'OfmapOffset', str(self.ofmap_offset))

        config.set(section, 'Dataflow', str(self.df))
        config.set(section, 'Bandwidth', ','.join([str(x) for x in self.bandwidths]))
        config.set(section, 'ReadRequestBuffer', str(self.req_buf_sz_rd))
        config.set(section, 'WriteRequestBuffer', str(self.req_buf_sz_wr))

        section = 'layout'
        config.add_section(section)
        config.set(section, 'IfmapCustomLayout', str(self.using_ifmap_custom_layout))
        config.set(section, 'IfmapSRAMBankBandwidth', str(self.ifmap_sram_bank_bandwidth))
        config.set(section, 'IfmapSRAMBankNum', str(self.ifmap_sram_bank_num))
        config.set(section, 'IfmapSRAMBankPort', str(self.ifmap_sram_bank_port))
        config.set(section, 'FilterCustomLayout', str(self.using_filter_custom_layout))
        config.set(section, 'FilterSRAMBankBandwidth', str(self.filter_sram_bank_bandwidth))
        config.set(section, 'FilterSRAMBankNum', str(self.filter_sram_bank_num))
        config.set(section, 'FilterSRAMBankPort', str(self.filter_sram_bank_port))
        config.set(section, 'OfmapSRAMBankBandwidth', str(self.ofmap_sram_bank_bandwidth))
        config.set(section, 'OfmapSRAMBankNum', str(self.ofmap_sram_bank_num))
        config.set(section, 'OfmapSRAMBankPort', str(self.ofmap_sram_bank_port))

        section = 'sparsity'
        config.add_section(section)
        config.set(section, 'SparsitySupport', str(self.sparsity_support))
        config.set(section, 'SparseRep', str(self.sparsity_representation or 'ellpack_block'))
        config.set(section, 'OptimizedMapping', str(self.sparsity_optimized_mapping))
        config.set(section, 'BlockSize', str(self.sparsity_block_size))
        config.set(section, 'RandomNumberGeneratorSeed', str(self.sparsity_rand_seed))

        section = 'network_presets'
        config.add_section(section)
        topofile = '"' + self.topofile + '"'
        config.set(section, 'TopologyCsvLoc', str(topofile))
        
        section = 'run_presets'
        config.add_section(section)
        bw_mode = 'USER' if self.use_user_bandwidth else 'CALC'
        config.set(section, 'InterfaceBandwidth', str(bw_mode))
        config.set(section, 'UseRamulatorTrace', str(self.use_ramulator_trace))
        config.set(section, 'TimeLinearModel', str(self.time_linear_model))
        config.set(section, 'EnableBankModel', str(self.enable_bank_model))
        config.set(section, 'EnableDynamic', str(self.enable_dynamic))
        config.set(section, 'BankConflictPenalty', str(self.bank_conflict_penalty))
        config.set(section, 'EnableCapacityPenalty', str(self.enable_capacity_penalty))
        config.set(section, 'DRAMPenaltyScale', str(self.dram_penalty_scale))
        config.set(section, 'EnableEPMoE', str(self.enable_ep_moe))
        config.set(section, 'EnableParallelMoE', str(self.enable_parallel_moe))
        config.set(section, 'NumGPUs', str(self.num_gpus))
        config.set(section, 'DetailedGPUId', str(self.detailed_gpu_id))
        config.set(section, 'BlackBoxGPUIds', ','.join(str(x) for x in self.get_blackbox_gpu_ids()))
        config.set(section, 'ExpertsPerGPU', str(self.experts_per_gpu))
        config.set(section, 'ComputeEnginesPerGPU', str(self.compute_engines_per_gpu))
        config.set(section, 'TopK', str(self.top_k))
        config.set(section, 'MoERoutingMode', str(self.moe_routing_mode))
        config.set(section, 'MoETokens', str(self.moe_tokens))
        config.set(section, 'RoutingFile', str(self.get_routing_file()))
        config.set(section, 'RoutingSeed', str(self.routing_seed))
        config.set(section, 'RoutingSkewFactor', str(self.routing_skew_factor))
        config.set(section, 'MoEActiveExpertMode', str(self.moe_active_expert_mode))
        config.set(section, 'ActiveExpertIds', str(self.active_expert_ids))
        config.set(section, 'EnableChunkPrefetch', str(self.enable_chunk_prefetch))
        config.set(section, 'InitialChunk', str(self.initial_chunk))
        config.set(section, 'ChunkPrefetchWindow', str(self.chunk_prefetch_window))
        config.set(section, 'BlackBoxWorkloadMode', str(self.blackbox_workload_mode))
        config.set(section, 'BlackBoxBandwidthBytesPerCycle', str(self.blackbox_bandwidth_bytes_per_cycle))
        config.set(section, 'EnableBlackBoxBackgroundPressure', str(self.enable_blackbox_background_pressure))
        config.set(section, 'GlobalMemoryBandwidthBytesPerCycle', str(self.global_memory_bandwidth_bytes_per_cycle))
        config.set(section, 'DynamicBankOverhead', str(self.dynamic_bank_overhead))
        config.set(section, 'CommunicationModel', str(self.communication_model))
        config.set(section, 'PrecisionBytes', str(self.precision_bytes))
        config.set(section, 'CommunicationLatencyCycles', str(self.communication_latency_cycles))
        config.set(section, 'CommunicationBandwidthBytesPerCycle', str(self.communication_bandwidth_bytes_per_cycle))
        config.set(section, 'CommunicationOverlapMode', str(self.communication_overlap_mode))
        config.set(section, 'AllowCommPrefetchOverlap', str(self.allow_comm_prefetch_overlap))

        with open(conf_file_out, 'w') as configfile:
            config.write(configfile)

    #
    def set_arr_dims(self, rows=1, cols=1):
        """
        Method to set the dimensions of the PE array, with default dimensions set to 1x1.
        """
        self.array_rows = rows
        self.array_cols = cols

    #
    def set_dataflow(self, dataflow='os'):
        """
        Method to set the dataflow for the matric multiplication with Output Stationary being the
        default dataflow.
        """
        self.df = dataflow

    #
    def set_buffer_sizes_kb(self, ifmap_size_kb=1, filter_size_kb=1, ofmap_size_kb=1):
        """
        Method to set the IFMAP, Filter and OFMAP SRAM sizes, with the defaults set to 1kB.
        """
        self.ifmap_sz_kb = ifmap_size_kb
        self.filter_sz_kb = filter_size_kb
        self.ofmap_sz_kb = ofmap_size_kb

    #
    def set_topology_file(self, topofile=''):
        """
        Method to set the topology file path.
        """
        self.topofile = topofile
    
    #
    def set_layout_file(self, layoutfile=''):
        self.layoutfile = layoutfile

    #
    def set_offsets(self,
                    ifmap_offset=0,
                    filter_offset=10000000,
                    ofmap_offset=20000000
                    ):
        """
        Method to set the offsets used for IFMAP, Filter and OFMAP addresses, with the defaults set
        to 0, 10M and 20M respectively.
        """
        self.ifmap_offset = ifmap_offset
        self.filter_offset = filter_offset
        self.ifmap_offset = ofmap_offset
        self.valid_conf_flag = True

    #
    def force_valid(self):
        """
        Method to set the 'valid_config_flag' without any checks.
        """
        self.valid_conf_flag = True

    #
    def set_bw_mode_to_calc(self):
        """
        Method to set the 'use_user_bandwidth' to CALC mode.
        """
        self.use_user_bandwidth = False

    #
    def use_user_dram_bandwidth(self):
        """
        Method that returns the value of 'use_user_bandwidth'.
        """
        if not self.valid_conf_flag:
            me = 'scale_config.' + 'use_user_dram_bandwidth()'
            message = 'ERROR: ' + me + ': Configuration is not valid'
            print(message)
            return

        return self.use_user_bandwidth

    #
    def get_conf_as_list(self):
        """
        Method to extract the configuration parameters in the form of a list.
        """
        out_list = []

        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_conf_as_list: Configuration is not valid")
            return

        out_list.append(str(self.run_name))

        out_list.append(str(self.array_rows))
        out_list.append(str(self.array_cols))

        out_list.append(str(self.ifmap_sz_kb))
        out_list.append(str(self.filter_sz_kb))
        out_list.append(str(self.ofmap_sz_kb))

        out_list.append(str(self.ifmap_offset))
        out_list.append(str(self.filter_offset))
        out_list.append(str(self.ofmap_offset))

        out_list.append(str(self.df))
        out_list.append(str(self.topofile))

        return out_list

    #
    def get_run_name(self):
        """
        Method to get the run name used for the simulation.
        """
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_run_name() : Config data is not valid")
            return

        return self.run_name

    #
    def get_topology_path(self):
        """
        Method to get the topology file path used for the simulation.
        """
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_topology_path() : Config data is not valid")
            return
        return self.topofile

    def get_layout_path(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_layout_path() : Config data is not valid")
            return
        return self.layoutfile

    def get_topology_name(self):
        """
        Method to extract the name of the topology file from the topology path.
        """
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_topology_name() : Config data is not valid")
            return

        name = self.topofile.split('/')[-1].strip()
        name = name.split('.')[0]

        return name

    #
    def get_dataflow(self):
        """
        Method to get the dataflow used for the simulation.
        """
        if self.valid_conf_flag:
            return self.df

    #
    def get_array_dims(self):
        """
        Method to get the dimensions of the PE array.
        """
        if self.valid_conf_flag:
            return self.array_rows, self.array_cols

    #
    def get_mem_sizes(self):
        """
        Method to get the IFMAP, Filter and OFMAP SRAM sizes.
        """
        me = 'scale_config.' + 'get_mem_sizes()'

        if not self.valid_conf_flag:
            message = 'ERROR: ' + me
            message += 'Config is not valid. Not returning any values'
            return

        return self.ifmap_sz_kb, self.filter_sz_kb, self.ofmap_sz_kb

    #
    def get_offsets(self):
        """
        Method to get the offsets used for IFMAP, Filter and OFMAP addresses.
        """
        if self.valid_conf_flag:
            return self.ifmap_offset, self.filter_offset, self.ofmap_offset
    
    def get_ramulator_trace(self):
        """
        Method to check if the run considers ramulator trace numpy files
        """
        if self.valid_conf_flag:
            return self.use_ramulator_trace
    
    def get_req_buf_sz_rd(self):
        """
        Method to set the read request buffer size
        """
        if self.valid_conf_flag:
            return self.req_buf_sz_rd
    
    def get_req_buf_sz_wr(self):
        """
        Method to set the write request buffer size
        """
        if self.valid_conf_flag:
            return self.req_buf_sz_wr
    
    #
    def get_bandwidths_as_string(self):
        """
        Method to get the bandwidths as a string.
        """
        if self.valid_conf_flag:
            return ','.join([str(x) for x in self.bandwidths])

    #
    def get_ifmap_sram_bandwidth(self):
        """
        Method to get the IFMAP SRAM bank bandwidth in bits/cycle.
        """
        if self.valid_conf_flag:
            return self.ifmap_sram_bank_bandwidth

    def get_ifmap_sram_bandwidth_bytes(self):
        """
        Method to get the IFMAP SRAM bank bandwidth in bytes/cycle.
        """
        if self.valid_conf_flag:
            return max(1, int(math.ceil(float(self.ifmap_sram_bank_bandwidth) / 8.0)))

    def get_filter_sram_bandwidth(self):
        """
        Method to get the FILTER SRAM bank bandwidth in bits/cycle.
        """
        if self.valid_conf_flag:
            return self.filter_sram_bank_bandwidth

    def get_filter_sram_bandwidth_bytes(self):
        """
        Method to get the FILTER SRAM bank bandwidth in bytes/cycle.
        """
        if self.valid_conf_flag:
            return max(1, int(math.ceil(float(self.filter_sram_bank_bandwidth) / 8.0)))

    def get_ofmap_sram_bandwidth(self):
        """
        Method to get the OFMAP SRAM bank bandwidth in bits/cycle.
        """
        if self.valid_conf_flag:
            return self.ofmap_sram_bank_bandwidth

    def get_ofmap_sram_bandwidth_bytes(self):
        """
        Method to get the OFMAP SRAM bank bandwidth in bytes/cycle.
        """
        if self.valid_conf_flag:
            return max(1, int(math.ceil(float(self.ofmap_sram_bank_bandwidth) / 8.0)))

    #
    def get_bandwidths_as_list(self):
        """
        Method to get the bandwidths as a list.
        """
        if self.valid_conf_flag:
            return self.bandwidths
        
    def get_num_bank(self):
        if self.valid_conf_flag:
            return self.num_bank
        
    def get_num_port(self):
        if self.valid_conf_flag:
            return self.num_port
        
    def get_min_dram_bandwidth(self):
        """
        Method to get the minimum DRAM bandwidth defined in the configuration.
        """
        if not self.use_user_dram_bandwidth():
            me = 'scale_config.' + 'get_min_dram_bandwidth()'
            message = 'ERROR: ' + me + ': No user bandwidth provided'
            print(message)
        else:
            return min(self.bandwidths)

    def get_time_linear_model(self):
        """
        Method to get the time linear model used for the simulation.
        """
        if self.valid_conf_flag:
            return self.time_linear_model
        return "Default"

    def get_enable_bank_model(self):
        """
        Method to check if the pure bank-conflict memory model is enabled.
        """
        if self.valid_conf_flag:
            return self.enable_bank_model
        return False

    def get_enable_prefetch(self):
        """Enable lightweight next-layer prefetch experiment."""
        if self.valid_conf_flag:
            return bool(self.enable_prefetch)
        return False

    def get_prefetch_window(self):
        """Prefetch lookahead window (0 disables prefetch effects)."""
        if self.valid_conf_flag:
            return max(0, int(self.prefetch_window))
        return 0

    def get_prefetch_policy(self):
        if self.valid_conf_flag:
            return str(self.prefetch_policy)
        return 'next_layer'

    def get_prefetch_priority(self):
        if self.valid_conf_flag:
            return str(self.prefetch_priority)
        return 'low'

    def get_prefetch_target(self):
        """Raw prefetch target string (e.g., 'ifmap,filter')."""
        if self.valid_conf_flag:
            return str(self.prefetch_target)
        return 'ifmap,filter'

    def get_prefetch_target_list(self):
        """Prefetch targets as a normalized list. Currently supports ifmap/filter only."""
        raw = str(self.get_prefetch_target()).lower().strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        allowed = {'ifmap', 'filter'}
        out = []
        for p in parts:
            if p not in allowed:
                raise ValueError(f"ERROR: Unsupported PrefetchTarget '{p}'. Only ifmap/filter are supported.")
            if p not in out:
                out.append(p)
        return out

    def get_enable_ep_moe(self):
        if self.valid_conf_flag:
            return bool(self.enable_ep_moe)
        return False

    def get_enable_parallel_moe(self):
        if self.valid_conf_flag:
            return bool(self.enable_parallel_moe)
        return True

    def get_num_gpus(self):
        if self.valid_conf_flag:
            return max(1, int(self.num_gpus))
        return 1

    def get_detailed_gpu_id(self):
        if self.valid_conf_flag:
            return max(0, int(self.detailed_gpu_id))
        return 0

    def get_blackbox_gpu_ids(self):
        if not self.valid_conf_flag:
            return []
        raw = str(self.blackbox_gpu_ids).strip()
        if raw == '' or raw.lower() == 'auto':
            return [gpu_id for gpu_id in range(self.get_num_gpus()) if gpu_id != self.get_detailed_gpu_id()]
        if raw.startswith('[') and raw.endswith(']'):
            raw = raw[1:-1].strip()
        gpu_ids = []
        for item in raw.split(','):
            item = item.strip()
            if item:
                gpu_ids.append(int(item))
        return gpu_ids

    def get_experts_per_gpu(self):
        if self.valid_conf_flag:
            return max(1, int(self.experts_per_gpu))
        return 1

    def get_compute_engines_per_gpu(self):
        if self.valid_conf_flag:
            return max(1, int(self.compute_engines_per_gpu))
        return 1

    def get_num_experts(self):
        return int(self.get_num_gpus() * self.get_experts_per_gpu())

    def get_top_k(self):
        if self.valid_conf_flag:
            return max(1, int(self.top_k))
        return 1

    def get_moe_routing_mode(self):
        if self.valid_conf_flag:
            return str(self.moe_routing_mode).lower().strip()
        return 'topology_counts'

    def get_moe_tokens(self):
        if self.valid_conf_flag:
            return max(0, int(self.moe_tokens))
        return 0

    def get_routing_file(self):
        if not self.valid_conf_flag or not str(self.routing_file).strip():
            return ''
        path = Path(str(self.routing_file).strip()).expanduser()
        if not path.is_absolute():
            path = self.config_dir / path
        return str(path.resolve())

    def get_routing_seed(self):
        if self.valid_conf_flag:
            return int(self.routing_seed)
        return 40

    def get_routing_skew_factor(self):
        if self.valid_conf_flag:
            return float(self.routing_skew_factor)
        return 1.0

    def get_moe_active_expert_mode(self):
        if self.valid_conf_flag:
            return str(self.moe_active_expert_mode).lower().strip()
        return 'all'

    def get_active_expert_ids(self):
        if not self.valid_conf_flag:
            return []
        raw = str(self.active_expert_ids).strip()
        if raw == '' or raw.lower() == 'all':
            return []
        if raw.startswith('[') and raw.endswith(']'):
            raw = raw[1:-1].strip()
        expert_ids = []
        for item in raw.split(','):
            item = item.strip()
            if item == '':
                continue
            expert_ids.append(int(item))
        return expert_ids

    def get_initial_chunk(self):
        if self.valid_conf_flag:
            return max(1, int(self.initial_chunk))
        return 1

    def get_enable_chunk_prefetch(self):
        if self.valid_conf_flag:
            return bool(self.enable_chunk_prefetch)
        return False

    def get_chunk_prefetch_window(self):
        if self.valid_conf_flag and self.get_enable_chunk_prefetch():
            return max(0, int(self.chunk_prefetch_window))
        return 0

    def get_blackbox_workload_mode(self):
        if self.valid_conf_flag:
            return str(self.blackbox_workload_mode)
        return 'analytical'

    def get_blackbox_bandwidth_bytes_per_cycle(self):
        if self.valid_conf_flag:
            return max(1, int(self.blackbox_bandwidth_bytes_per_cycle))
        return 128

    def get_enable_blackbox_background_pressure(self):
        if self.valid_conf_flag:
            return bool(self.enable_blackbox_background_pressure)
        return False

    def get_global_memory_bandwidth_bytes_per_cycle(self):
        if self.valid_conf_flag:
            return max(1, int(self.global_memory_bandwidth_bytes_per_cycle))
        return 1024

    def get_dynamic_bank_overhead(self):
        if self.valid_conf_flag:
            return str(self.dynamic_bank_overhead)
        return 'old_model'

    def get_communication_model(self):
        if self.valid_conf_flag:
            return str(self.communication_model)
        return 'latency_plus_bandwidth'

    def get_precision_bytes(self):
        if self.valid_conf_flag:
            return max(1, int(self.precision_bytes))
        return 2

    def get_communication_latency_cycles(self):
        if self.valid_conf_flag:
            return max(0, int(self.communication_latency_cycles))
        return 0

    def get_communication_bandwidth_bytes_per_cycle(self):
        if self.valid_conf_flag:
            return max(1, int(self.communication_bandwidth_bytes_per_cycle))
        return 128

    def get_communication_input_bytes_per_elem(self):
        """Backward-compatible alias for the unified precision setting."""
        return self.get_precision_bytes()

    def get_communication_output_bytes_per_elem(self):
        """Backward-compatible alias for the unified precision setting."""
        return self.get_precision_bytes()

    def get_communication_overlap_mode(self):
        if self.valid_conf_flag:
            return str(self.communication_overlap_mode).lower().strip()
        return 'prefetch_only'

    def get_enable_communication_overlap(self):
        """Backward-compatible alias for AllowCommPrefetchOverlap."""
        return self.get_allow_comm_prefetch_overlap()

    def get_allow_comm_prefetch_overlap(self):
        if self.valid_conf_flag:
            return bool(self.allow_comm_prefetch_overlap)
        return True

    def get_enable_dynamic(self):
        """
        Method to check if dynamic bank allocation is enabled.
        """
        if self.valid_conf_flag:
            return self.enable_dynamic
        return False

    def get_bank_conflict_penalty(self):
        """
        Method to get the bank conflict penalty factor in cycles per request.
        """
        if self.valid_conf_flag:
            return max(1, int(self.bank_conflict_penalty))
        return 1

    def get_enable_capacity_penalty(self):
        """
        Method to check if SRAM overflow should incur DRAM penalty in bank model.
        """
        if self.valid_conf_flag:
            return bool(self.enable_capacity_penalty)
        return True

    def get_dram_penalty_scale(self):
        """
        Method to get DRAM overflow penalty scale used by bank model.
        """
        if self.valid_conf_flag:
            return max(1, int(self.dram_penalty_scale))
        return 8

    def get_bank_allocation(self):
        """
        Method to get static bank allocation tuple in config order.
        """
        if self.valid_conf_flag:
            return self.ifmap_sram_bank_num, self.filter_sram_bank_num, self.ofmap_sram_bank_num
        return 0, 0, 0

    def get_total_banknum(self):
        """
        Method to get total number of configured SRAM banks.
        """
        if self.valid_conf_flag:
            return self.ifmap_sram_bank_num + self.filter_sram_bank_num + self.ofmap_sram_bank_num
        return 0
    
    # FIX ISSUE #14
    @staticmethod
    def get_default_conf_as_list():
        """
        Method to get the default configuration as a list.
        """
        dummy_obj = scale_config()
        dummy_obj.force_valid()
        out_list = dummy_obj.get_conf_as_list()
        return out_list
