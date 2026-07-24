import unittest

import numpy as np

from scalesim.memory.banked_memory_system import TensorBankModel
from scalesim.scale_config import scale_config
from scalesim.simulator import simulator
from scalesim.topology_utils import topologies


class Date1ExtensionTests(unittest.TestCase):
    def test_chunk_size_config_and_rechunk_conserve_totals(self):
        config = scale_config()
        config.read_conf_file('configs/MoE/DATE1/exp6/chunk_4096_window_1.cfg')
        self.assertEqual(config.get_chunk_size_bytes(), 4096)

        sim = simulator()
        sim.conf = config
        source = []
        for index, size in enumerate((8192, 2048, 2048)):
            elements = size // 2
            source.append({
                'weight_bytes': size, 'weight_elements': elements,
                'compute_cycles': 100, 'ifmap_requests': 100,
                'filter_requests': 100, 'ofmap_requests': 100,
                'raw_weight_address_min': index * 10000,
                'raw_weight_address_max': index * 10000 + elements - 1,
                'logical_weight_address_min': index * 10000,
                'logical_weight_address_max': index * 10000 + elements - 1,
                'trace_start_cycle': index * 100,
                'trace_end_cycle': (index + 1) * 100,
                'weight_trace_end_cycle': (index + 1) * 100,
            })
        chunks = sim._rechunk_detailed_trace_chunks(source)
        self.assertEqual(sum(item['weight_bytes'] for item in chunks), 12288)
        self.assertEqual(sum(item['compute_cycles'] for item in chunks), 300)
        self.assertEqual(sum(item['filter_requests'] for item in chunks), 300)
        self.assertTrue(all(item['actual_chunk_size_bytes'] <= 4096 for item in chunks))

    def test_tensor_bank_model_tracks_physical_bank_metrics(self):
        model = TensorBankModel('ifmap', bank_base=0, bank_count=2)
        model.service_line(0, np.asarray([0, 2, 1, -1]))
        self.assertEqual(sum(model.per_bank_access.values()), 3)
        self.assertGreater(sum(model.per_bank_busy_cycles.values()), 0)
        self.assertGreater(sum(model.per_bank_conflict_count.values()), 0)

    def test_routed_token_aware_trace_scales_detailed_gemm_m(self):
        config = scale_config()
        config.read_conf_file('configs/MoE/DATE1/exp7/tokens_128.cfg')
        self.assertTrue(config.get_enable_routed_token_aware_trace())

        topology = topologies()
        topology.load_arrays('topologies/MoE/DATE1/exp7/moe_8e.csv', mnk_inputs=True)
        sim = simulator()
        sim.set_params(config_obj=config, topo_obj=topology, verbosity=False)
        plan = sim._build_ep_moe_execution_plan()
        rows = sim._apply_routed_token_aware_traces(plan)

        detailed = [row for row in rows if row['IsDetailedGPU'] and row['IsActiveExpert']]
        blackbox = [row for row in rows if not row['IsDetailedGPU']]
        self.assertTrue(detailed)
        self.assertTrue(all(row['OriginalM'] == 32 for row in detailed))
        self.assertTrue(all(row['RoutedTokens'] == 16 for row in detailed))
        self.assertTrue(all(row['EffectiveM'] == 16 and row['TraceScaled'] for row in detailed))
        self.assertTrue(all(row['EffectiveM'] == 32 and not row['TraceScaled'] for row in blackbox))
        self.assertEqual(topology.get_layer_ifmap_dims(0)[0], 16)
        self.assertEqual(topology.get_layer_ifmap_dims(8)[0], 32)


if __name__ == '__main__':
    unittest.main()
