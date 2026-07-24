import json, subprocess, sys, unittest
from pathlib import Path
from scalesim.memory.memdomain_runner import load_runner_config, run_matrix
from scalesim.memory.memdomain_experiment import Baseline

ROOT=Path(__file__).resolve().parents[1]

class Date2PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable,str(ROOT/"scripts/prepare_date2_experiments.py")],cwd=ROOT,check=True)

    def test_manifest_covers_all_paper_simulator_sweeps(self):
        manifest=json.loads((ROOT/"configs/MoE/DATE2/manifest.json").read_text())
        self.assertEqual(manifest["suites"],{"overall":4,"window_chunk":32,"robustness":26,"characterization":3})
        self.assertTrue(manifest["simulator_only"]); self.assertTrue(manifest["rtl_dc_out_of_scope"])

    def test_every_config_parses_and_paths_exist(self):
        configs=list((ROOT/"configs/MoE/DATE2").glob("*/*.json"))
        self.assertEqual(len(configs),65)
        for path in configs:
            if path.parent.name not in {"exp1","exp2","exp3"}:
                self.assertGreater(len(load_runner_config(path).chunks),0)
        self.assertEqual(len(list((ROOT/"topologies/MoE/DATE2/models").glob("*.csv"))),4)

    def test_ep_model_is_zero_for_one_gpu_and_nonzero_for_two(self):
        one={r.baseline:r for r in run_matrix(load_runner_config(ROOT/"configs/MoE/DATE2/robustness/ep_1gpu.json"))}
        two={r.baseline:r for r in run_matrix(load_runner_config(ROOT/"configs/MoE/DATE2/robustness/ep_2gpu.json"))}
        self.assertEqual(one[Baseline.STATIC_NOPF.value].communication_stall_cycles,0)
        self.assertGreater(two[Baseline.STATIC_NOPF.value].communication_stall_cycles,0)

    def test_largest_chunk_uses_multibank_groups(self):
        config=load_runner_config(ROOT/"configs/MoE/DATE2/window_chunk/w8_c8.json")
        self.assertEqual(max(chunk.bank_group_size for chunk in config.chunks),2)

if __name__=="__main__": unittest.main()
