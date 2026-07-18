"""Run one DATE1 experiment group with the correct config/topology/output mapping."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def topology_for(exp, config):
    topology_root = ROOT / 'topologies' / 'MoE' / 'DATE1' / exp
    if exp in ('exp1', 'exp2'):
        return topology_root / 'modse_full.csv'
    if exp == 'exp7' and config.stem.startswith('experts_'):
        count = config.stem.split('_', 1)[1]
        return topology_root / f'moe_{count}e.csv'
    if exp == 'exp7':
        return topology_root / 'moe_8e.csv'
    return topology_root / 'modse_moe_8e.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, choices=[f'exp{i}' for i in range(1, 8)])
    parser.add_argument('--variant', help='Run only this config stem, for example dynamic_prefetch')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    config_dir = ROOT / 'configs' / 'MoE' / 'DATE1' / args.exp
    configs = sorted(config_dir.glob('*.cfg'))
    if args.variant:
        configs = [path for path in configs if path.stem == args.variant]
    if not configs:
        raise SystemExit('No matching DATE1 configs')

    output_root = ROOT / 'outputs' / 'DATE1' / args.exp
    for config in configs:
        topology = topology_for(args.exp, config)
        command = [
            sys.executable, '-m', 'scalesim.scale',
            '-c', str(config), '-t', str(topology),
            '-p', str(output_root), '-i', 'gemm', '-s', 'N',
        ]
        print(' '.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
