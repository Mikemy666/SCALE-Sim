"""Run architectural EP-MoE degradation checks."""

import argparse
import configparser
import csv
from pathlib import Path
import tempfile

from scalesim.scale_sim import scalesim


def _read_one(path):
    with path.open(newline='', encoding='utf-8') as report:
        return next(csv.DictReader(report, skipinitialspace=True))


def validate_sanity(results):
    baseline = results['baseline']
    single = results['single_gpu']
    pressure = results['background_pressure']
    sequential = results['sequential']
    if int(single['NumBlackBoxExperts']) != 0:
        raise RuntimeError('NumGPUs=1 did not remove black-box experts')
    if int(baseline['TotalPrefetchHit']) or int(baseline['TotalPrefetchMiss']):
        raise RuntimeError('Disabled prefetch produced hit/miss activity')
    if int(baseline['TotalPrefetchBandwidthOverhead']):
        raise RuntimeError('Disabled prefetch produced bandwidth overhead')
    if int(baseline['GroupRuntimeBlackBoxBackgroundPressureStall']):
        raise RuntimeError('Disabled black-box pressure produced runtime stall')
    if int(pressure['GroupRuntimeBlackBoxBackgroundPressureStall']) <= 0:
        raise RuntimeError('Enabled black-box pressure produced no runtime stall')
    if int(sequential['MoEGroupTime']) < int(baseline['MoEGroupTime']):
        raise RuntimeError('Sequential EP unexpectedly completed before parallel EP')
    if baseline['DynamicBankOverheadModel'] != 'old_model':
        raise RuntimeError('Dynamic bank overhead no longer uses old_model')


def _write_variant(base_path, output_path, changes):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(base_path)
    for key, value in changes.items():
        config.set('run_presets', key, str(value))
    config.set('general', 'run_name', output_path.stem)
    with output_path.open('w', encoding='utf-8') as output:
        config.write(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/ep_moe_sanity')
    parser.add_argument('--config', default='configs/MoE/ep_default.cfg')
    parser.add_argument('--topology', default='topologies/MoE/test.csv')
    parser.add_argument('--layout', default='layouts/conv_nets/test.csv')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        'baseline': {},
        'single_gpu': {'NumGPUs': 1, 'ExpertsPerGPU': 8, 'BlackBoxGPUIds': ''},
        'background_pressure': {'EnableBlackBoxBackgroundPressure': True},
        'sequential': {'EnableParallelMoE': False},
    }
    results = {}
    with tempfile.TemporaryDirectory(prefix='scalesim-sanity-') as temp_dir:
        for name, changes in variants.items():
            summary_path = output / name / 'EP_MOE_SUMMARY.csv'
            if summary_path.exists():
                results[name] = _read_one(summary_path)
                continue
            config_path = Path(temp_dir) / (name + '.cfg')
            _write_variant(args.config, config_path, changes)
            run = scalesim(True, False, str(config_path), args.topology, args.layout, True)
            run.run_scale(top_path=str(output))
            results[name] = _read_one(summary_path)
    validate_sanity(results)
    print('EP-MoE sanity checks passed')


if __name__ == '__main__':
    main()
