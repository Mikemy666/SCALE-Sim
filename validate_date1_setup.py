"""Validate DATE1 config expert counts against their selected topologies."""

import argparse
import configparser
import csv
import re
from pathlib import Path


EXPERT_RE = re.compile(r'^MoE-E(\d+)-FF([12])$')


def topology_experts(path):
    parts = {}
    with Path(path).open(newline='', encoding='utf-8') as topology_file:
        for row in csv.DictReader(topology_file):
            name = (row.get('Layer') or '').strip()
            match = EXPERT_RE.match(name)
            if match:
                parts.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    incomplete = {expert: sorted(value) for expert, value in parts.items() if value != {1, 2}}
    if incomplete:
        raise ValueError(f'Incomplete expert FFN pairs in {path}: {incomplete}')
    ids = sorted(parts)
    if ids and ids != list(range(len(ids))):
        raise ValueError(f'Expert IDs must be contiguous in {path}: {ids}')
    return len(ids)


def config_experts(path):
    parser = configparser.ConfigParser()
    parser.read(path)
    run = parser['run_presets']
    if not run.getboolean('EnableEPMoE', fallback=False):
        return None
    return run.getint('NumGPUs') * run.getint('ExpertsPerGPU')


def validate_pair(config_path, topology_path):
    expected = config_experts(config_path)
    actual = topology_experts(topology_path)
    if expected is not None and expected != actual:
        raise ValueError(
            f'Expert count mismatch: config={config_path} expects {expected}, '
            f'topology={topology_path} contains {actual}'
        )
    return {'config_experts': expected, 'topology_experts': actual}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('topology')
    args = parser.parse_args()
    print('DATE1 setup valid:', validate_pair(args.config, args.topology))
