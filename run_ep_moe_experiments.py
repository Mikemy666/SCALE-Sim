"""Run and summarize the four canonical EP-MoE bank/prefetch experiments."""

import argparse
import csv
from pathlib import Path

from scalesim.scale_sim import scalesim


EXPERIMENTS = (
    'static_no_prefetch',
    'static_prefetch',
    'dynamic_no_prefetch',
    'dynamic_prefetch',
)


def _read_one(path):
    with path.open(newline='', encoding='utf-8') as report:
        return next(csv.DictReader(report, skipinitialspace=True))


def collect_result(run_dir, experiment):
    summary = _read_one(run_dir / 'EP_MOE_SUMMARY.csv')
    with (run_dir / 'EP_MOE_BANK_ALLOCATION.csv').open(newline='', encoding='utf-8') as report:
        bank_rows = list(csv.DictReader(report, skipinitialspace=True))
    detailed_bank_stall = sum(int(row['LayerBankConflictStall']) for row in bank_rows)
    return {
        'Experiment': experiment,
        'BankMode': 'dynamic' if experiment.startswith('dynamic') else 'static',
        'PrefetchEnabled': experiment.endswith('_prefetch') and not experiment.endswith('no_prefetch'),
        'MoEGroupTime': int(summary['MoEGroupTime']),
        'TotalExpertWaitingCycles': int(summary['TotalExpertWaitingCycles']),
        'TotalPrefetchHit': int(summary['TotalPrefetchHit']),
        'TotalPrefetchMiss': int(summary['TotalPrefetchMiss']),
        'AvgPrefetchHitRate': float(summary['AvgPrefetchHitRate']),
        'TotalPrefetchMissStall': int(summary['TotalPrefetchMissStall']),
        'TotalWeightLoadingStall': int(summary['TotalWeightLoadingStall']),
        'TotalPrefetchBankInterferenceStall': int(summary['TotalPrefetchBankInterferenceStall']),
        'TotalPrefetchBandwidthOverhead': int(summary['TotalPrefetchBandwidthOverhead']),
        'DetailedBankConflictStall': int(detailed_bank_stall),
        'DynamicBankOverheadModel': summary['DynamicBankOverheadModel'],
    }


def validate_matrix(rows):
    if len(rows) != 4:
        raise RuntimeError('The canonical experiment matrix must contain four rows')
    pairs = {(row['BankMode'], bool(row['PrefetchEnabled'])) for row in rows}
    expected = {(mode, enabled) for mode in ('static', 'dynamic') for enabled in (False, True)}
    if pairs != expected:
        raise RuntimeError('Experiment matrix does not cover static/dynamic x prefetch on/off')
    for row in rows:
        if row['DynamicBankOverheadModel'] != 'old_model':
            raise RuntimeError('DynamicBankOverhead must remain old_model')
        if not row['PrefetchEnabled'] and row['TotalPrefetchBandwidthOverhead'] != 0:
            raise RuntimeError('No-prefetch run issued runtime prefetch traffic')


def build_comparisons(rows):
    indexed = {(row['BankMode'], bool(row['PrefetchEnabled'])): row for row in rows}
    comparisons = []
    for mode in ('static', 'dynamic'):
        baseline = indexed[(mode, False)]['MoEGroupTime']
        prefetched = indexed[(mode, True)]['MoEGroupTime']
        comparisons.append({
            'Comparison': mode + '_prefetch_vs_no_prefetch',
            'BaselineCycles': baseline,
            'VariantCycles': prefetched,
            'CycleDelta': prefetched - baseline,
            'Speedup': float(baseline) / float(prefetched),
        })
    for enabled, label in ((False, 'no_prefetch'), (True, 'prefetch')):
        baseline = indexed[('static', enabled)]['MoEGroupTime']
        dynamic = indexed[('dynamic', enabled)]['MoEGroupTime']
        comparisons.append({
            'Comparison': 'dynamic_vs_static_' + label,
            'BaselineCycles': baseline,
            'VariantCycles': dynamic,
            'CycleDelta': dynamic - baseline,
            'Speedup': float(baseline) / float(dynamic),
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/ep_moe_matrix')
    parser.add_argument('--topology', default='topologies/MoE/test.csv')
    parser.add_argument('--layout', default='layouts/conv_nets/test.csv')
    parser.add_argument('--configs', default='configs/MoE/ep_experiments')
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment in EXPERIMENTS:
        config_path = Path(args.configs) / (experiment + '.cfg')
        run = scalesim(
            save_disk_space=True,
            verbose=False,
            config=str(config_path),
            topology=args.topology,
            layout=args.layout,
            input_type_gemm=True,
        )
        run.run_scale(top_path=str(output))
        rows.append(collect_result(output / ('ep_' + experiment), experiment))

    validate_matrix(rows)
    matrix_path = output / 'EP_MOE_EXPERIMENT_MATRIX.csv'
    with matrix_path.open('w', newline='', encoding='utf-8') as report:
        writer = csv.DictWriter(report, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = build_comparisons(rows)
    comparison_path = output / 'EP_MOE_EXPERIMENT_COMPARISONS.csv'
    with comparison_path.open('w', newline='', encoding='utf-8') as report:
        writer = csv.DictWriter(report, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(matrix_path)


if __name__ == '__main__':
    main()
