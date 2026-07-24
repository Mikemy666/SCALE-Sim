"""Select representative static allocations and propagate the best one to exp3."""

import configparser
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'outputs/DATE1/exp2/exhaustive_static_search/BANK_ALLOCATION_SWEEP_REPORT.csv'
PARTIAL_REPORT = ROOT / 'outputs/DATE1/exp2/exhaustive_static_search/BANK_ALLOCATION_SWEEP_PARTIAL.csv'
TOPOLOGY = ROOT / 'topologies/MoE/DATE1/exp2/modse_full.csv'
CONFIG_ROOT = ROOT / 'configs/MoE/DATE1'
OUTPUT = ROOT / 'outputs/DATE1/exp2/exhaustive_static_search'
PAPER_STATIC_BASELINE = '4:14:6'


def load_parser(path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)
    return parser


def write_selected_config(ratio, rank):
    ia, weight, oa = map(int, ratio.split(':'))
    source = CONFIG_ROOT / 'exp2/exhaustive_static_search.cfg'
    parser = load_parser(source)
    name = f'selected_{rank}_{ia}_{weight}_{oa}'
    parser['general']['run_name'] = name
    parser['run_presets']['EnableDynamic'] = 'False'
    parser['layout']['ifmapsrambanknum'] = str(ia)
    parser['layout']['filtersrambanknum'] = str(weight)
    parser['layout']['ofmapsrambanknum'] = str(oa)
    destination = CONFIG_ROOT / 'exp2/selected' / (name + '.cfg')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as config_file:
        parser.write(config_file)


def update_exp3(_measured_best_ratio):
    ia, weight, oa = map(int, PAPER_STATIC_BASELINE.split(':'))
    for path in sorted((CONFIG_ROOT / 'exp3').glob('*.cfg')):
        parser = load_parser(path)
        parser['layout']['ifmapsrambanknum'] = str(ia)
        parser['layout']['filtersrambanknum'] = str(weight)
        parser['layout']['ofmapsrambanknum'] = str(oa)
        with path.open('w', encoding='utf-8') as config_file:
            parser.write(config_file)
    marker = CONFIG_ROOT / 'exp3/STATIC_BEST_FROM_EXP2.txt'
    marker.write_text(
        f'实验3--7统一采用的论文静态基线 IA:Weight:OA = {PAPER_STATIC_BASELINE}\n'
        '该比例由实验方案固定，不会被实验2扫描结果自动覆盖。\n',
        encoding='utf-8',
    )


def main():
    report_path = REPORT if REPORT.exists() else PARTIAL_REPORT
    if not report_path.exists():
        raise SystemExit(
            '缺少全静态扫描报告。请先运行：\n'
            './scripts/DATE1/run_exp2.sh'
        )

    topology = pd.read_csv(TOPOLOGY, index_col=False)
    moe_ids = set(topology.index[topology['Layer'].astype(str).str.startswith('MoE-')])
    sweep = pd.read_csv(report_path)
    sweep = sweep[sweep['LayerID'].isin(moe_ids)].copy()
    expected = 253 * len(moe_ids)
    if len(sweep) != expected:
        raise SystemExit(f'扫描报告不完整：期望 {expected} 行，实际 {len(sweep)} 行')

    summary = sweep.groupby('AllocationRatio').agg(
        MoETotalCycles=('TotalCycles', 'sum'),
        MoEStallCycles=('StallCycles', 'sum'),
        MoETotalConflictDelay=('TotalConflictDelay', 'sum'),
    ).reset_index()
    summary = summary.sort_values(['MoETotalCycles', 'MoEStallCycles', 'AllocationRatio']).reset_index(drop=True)
    summary['NormalizedCycles'] = summary['MoETotalCycles'] / summary['MoETotalCycles'].min()
    summary['Rank'] = summary.index + 1

    targets = [1.0, 1.15, 1.35, 1.75, float(summary['NormalizedCycles'].max())]
    selected_indices = []
    for target in targets:
        available = summary.loc[~summary.index.isin(selected_indices)]
        index = (available['NormalizedCycles'] - target).abs().idxmin()
        selected_indices.append(index)
    equal = summary.index[summary['AllocationRatio'] == '8:8:8']
    if len(equal) and int(equal[0]) not in selected_indices:
        selected_indices.append(int(equal[0]))

    selected = summary.loc[selected_indices].sort_values('NormalizedCycles').reset_index(drop=True)
    selected['SelectionLabel'] = ['best'] + [f'representative_{i}' for i in range(1, len(selected))]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / 'ALL_STATIC_ALLOCATION_SUMMARY.csv', index=False)
    selected.to_csv(OUTPUT / 'SELECTED_STATIC_ALLOCATION_SUMMARY.csv', index=False)

    for _, row in selected.iterrows():
        write_selected_config(row['AllocationRatio'], row['SelectionLabel'])

    best_ratio = str(summary.iloc[0]['AllocationRatio'])
    update_exp3(best_ratio)
    print(selected.to_string(index=False))
    print(f'\n全局静态最优：{best_ratio}')
    print(f'实验3配置保持论文指定静态基线：{PAPER_STATIC_BASELINE}。')


if __name__ == '__main__':
    main()
