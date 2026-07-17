"""Validate cross-report invariants for one completed EP-MoE run."""

import argparse
import csv
import hashlib
from pathlib import Path


def _rows(path):
    with path.open(newline='', encoding='utf-8') as report:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}
            for row in csv.DictReader(report, skipinitialspace=True)
        ]


def validate_report_directory(run_dir):
    run_dir = Path(run_dir)
    timeline = _rows(run_dir / 'EP_MOE_TIMELINE.csv')
    experts = _rows(run_dir / 'EP_MOE_REPORT.csv')
    summary = _rows(run_dir / 'EP_MOE_SUMMARY.csv')
    routing = _rows(run_dir / 'EP_MOE_ROUTING.csv')
    runtime = _rows(run_dir / 'EP_MOE_RUNTIME_STATE.csv')
    chunks = _rows(run_dir / 'EP_MOE_CHUNKS.csv')
    events = _rows(run_dir / 'EP_MOE_EVENTS.csv')
    manifest_path = run_dir / 'EP_MOE_RUN_MANIFEST.csv'
    if manifest_path.exists():
        for row in _rows(manifest_path):
            source = Path(row['Path'])
            if not source.is_file():
                continue
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != row['SHA256'] or source.stat().st_size != int(row['SizeBytes']):
                raise RuntimeError('Run manifest no longer matches input file ' + row['InputKind'])

    previous_finish = 0
    group_timeline = {}
    for row in timeline:
        start = int(row['StartCycle'])
        finish = int(row['FinishCycle'])
        duration = int(row['DurationCycles'])
        if start != previous_finish or finish - start != duration or finish < start:
            raise RuntimeError('EP timeline is not contiguous and duration-consistent')
        previous_finish = finish
        if row['TimelineType'] == 'moe_group':
            group_timeline[int(row['MoEGroupID'])] = row

    for group_id, timeline_row in group_timeline.items():
        group_experts = [row for row in experts if int(row['MoEGroupID']) == group_id]
        active = [row for row in group_experts if row['IsActiveExpert'] == 'True']
        if active and max(int(row['ExpertFinishCycle']) for row in active) != int(timeline_row['FinishCycle']):
            raise RuntimeError('MoE group finish is not the slowest active expert finish')
        group_summary = next(row for row in summary if int(row['MoEGroupID']) == group_id)
        if int(group_summary['MoEGroupTime']) != int(timeline_row['DurationCycles']):
            raise RuntimeError('MoE summary time disagrees with the timeline')

    for row in routing:
        expert_ids = [item for item in row['ExpertIDs'].split('|') if item != '']
        if len(expert_ids) != int(row['TopK']) or len(expert_ids) != len(set(expert_ids)):
            raise RuntimeError('Routing row does not contain TopK unique experts')

    chunk_counts = {}
    for row in chunks:
        key = (int(row['MoEGroupID']), int(row['ExpertID']))
        chunk_counts[key] = chunk_counts.get(key, 0) + 1
    for row in runtime:
        key = (int(row['MoEGroupID']), int(row['ExpertID']))
        if chunk_counts.get(key, 0) != int(row['ChunkCount']):
            raise RuntimeError('Runtime chunk count disagrees with the chunk report')

    runtime_by_expert = {
        (int(row['MoEGroupID']), int(row['ExpertID'])): row for row in runtime
    }
    metric_pairs = (
        ('WeightChunkCount', 'ChunkCount'),
        ('InitialWeightStall', 'RuntimeInitialWeightStall'),
        ('WeightLoadingStall', 'RuntimeWeightLoadingStall'),
        ('PrefetchHit', 'RuntimePrefetchHit'),
        ('PrefetchMiss', 'RuntimePrefetchMiss'),
        ('PrefetchMissStall', 'RuntimePrefetchMissStall'),
        ('PrefetchBandwidthOverhead', 'RuntimePrefetchBandwidthOverhead'),
        ('UsefulPrefetchTraffic', 'RuntimeUsefulPrefetchTraffic'),
        ('UselessPrefetchTraffic', 'RuntimeUselessPrefetchTraffic'),
    )
    for row in experts:
        key = (int(row['MoEGroupID']), int(row['ExpertID']))
        state = runtime_by_expert[key]
        for expert_field, runtime_field in metric_pairs:
            if int(row[expert_field]) != int(state[runtime_field]):
                raise RuntimeError('Expert report metric disagrees with runtime state: ' + expert_field)

    for group_row in summary:
        group_id = int(group_row['MoEGroupID'])
        states = [
            row for key, row in runtime_by_expert.items()
            if key[0] == group_id and row['IsActiveExpert'] == 'True'
        ]
        aggregate_pairs = (
            ('TotalPrefetchHit', 'RuntimePrefetchHit'),
            ('TotalPrefetchMiss', 'RuntimePrefetchMiss'),
            ('TotalPrefetchMissStall', 'RuntimePrefetchMissStall'),
            ('TotalWeightLoadingStall', 'RuntimeWeightLoadingStall'),
            ('TotalPrefetchBandwidthOverhead', 'RuntimePrefetchBandwidthOverhead'),
        )
        for summary_field, runtime_field in aggregate_pairs:
            expected = sum(int(state[runtime_field]) for state in states)
            if int(group_row[summary_field]) != expected:
                raise RuntimeError('Summary metric disagrees with runtime states: ' + summary_field)

    layer_ranges = {}
    for row in chunks:
        if row['ChunkSource'] != 'detailed_demand_trace':
            continue
        layer_id = int(row['LayerID'])
        low = int(row['LogicalWeightAddressMin'])
        high = int(row['LogicalWeightAddressMax'])
        old = layer_ranges.get(layer_id, (low, high))
        layer_ranges[layer_id] = (min(old[0], low), max(old[1], high))
    ordered_ranges = sorted(layer_ranges.items())
    for (_, previous), (_, current) in zip(ordered_ranges, ordered_ranges[1:]):
        if previous[1] >= current[0]:
            raise RuntimeError('Detailed layer logical weight ranges overlap')

    event_cycles = [int(row['Cycle']) for row in events]
    if event_cycles != sorted(event_cycles):
        raise RuntimeError('EP event report is not cycle ordered')
    return {
        'TotalCycles': previous_finish,
        'MoEGroups': len(group_timeline),
        'Experts': len(experts),
        'RoutingRows': len(routing),
        'Chunks': len(chunks),
        'Events': len(events),
        'Manifest': manifest_path.exists(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    args = parser.parse_args()
    result = validate_report_directory(args.run_dir)
    print('EP-MoE reports valid:', result)


if __name__ == '__main__':
    main()
