"""Generate the DATE1 experiment configs, topologies, and output hierarchy."""

from configparser import ConfigParser
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'configs' / 'MoE' / 'DATE1'
TOPOLOGY_ROOT = ROOT / 'topologies' / 'MoE' / 'DATE1'
OUTPUT_ROOT = ROOT / 'outputs' / 'DATE1'


BASE = {
    'arrayheight': '64', 'arraywidth': '64',
    'ifmapsramszkb': '32', 'filtersramszkb': '32', 'ofmapsramszkb': '32',
    'ifmapoffset': '0', 'filteroffset': '10000000', 'ofmapoffset': '20000000',
    'bandwidth': '1024', 'dataflow': 'ws',
    'readrequestbuffer': '128', 'writerequestbuffer': '128',
}

LAYOUT = {
    'ifmapcustomlayout': 'True', 'ifmapsrambankbandwidth': '128',
    'ifmapsrambanknum': '8', 'ifmapsrambankport': '1',
    'filtercustomlayout': 'True', 'filtersrambankbandwidth': '128',
    'filtersrambanknum': '8', 'filtersrambankport': '1',
    'ofmapsrambankbandwidth': '128', 'ofmapsrambanknum': '8',
    'ofmapsrambankport': '1',
}

RUN = {
    'interfacebandwidth': 'USER', 'useramulatortrace': 'False',
    'EnableBankModel': 'True', 'EnableDynamic': 'False',
    'BankConflictPenalty': '4', 'EnableCapacityPenalty': 'False',
    'DRAMPenaltyScale': '2', 'EnablePrefetch': 'False', 'PrefetchWindow': '0',
    'EnableEPMoE': 'True', 'EnableParallelMoE': 'True',
    'NumGPUs': '2', 'DetailedGPUId': '0', 'BlackBoxGPUIds': '1',
    'ExpertsPerGPU': '4', 'ComputeEnginesPerGPU': '4',
    'TopK': '1', 'MoERoutingMode': 'topology_counts', 'MoETokens': '256',
    'RoutingFile': '', 'RoutingSeed': '40', 'RoutingSkewFactor': '1.0',
    'MoEActiveExpertMode': 'all', 'ActiveExpertIds': '',
    'EnableChunkPrefetch': 'False', 'InitialChunk': '1',
    'ChunkPrefetchWindow': '0', 'ChunkSizeBytes': '0',
    'BlackBoxWorkloadMode': 'analytical',
    'BlackBoxBandwidthBytesPerCycle': '128',
    'EnableBlackBoxBackgroundPressure': 'False',
    'GlobalMemoryBandwidthBytesPerCycle': '1024',
    'DynamicBankOverhead': 'old_model',
    'CommunicationModel': 'latency_plus_bandwidth', 'PrecisionBytes': '2',
    'CommunicationLatencyCycles': '20',
    'CommunicationBandwidthBytesPerCycle': '128',
    'CommunicationOverlapMode': 'prefetch_only',
    'AllowCommPrefetchOverlap': 'True',
}


def write_config(exp, name, overrides=None, ep=True, banks=(8, 8, 8)):
    parser = ConfigParser()
    parser.optionxform = str
    parser['general'] = {'run_name': name}
    parser['architecture_presets'] = BASE
    layout = dict(LAYOUT)
    layout['ifmapsrambanknum'], layout['filtersrambanknum'], layout['ofmapsrambanknum'] = map(str, banks)
    parser['layout'] = layout
    parser['sparsity'] = {
        'sparsitysupport': 'False', 'sparserep': 'ellpack_block',
        'optimizedmapping': 'False', 'blocksize': '8',
        'randomnumbergeneratorseed': '40',
    }
    run = dict(RUN)
    run['EnableEPMoE'] = str(bool(ep))
    if not ep:
        run['EnableParallelMoE'] = 'False'
        run['EnableChunkPrefetch'] = 'False'
        run['ChunkPrefetchWindow'] = '0'
    if overrides:
        run.update({key: str(value) for key, value in overrides.items()})
    parser['run_presets'] = run
    path = CONFIG_ROOT / exp / (name + '.cfg')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as config_file:
        parser.write(config_file)
    output = OUTPUT_ROOT / exp / name
    output.mkdir(parents=True, exist_ok=True)
    (output / '.gitkeep').touch()


def write_expert_topology(path, experts):
    dims = [1728, 192, 1536, 384, 1152, 768, 960, 960]
    lines = ['Layer,M,N,K,']
    for expert_id in range(experts):
        width = dims[expert_id % len(dims)]
        lines.extend([
            f'MoE-E{expert_id}-FF1,32,{width},384,',
            f'MoE-E{expert_id}-FF2,32,384,{width},',
            '',
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def copy_topology(exp, source, target):
    destination = TOPOLOGY_ROOT / exp / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / source, destination)


def main():
    for root in (CONFIG_ROOT, TOPOLOGY_ROOT, OUTPUT_ROOT):
        root.mkdir(parents=True, exist_ok=True)

    # Exp1: complete MoDSE, one fixed static/no-prefetch baseline.
    copy_topology('exp1', 'topologies/MoE/MoDSE.csv', 'modse_full.csv')
    write_config('exp1', 'static_no_prefetch', ep=False)

    # Exp2: complete MoDSE, static IA/W/OA ratio sweep.
    copy_topology('exp2', 'topologies/MoE/MoDSE.csv', 'modse_full.csv')
    for banks in ((8, 8, 8), (5, 15, 4), (7, 14, 3), (14, 7, 3),
                  (11, 7, 6), (5, 7, 12), (3, 7, 14)):
        write_config('exp2', 'static_' + '_'.join(map(str, banks)), ep=False, banks=banks)

    # Exp3: MoE-only static prefetch interference sweep.
    copy_topology('exp3', 'topologies/MoE/MoE.csv', 'modse_moe_8e.csv')
    write_config('exp3', 'static_no_prefetch', {'MoERoutingMode': 'topology_counts'})
    for window in (1, 2, 4):
        write_config('exp3', f'static_prefetch_w{window}', {
            'EnableChunkPrefetch': 'True', 'ChunkPrefetchWindow': window,
            'MoERoutingMode': 'topology_counts',
        })

    # Exp4: static equal, selected best-static candidate, and dynamic.
    copy_topology('exp4', 'topologies/MoE/MoE.csv', 'modse_moe_8e.csv')
    write_config('exp4', 'static_equal_8_8_8', {'MoERoutingMode': 'topology_counts'})
    write_config('exp4', 'static_candidate_7_14_3', {'MoERoutingMode': 'topology_counts'}, banks=(7, 14, 3))
    write_config('exp4', 'dynamic_24', {'EnableDynamic': 'True', 'MoERoutingMode': 'topology_counts'})

    # Exp5: canonical 2x2 ablation.
    copy_topology('exp5', 'topologies/MoE/MoE.csv', 'modse_moe_8e.csv')
    for dynamic, prefetch in ((False, False), (False, True), (True, False), (True, True)):
        name = ('dynamic' if dynamic else 'static') + '_' + ('prefetch' if prefetch else 'no_prefetch')
        write_config('exp5', name, {
            'EnableDynamic': dynamic,
            'EnableChunkPrefetch': prefetch,
            'ChunkPrefetchWindow': 1 if prefetch else 0,
            'MoERoutingMode': 'topology_counts',
        })

    # Exp6: window sweep plus ChunkSizeBytes x window matrix.
    copy_topology('exp6', 'topologies/MoE/MoE.csv', 'modse_moe_8e.csv')
    for window in (0, 1, 2, 4, 8):
        write_config('exp6', f'window_{window}', {
            'EnableDynamic': 'True', 'EnableChunkPrefetch': window > 0,
            'ChunkPrefetchWindow': window, 'ChunkSizeBytes': 0,
            'MoERoutingMode': 'topology_counts',
        })
    for chunk_size in (4096, 8192, 16384, 32768):
        for window in (1, 2, 4, 8):
            write_config('exp6', f'chunk_{chunk_size}_window_{window}', {
                'EnableDynamic': 'True', 'EnableChunkPrefetch': 'True',
                'ChunkPrefetchWindow': window, 'ChunkSizeBytes': chunk_size,
                'MoERoutingMode': 'topology_counts',
            })

    # Exp7: controlled 4/8/16-expert topologies and robustness sweeps.
    for experts in (4, 8, 16):
        write_expert_topology(TOPOLOGY_ROOT / 'exp7' / f'moe_{experts}e.csv', experts)
        write_config('exp7', f'experts_{experts}', {
            'EnableDynamic': 'True', 'EnableChunkPrefetch': 'True',
            'ChunkPrefetchWindow': 1, 'NumGPUs': 2,
            'ExpertsPerGPU': experts // 2, 'MoERoutingMode': 'balanced',
            'MoETokens': 256,
        })
    for top_k in (1, 2):
        write_config('exp7', f'topk_{top_k}', {
            'EnableDynamic': 'True', 'EnableChunkPrefetch': 'True',
            'ChunkPrefetchWindow': 1, 'TopK': top_k,
            'MoERoutingMode': 'balanced', 'MoETokens': 256,
        })
    for tokens in (32, 128, 256, 512):
        write_config('exp7', f'tokens_{tokens}', {
            'EnableDynamic': 'True', 'EnableChunkPrefetch': 'True',
            'ChunkPrefetchWindow': 1, 'MoERoutingMode': 'balanced',
            'MoETokens': tokens,
        })
    write_config('exp7', 'routing_balanced', {
        'EnableDynamic': 'True', 'EnableChunkPrefetch': 'True',
        'ChunkPrefetchWindow': 1, 'MoERoutingMode': 'balanced', 'MoETokens': 256,
    })
    for skew in (0.5, 1.0, 2.0, 4.0):
        for seed in range(40, 45):
            skew_name = str(skew).replace('.', 'p')
            write_config('exp7', f'routing_skew_{skew_name}_seed_{seed}', {
                'EnableDynamic': 'True', 'EnableChunkPrefetch': 'True',
                'ChunkPrefetchWindow': 1, 'MoERoutingMode': 'seeded_skewed',
                'MoETokens': 256, 'RoutingSkewFactor': skew, 'RoutingSeed': seed,
            })

    readme = OUTPUT_ROOT / 'README.md'
    readme.write_text(
        '# DATE1 outputs\n\n'
        'Run a group with `python3 run_date1_experiments.py --exp expN`.\n'
        'Use `--variant NAME` for one config and `--dry-run` to inspect commands.\n'
        'The config run_name creates the variant subdirectory.\n'
        'The checked-in `.gitkeep` files preserve the expected output hierarchy.\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
