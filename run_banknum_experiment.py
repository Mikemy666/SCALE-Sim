#!/usr/bin/env python3
"""
批量实验脚本：自动修改配置文件并运行 SCALE-Sim
"""
import subprocess
import os
import configparser

# ==================== 配置区域 ====================
# 定义实验参数组合（根据你的需求修改）
experiments = [
    {
        "run_name": "baseline_dynamic",
        "IfmapSRAMBankBandwidth": 128,
        "IfmapSRAMBankNum": 8,
        "FilterSRAMBankBandwidth": 128,
        "FilterSRAMBankNum": 8,
    },
    {
        "run_name": "baseline_static",
        "IfmapSRAMBankBandwidth": 8,
        "IfmapSRAMBankNum": 1,
        "FilterSRAMBankBandwidth": 8,
        "FilterSRAMBankNum": 1,
    },
    {
        "run_name": "static_bw8",
        "IfmapSRAMBankBandwidth": 8,
        "IfmapSRAMBankNum": 1,
        "FilterSRAMBankBandwidth": 8,
        "FilterSRAMBankNum": 1,
    },
    {
        "run_name": "static_bw4",
        "IfmapSRAMBankBandwidth": 4,
        "IfmapSRAMBankNum": 1,
        "FilterSRAMBankBandwidth": 4,
        "FilterSRAMBankNum": 1,
    },
    {
        "run_name": "static_bw2",
        "IfmapSRAMBankBandwidth": 2,
        "IfmapSRAMBankNum": 1,
        "FilterSRAMBankBandwidth": 2,
        "FilterSRAMBankNum": 1,
    },
        {
        "run_name": "dynamic_bw8",
        "IfmapSRAMBankBandwidth": 64,
        "IfmapSRAMBankNum": 8,
        "FilterSRAMBankBandwidth": 64,
        "FilterSRAMBankNum": 8,
    },
    {
        "run_name": "dynamic_bw4",
        "IfmapSRAMBankBandwidth": 32,
        "IfmapSRAMBankNum": 8,
        "FilterSRAMBankBandwidth": 32,
        "FilterSRAMBankNum": 8,
    },
    {
        "run_name": "dynamic_bw2",
        "IfmapSRAMBankBandwidth": 16,
        "IfmapSRAMBankNum": 8,
        "FilterSRAMBankBandwidth": 16,
        "FilterSRAMBankNum": 8,
    },
    {
        "run_name": "dynamic_banknum2_14",
        "IfmapSRAMBankBandwidth": 16,
        "IfmapSRAMBankNum": 2,
        "FilterSRAMBankBandwidth": 112,
        "FilterSRAMBankNum": 14,
    },
    {
        "run_name": "dynamic_banknum4_12",
        "IfmapSRAMBankBandwidth": 32,
        "IfmapSRAMBankNum": 4,
        "FilterSRAMBankBandwidth": 96,
        "FilterSRAMBankNum": 12,
    },
    {
        "run_name": "dynamic_banknum6_10",
        "IfmapSRAMBankBandwidth": 48,
        "IfmapSRAMBankNum": 6,
        "FilterSRAMBankBandwidth": 80,
        "FilterSRAMBankNum": 10,
    },
    {
        "run_name": "dynamic_banknum10_6",
        "IfmapSRAMBankBandwidth": 80,
        "IfmapSRAMBankNum": 10,
        "FilterSRAMBankBandwidth": 48,
        "FilterSRAMBankNum": 6,
    },
    {
        "run_name": "dynamic_banknum12_4",
        "IfmapSRAMBankBandwidth": 96,
        "IfmapSRAMBankNum": 12,
        "FilterSRAMBankBandwidth": 32,
        "FilterSRAMBankNum": 4,
    },
    {
        "run_name": "dynamic_banknum14_2",
        "IfmapSRAMBankBandwidth": 112,
        "IfmapSRAMBankNum": 14,
        "FilterSRAMBankBandwidth": 16,
        "FilterSRAMBankNum": 2,
    }
    # 
    # 添加更多实验配置...
]

# 文件路径配置
CONFIG_FILE = "/home/MikeNotFound/code/SCALE-Sim/configs/scale_moe.cfg"
TOPOLOGY_FILE = "/home/MikeNotFound/code/SCALE-Sim/topologies/MoE/topo.csv"
OUTPUT_DIR = "/home/MikeNotFound/code/SCALE-Sim/outputs/banknum_exp"
WORKLOAD_TYPE = "gemm"

# ==================== 函数定义 ====================

def modify_config(config_file, params):
    """
    修改配置文件中的参数
    
    Args:
        config_file: 配置文件路径
        params: 包含要修改的参数的字典
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 修改 [general] 区域的 run_name
    if 'run_name' in params:
        config.set('general', 'run_name', params['run_name'])
    
    # 修改 [layout] 区域的参数
    if 'IfmapSRAMBankBandwidth' in params:
        config.set('layout', 'IfmapSRAMBankBandwidth', str(params['IfmapSRAMBankBandwidth']))
    if 'IfmapSRAMBankNum' in params:
        config.set('layout', 'IfmapSRAMBankNum', str(params['IfmapSRAMBankNum']))
    if 'FilterSRAMBankBandwidth' in params:
        config.set('layout', 'FilterSRAMBankBandwidth', str(params['FilterSRAMBankBandwidth']))
    if 'FilterSRAMBankNum' in params:
        config.set('layout', 'FilterSRAMBankNum', str(params['FilterSRAMBankNum']))
    
    # 写回配置文件
    with open(config_file, 'w') as f:
        config.write(f)
    
    print(f"✓ 配置已更新: {params['run_name']}")


def run_experiment(config_file, topology_file, output_dir, workload_type):
    """
    运行 SCALE-Sim 实验
    
    Args:
        config_file: 配置文件路径
        topology_file: 拓扑文件路径
        output_dir: 输出目录
        workload_type: 工作负载类型
    """
    cmd = [
        "python3", "-m", "scalesim.scale",
        "-c", config_file,
        "-t", topology_file,
        "-p", output_dir,
        "-i", workload_type
    ]
    
    print(f"▶ 执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✓ 实验完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 实验失败:")
        print(e.stderr)
        return False


# ==================== 主程序 ====================

def main():
    """主函数：循环执行所有实验"""
    print("=" * 60)
    print("开始批量实验")
    print(f"总实验数: {len(experiments)}")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for idx, exp_params in enumerate(experiments, 1):
        print(f"\n{'=' * 60}")
        print(f"实验 {idx}/{len(experiments)}: {exp_params['run_name']}")
        print(f"{'=' * 60}")
        
        # 修改配置文件
        modify_config(CONFIG_FILE, exp_params)
        
        # 运行实验
        if run_experiment(CONFIG_FILE, TOPOLOGY_FILE, OUTPUT_DIR, WORKLOAD_TYPE):
            success_count += 1
        else:
            fail_count += 1
    
    # 输出总结
    print("\n" + "=" * 60)
    print("所有实验完成")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
