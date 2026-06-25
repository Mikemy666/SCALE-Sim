#!/usr/bin/env bash
# 批量执行 SCALE-Sim：遍历 MoE 配置目录下所有 cfg 文件
# 每个 cfg 单独跑一次仿真，并保存日志

set -euo pipefail

# ==================== 固定路径配置 ====================
CFG_DIR="/home/MikeNotFound/code/SCALE-Sim/configs/MoE/end2end/port"
TOPOLOGY_FILE="/home/MikeNotFound/code/SCALE-Sim/topologies/MoE/MoE.csv"
OUTPUT_DIR="/home/MikeNotFound/code/SCALE-Sim/outputs/final4/end2end/port"
WORKLOAD_TYPE="gemm"
LOG_DIR="$OUTPUT_DIR/logs"

# ==================== 准备环境 ====================
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================"
echo "开始批量运行 SCALE-Sim"
echo "CFG 目录 : $CFG_DIR"
echo "Topo 文件: $TOPOLOGY_FILE"
echo "输出目录 : $OUTPUT_DIR"
echo "日志目录 : $LOG_DIR"
echo "============================================"

# ==================== 主循环 ====================
count=0
ok=0
fail=0

shopt -s nullglob
for cfg in "$CFG_DIR"/*; do
  [[ -f "$cfg" ]] || continue

  count=$((count + 1))
  cfg_name=$(basename "$cfg")
  run_name="${cfg_name%.*}"
  log_file="$LOG_DIR/${run_name}.log"

  echo
  echo "--------------------------------------------"
  echo "[$count] 运行配置: $cfg_name"
  echo "--------------------------------------------"
  echo "python3 -m scalesim.scale \\"
  echo "  -c $cfg \\"
  echo "  -t $TOPOLOGY_FILE \\"
  echo "  -p $OUTPUT_DIR \\"
  echo "  -i $WORKLOAD_TYPE"
  echo

  # 若输出目录已存在，先清理（防止旧结果污染）
  rm -rf "$OUTPUT_DIR/$run_name"

  # 执行并记录日志
  if python3 -m scalesim.scale \
      -c "$cfg" \
      -t "$TOPOLOGY_FILE" \
      -p "$OUTPUT_DIR" \
      -i "$WORKLOAD_TYPE" \
      2>&1 | tee "$log_file"; then
    echo "✓ 完成: $cfg_name"
    ok=$((ok + 1))
  else
    echo "✗ 失败: $cfg_name"
    fail=$((fail + 1))
  fi
done
shopt -u nullglob

echo
echo "============================================"
echo "所有配置运行完成"
echo "成功: $ok  失败: $fail  总数: $count"
echo "日志目录: $LOG_DIR"
echo "============================================"
