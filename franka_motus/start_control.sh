#!/bin/bash
# 启动控制节点（在新终端中运行）

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 加载配置
source "$SCRIPT_DIR/config.sh"

clear
echo -e "${CYAN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                  控制节点（Franka机器人）                  ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${YELLOW}配置：${NC}"
echo -e "  机器人IP:      ${GREEN}$ROBOT_IP${NC}"
echo -e "  动力学因子:    ${GREEN}$DYNAMICS_FACTOR${NC}"
echo -e "  控制频率:      ${GREEN}$CONTROL_FREQUENCY Hz${NC}"
echo ""

echo -e "${CYAN}准备启动控制节点...${NC}"
echo ""

# 激活环境（控制节点使用系统 python3，franka_inference_env 中的 franky 是推理框架包会冲突）
source /opt/ros/humble/setup.bash

# 进入工作目录
cd "$SCRIPT_DIR"

echo -e "${GREEN}✓ 环境已激活${NC}"
echo -e "${GREEN}✓ 开始连接机器人...${NC}"
echo ""

# 启动控制节点
python3 franka_control_node.py \
    --robot-ip="$ROBOT_IP" \
    --dynamics-factor="$DYNAMICS_FACTOR" \
    --control-frequency="$CONTROL_FREQUENCY" \
    --wait-for-motion
