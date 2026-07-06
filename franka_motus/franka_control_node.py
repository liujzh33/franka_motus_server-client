#!/usr/bin/env python3
"""
Franka控制节点（独立节点）

功能：
- 使用franky连接和控制机器人
- 订阅动作指令（来自推理节点）
- 发布机器人状态到ROS2（供推理节点使用）
- 执行动作控制

使用方法：
---------
# 1. 先启动此控制节点
python franka_control_node.py --robot-ip 172.16.0.2

# 2. 再启动推理节点（见 franka_inference_node.py）
python franka_inference_node.py \
    --policy-server-host 39.101.65.229 \
    --policy-server-port 33050
"""

import dataclasses
import time
import threading
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Bool
import tyro

import franky
from franky import Robot, CartesianMotion, Affine


@dataclasses.dataclass
class Args:
    """命令行参数"""

    robot_ip: str = "172.16.0.2"
    """Franka机器人IP地址"""

    # ROS2 topics
    ee_states_topic: str = "/franka/ee_states"
    """发布机器人EE状态的topic"""

    action_topic: str = "/franka/action_command"
    """订阅动作指令的topic"""

    queue_status_topic: str = "/franka/queue_status"
    """发布队列完成状态的topic"""

    allow_inference_topic: str = "/franka/allow_inference"
    """发布是否允许推理的topic"""

    # 控制参数
    control_frequency: float = 10.0
    """控制频率(Hz) - 降低到5Hz避免运动冲突"""

    state_publish_frequency: float = 50.0
    """状态发布频率(Hz)"""

    dynamics_factor: float = 0.1
    """动力学因子（越小越慢越安全）"""

    action_buffer_size: int = 20
    """动作缓冲区大小 - 减小避免累积过时动作"""

    wait_for_motion: bool = True
    """是否等待运动完成再发送下一个命令"""

    # Gripper参数
    gripper_speed: float = 0.3
    """夹爪速度 (m/s)"""

    gripper_force: float = 40.0
    """夹爪抓取力 (N)"""

    # 位置限制参数
    min_z: float = 0.03
    """Z轴最小位置限制 (米) - 防止末端执行器过低"""

    orientation_correction: str = "none"
    """dataset frame 到 franky command-frame 的固定姿态修正。
    新 pipeline（pose 来自 /franka_robot_state_broadcaster/current_pose）已经在 libfranka frame，无需修正，保持 none。
    仅当数据集是用 franky.Kinematics.forward 反算 RPY 的旧 stack_bowls_raw 那类时，才需要 rx_pi_rz_pos90。"""


class FrankaControlNode(Node):
    """Franka控制节点 - 只负责机器人控制，不做推理"""

    def __init__(self, args: Args):
        super().__init__('franka_control_node')
        self.args = args

        # 初始化文件日志
        self._setup_file_logging()

        # 连接机器人（使用franky）
        self.get_logger().info(f"连接Franka机器人: {args.robot_ip}")
        self.robot = Robot(args.robot_ip)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = args.dynamics_factor
        self.get_logger().info(f"机器人连接成功 (动力学因子: {args.dynamics_factor})")

        # 连接夹爪
        self.get_logger().info(f"连接夹爪: {args.robot_ip}")
        self.gripper = franky.Gripper(args.robot_ip)
        self.get_logger().info("夹爪连接成功")

        # Gripper状态记忆（避免重复执行相同指令）
        self.last_gripper_state = None  # None, 0 (打开), 或 1 (抓取)

        # 动作缓冲队列（线程安全）
        self.action_queue = deque(maxlen=args.action_buffer_size)
        self.queue_lock = threading.Lock()
        self.queue_was_empty = True  # 跟踪队列是否为空，用于避免重复发布完成状态

        # 订阅动作指令
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            args.action_topic,
            self.action_callback,
            10
        )

        # 发布机器人EE状态
        self.ee_states_pub = self.create_publisher(
            JointState,
            args.ee_states_topic,
            10
        )

        # 发布队列完成状态
        self.queue_status_pub = self.create_publisher(
            Bool,
            args.queue_status_topic,
            10
        )

        # 发布推理控制信号
        self.allow_inference_pub = self.create_publisher(
            Bool,
            args.allow_inference_topic,
            10
        )

        # 状态发布定时器
        state_period = 1.0 / args.state_publish_frequency
        self.state_timer = self.create_timer(
            state_period,
            self.publish_robot_state
        )

        # 控制线程
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_running = True
        self.control_thread.start()

        # 统计
        self.actions_received = 0
        self.actions_executed = 0

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[控制节点] 已启动 | 频率: {args.control_frequency}Hz | IP: {args.robot_ip}")
        self.get_logger().info(f"[控制节点] 最小高度: {args.min_z*100:.1f}cm | 动力学因子: {args.dynamics_factor}")
        self.get_logger().info(f"[控制节点] 姿态帧修正: {args.orientation_correction}")
        self.get_logger().info("=" * 60)
        
        # 检查话题连接（延迟检查，给ROS2时间建立连接）
        def check_topic_connection():
            time.sleep(2.0)  # 等待2秒让ROS2建立连接
            publisher_count = self.ee_states_pub.get_subscription_count()
            subscriber_count = self.count_publishers(args.action_topic)
            self.get_logger().info(f"话题连接状态: EE状态订阅者={publisher_count}, 动作发布者={subscriber_count}")
            if subscriber_count == 0:
                self.get_logger().warn(f"[连接] 未检测到动作发布者 | 话题: {args.action_topic}")
        
        # 在后台线程中检查连接
        threading.Thread(target=check_topic_connection, daemon=True).start()
        
        # 启动时如果队列为空，发布初始完成信号，让推理节点可以开始第一次推理
        def publish_initial_status():
            time.sleep(3.0)  # 等待3秒，确保ROS2连接建立
            with self.queue_lock:
                if len(self.action_queue) == 0:
                    self.get_logger().info("[启动] 发布初始完成信号")
                    # 发布多次确保收到（ROS2消息可能丢失）
                    for _ in range(3):
                        self._publish_bool(self.queue_status_pub, True)
                        self._publish_bool(self.allow_inference_pub, True)
                        time.sleep(0.1)
                    self.get_logger().info("[启动] 已发布初始允许推理信号 ✓ (发送3次确保收到)")
        
        threading.Thread(target=publish_initial_status, daemon=True).start()

    def _setup_file_logging(self):
        """设置文件日志记录"""
        # 创建logs目录
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        # 创建日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"control_{timestamp}.log"
        
        # 配置文件日志记录器
        self.file_logger = logging.getLogger("control_file")
        self.file_logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.file_logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)
        
        self.get_logger().info(f"日志文件: {log_file}")

    def _publish_bool(self, publisher, value: bool):
        """辅助方法：发布Bool消息"""
        msg = Bool()
        msg.data = value
        publisher.publish(msg)

    def action_callback(self, msg: Float32MultiArray):
        """接收动作指令"""
        try:
            action = np.array(msg.data, dtype=np.float32)
            
            # 验证动作维度
            if len(action) != 7:
                self.get_logger().error(f"收到错误维度的动作: {len(action)}维, 期望7维")
                return

            with self.queue_lock:
                was_empty = len(self.action_queue) == 0
                self.action_queue.append(action)
                self.actions_received += 1
                # 队列有新动作，更新状态
                if was_empty:
                    self.queue_was_empty = False
                    # 队列从空变为有动作，禁止推理
                    self._publish_bool(self.allow_inference_pub, False)
                    self.get_logger().info("[推理控制] 禁止推理")

                # 第一个动作立即打印
                if self.actions_received == 1:
                    self.get_logger().info(f"[动作] 首次收到 | 队列: {len(self.action_queue)}")
                # 每10个动作打印一次
                elif self.actions_received % 10 == 0:
                    self.get_logger().info(f"[动作] 收到#{self.actions_received} | 队列: {len(self.action_queue)}")
        except Exception as e:
            self.get_logger().error(f"处理动作消息失败: {e}", exc_info=True)

    RPY_WRAP_CENTERS = (math.pi, 0.0, 0.0)
    """与 convert_to_lerobot_rc*.py 同名常量保持一致；详见那两个脚本里的注释。"""

    @staticmethod
    def _wrap_to_center(value: float, center: float) -> float:
        """把单个角度值 mod 到 [center-π, center+π) 范围内。"""
        return ((value - center + math.pi) % (2 * math.pi)) - math.pi + center

    def quaternion_to_rpy(self, qx: float, qy: float, qz: float, qw: float):
        """将四元数转换为RPY角度（弧度制）
        
        Args:
            qx, qy, qz, qw: 四元数分量 (x, y, z, w)
        
        Returns:
            tuple: (roll, pitch, yaw) 角度，单位为弧度
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # 使用90度，如果超出范围
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def publish_robot_state(self):
        """发布机器人EE状态到ROS2
        
        发布格式: 7维 [x, y, z, roll, pitch, yaw, gripper]
        """
        try:
            # 获取当前笛卡尔状态
            cartesian_state = self.robot.current_cartesian_state
            robot_pose = cartesian_state.pose
            ee_pose = robot_pose.end_effector_pose  # Affine对象
            
            # 获取位置和旋转
            try:
                position = ee_pose.translation
                ee_position = [float(position[0]), float(position[1]), float(position[2])]
                quaternion = ee_pose.quaternion
                qx, qy, qz, qw = [float(q) for q in quaternion]
                # 新 pipeline 下 dataset frame == libfranka frame，本 call 默认 no-op；
                # orientation_correction != "none" 时（兼容旧数据集）才做实际修正。
                qx, qy, qz, qw = self.command_quaternion_to_dataset(qx, qy, qz, qw)
                roll, pitch, yaw = self.quaternion_to_rpy(qx, qy, qz, qw)
                # 与训练数据的 wrap 域约定保持一致（roll 中心 π，pitch/yaw 中心 0），
                # 避免 gripper 朝下时 atan2 在 ±π 边界给出 +π / -π 随机值。
                # RPY_WRAP_CENTERS 必须与 convert_to_lerobot_rc*.py 的同名常量一致。
                roll = self._wrap_to_center(roll, self.RPY_WRAP_CENTERS[0])
                pitch = self._wrap_to_center(pitch, self.RPY_WRAP_CENTERS[1])
                yaw = self._wrap_to_center(yaw, self.RPY_WRAP_CENTERS[2])
                ee_rpy = [roll, pitch, yaw]
            except (IndexError, TypeError, AttributeError, Exception) as e:
                self.get_logger().error(f"获取EE状态失败: {e}", exc_info=True)
                return
            
            # 读取夹爪状态
            try:
                gripper_width = float(self.gripper.width)
            except Exception:
                gripper_width = 0.0
            
            # 构造JointState消息（复用JointState消息类型，但内容为EE状态）
            try:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "franka_link0"
                
                # 字段名称（用于标识）
                msg.name = [
                    "ee_x", "ee_y", "ee_z",
                    "ee_roll", "ee_pitch", "ee_yaw",
                    "gripper_width"
                ]
                
                # 7维EE状态: [x, y, z, roll, pitch, yaw, gripper_width]
                ee_state = ee_position + ee_rpy + [gripper_width]
                msg.position = ee_state
                
                # 发布
                self.ee_states_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"构造或发布消息失败: {e}", exc_info=True)

        except Exception as e:
            self.get_logger().error(f"发布EE状态失败: {e}", exc_info=True)

    def rpy_to_quaternion(self, roll: float, pitch: float, yaw: float):
        """将RPY角度转换为四元数
        
        Args:
            roll, pitch, yaw: 欧拉角（弧度制）
        
        Returns:
            tuple: (qx, qy, qz, qw) 四元数
        """
        # 计算半角
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        
        # 计算四元数分量
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw

    def quaternion_multiply(self, q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    def quaternion_conjugate(self, q):
        qx, qy, qz, qw = q
        return (-qx, -qy, -qz, qw)

    def quaternion_normalize(self, q):
        qx, qy, qz, qw = q
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        return (qx / norm, qy / norm, qz / norm, qw / norm)

    def _orientation_offset_quaternion(self, mode: str):
        s45 = math.sqrt(0.5)
        offsets = {
            "none": (0.0, 0.0, 0.0, 1.0),
            "conjugate": (0.0, 0.0, 0.0, 1.0),
            "rx_pi": (1.0, 0.0, 0.0, 0.0),
            "ry_pi": (0.0, 1.0, 0.0, 0.0),
            "rz_pi": (0.0, 0.0, 1.0, 0.0),
            # rx_pi makes the tool point down; these add +/-90deg about the
            # corrected local tool-z axis to fix the remaining yaw offset.
            "rx_pi_rz_pos90": (s45, -s45, 0.0, 0.0),
            "rx_pi_rz_neg90": (s45, s45, 0.0, 0.0),
            "pre_rx_pi": (1.0, 0.0, 0.0, 0.0),
            "pre_ry_pi": (0.0, 1.0, 0.0, 0.0),
            "pre_rz_pi": (0.0, 0.0, 1.0, 0.0),
            "conjugate_rx_pi": (1.0, 0.0, 0.0, 0.0),
            "conjugate_ry_pi": (0.0, 1.0, 0.0, 0.0),
            "conjugate_rz_pi": (0.0, 0.0, 1.0, 0.0),
            "conjugate_pre_rx_pi": (1.0, 0.0, 0.0, 0.0),
            "conjugate_pre_ry_pi": (0.0, 1.0, 0.0, 0.0),
            "conjugate_pre_rz_pi": (0.0, 0.0, 1.0, 0.0),
        }
        if mode not in offsets:
            raise ValueError(
                f"未知 orientation_correction={mode!r}; "
                f"可选: {', '.join(offsets)}"
            )
        return offsets[mode]

    def _uses_pre_offset(self, mode: str):
        return mode.startswith("pre_") or mode.startswith("conjugate_pre_")

    def dataset_quaternion_to_command(self, qx: float, qy: float, qz: float, qw: float):
        """Map dataset FK-frame quaternion to franky CartesianMotion quaternion.

        数采把 franky.Kinematics.forward(...).quaternion 转成 RPY 后写入
        state/actions。这个 RPY 和 CartesianMotion(Affine(...)) 实际执行的
        command frame 差一个固定工具坐标系翻转，所以 replay/model action
        执行前必须经过这里。
        """
        mode = self.args.orientation_correction
        q = (qx, qy, qz, qw)
        if mode.startswith("conjugate"):
            q = self.quaternion_conjugate(q)
        offset = self._orientation_offset_quaternion(mode)
        if self._uses_pre_offset(mode):
            q = self.quaternion_multiply(offset, q)
        else:
            q = self.quaternion_multiply(q, offset)
        return self.quaternion_normalize(q)

    def command_quaternion_to_dataset(self, qx: float, qy: float, qz: float, qw: float):
        """Map live franky Cartesian quaternion back to the dataset FK frame.

        推理 observation/state 要保持训练分布，因此 live state 也要用和
        action 相反方向的同一个 frame correction 映射回 dataset frame。
        """
        mode = self.args.orientation_correction
        q = (qx, qy, qz, qw)
        offset = self._orientation_offset_quaternion(mode)
        inv_offset = self.quaternion_conjugate(offset)
        if self._uses_pre_offset(mode):
            q = self.quaternion_multiply(inv_offset, q)
        else:
            q = self.quaternion_multiply(q, inv_offset)
        if mode.startswith("conjugate"):
            q = self.quaternion_conjugate(q)
        return self.quaternion_normalize(q)

    def execute_action(self, action: np.ndarray):
        """执行动作 - 使用笛卡尔空间控制（EE版本）

        Args:
            action: shape (7,) [x, y, z, roll, pitch, yaw, gripper_binary]
                - x, y, z: 绝对位置（米）
                - roll, pitch, yaw: 绝对姿态（弧度）
                - gripper_binary: 浮点数，接近0=打开，接近1=抓取
        """
        try:
            action_start_time = time.time()
            action_num = self.actions_executed + 1
            
            # 提取EE位置和姿态
            target_position = action[:3].copy()  # [x, y, z] (使用copy避免修改原数组)

            target_rpy = action[3:6]  # [roll, pitch, yaw]

            original_position = target_position.copy()
            
            # 应用位置限制：Z轴最小高度限制
            z_limited = False
            if target_position[2] < self.args.min_z:
                original_z = target_position[2]
                target_position[2] = self.args.min_z
                z_limited = True
                warn_msg = f"⚠️  Z轴位置限制: {original_z:.4f}m -> {self.args.min_z:.4f}m (最小高度限制: {self.args.min_z*100:.1f}cm)"
                self.get_logger().warn(warn_msg)
                self.file_logger.warning(warn_msg)
            
            # 将RPY转换为四元数
            qx, qy, qz, qw = self.rpy_to_quaternion(
                target_rpy[0], target_rpy[1], target_rpy[2]
            )
            dataset_quat = (qx, qy, qz, qw)
            command_quat = self.dataset_quaternion_to_command(qx, qy, qz, qw)
            
            # 处理夹爪（第7维）
            # action[6] 是浮点数，二值化: 接近0=打开(0), 接近1=抓取(1)
            gripper_value = action[6]
            gripper_state = 0 if gripper_value < 0.5 else 1
            
            # 记录执行动作的详细信息
            self.get_logger().info(
                f"[执行#{action_num}] 开始执行动作 | "
                f"位置: [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}]m | "
                f"姿态: [{target_rpy[0]:.4f}, {target_rpy[1]:.4f}, {target_rpy[2]:.4f}]rad | "
                f"夹爪: {gripper_state} ({gripper_value:.4f})"
            )
            self.file_logger.info(
                f"[执行#{action_num}] 开始执行动作 | "
                f"原始动作值: {action.tolist()} | "
                f"位置: [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}]m | "
                f"姿态RPY: [{target_rpy[0]:.4f}, {target_rpy[1]:.4f}, {target_rpy[2]:.4f}]rad | "
                f"dataset四元数: [{dataset_quat[0]:.4f}, {dataset_quat[1]:.4f}, {dataset_quat[2]:.4f}, {dataset_quat[3]:.4f}] | "
                f"command四元数({self.args.orientation_correction}): "
                f"[{command_quat[0]:.4f}, {command_quat[1]:.4f}, {command_quat[2]:.4f}, {command_quat[3]:.4f}] | "
                f"夹爪状态: {gripper_state} (原始值: {gripper_value:.4f}) | "
                f"Z轴限制: {z_limited} | "
                f"等待运动完成: {self.args.wait_for_motion}"
            )
            
            # 构造Affine对象（位置 + 四元数）
            # 将数采 FK-frame RPY 转成 franky CartesianMotion/Affine 使用的
            # command frame；保留完整 RPY，只补固定 frame correction。
            qx, qy, qz, qw = command_quat
            target_affine = Affine(
                target_position.tolist(),
                [qx, qy, qz, qw]  # 四元数顺序: [x, y, z, w]
            )
            
            # 执行笛卡尔运动（绝对位置）
            motion = CartesianMotion(target_affine)
            
            # 执行运动并记录时间
            motion_start_time = time.time()
            if self.args.wait_for_motion:
                # 同步模式：等待运动完成
                self.robot.move(motion, asynchronous=False)
            else:
                # 异步模式：不等待，但可能产生运动冲突
                self.robot.move(motion, asynchronous=True)
            motion_time = (time.time() - motion_start_time) * 1000

            # 只在gripper状态变化时执行（避免重复指令冲刷）
            gripper_time = 0.0
            if gripper_state != self.last_gripper_state:
                try:
                    gripper_start_time = time.time()
                    if gripper_state == 0:
                        # 0 -> 张开夹爪
                        self.get_logger().info(f"[执行#{action_num}] [夹爪] 打开")
                        self.file_logger.info(f"[执行#{action_num}] [夹爪] 打开 | 速度: {self.args.gripper_speed}m/s")
                        self.gripper.open(speed=self.args.gripper_speed)
                    else:
                        # 1 -> 执行抓取
                        self.get_logger().info(f"[执行#{action_num}] [夹爪] 抓取中...")
                        self.file_logger.info(
                            f"[执行#{action_num}] [夹爪] 抓取中 | "
                            f"速度: {self.args.gripper_speed}m/s | "
                            f"力度: {self.args.gripper_force}N"
                        )
                        # width=0.0 表示尽可能闭合，epsilon_outer允许一定的外部误差
                        success = self.gripper.grasp(
                            width=0.0,
                            speed=self.args.gripper_speed,
                            force=self.args.gripper_force,
                            epsilon_outer=0.05
                        )
                        if success:
                            width = self.gripper.width
                            self.get_logger().info(f"[执行#{action_num}] [夹爪] 抓取成功 | 宽度: {width*1000:.1f}mm")
                            self.file_logger.info(f"[执行#{action_num}] [夹爪] 抓取成功 | 宽度: {width*1000:.1f}mm")
                        else:
                            self.get_logger().warn(f"[执行#{action_num}] [夹爪] 抓取失败")
                            self.file_logger.warning(f"[执行#{action_num}] [夹爪] 抓取失败")
                        
                        # 抓取后等待2秒
                        time.sleep(2.0)
                    
                    gripper_time = (time.time() - gripper_start_time) * 1000
                    # 更新状态记忆
                    self.last_gripper_state = gripper_state

                except Exception as gripper_error:
                    self.get_logger().error(f"[执行#{action_num}] [夹爪] 控制失败: {gripper_error}")
                    self.file_logger.error(f"[执行#{action_num}] [夹爪] 控制失败: {gripper_error}", exc_info=True)

            self.actions_executed += 1
            total_action_time = (time.time() - action_start_time) * 1000

            # 记录动作执行完成
            self.get_logger().info(
                f"[执行#{action_num}] 动作执行完成 | "
                f"运动耗时: {motion_time:.1f}ms | "
                f"夹爪耗时: {gripper_time:.1f}ms | "
                f"总耗时: {total_action_time:.1f}ms"
            )
            self.file_logger.info(
                f"[执行#{action_num}] 动作执行完成 | "
                f"运动耗时: {motion_time:.1f}ms | "
                f"夹爪耗时: {gripper_time:.1f}ms | "
                f"总耗时: {total_action_time:.1f}ms | "
                f"最终位置: [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}]m"
            )

            # 每10个动作打印一次统计
            if self.actions_executed % 10 == 0:
                self.get_logger().info(f"[执行] 已执行 {self.actions_executed} 个动作")
                self.file_logger.info(f"[执行] 统计 | 已执行: {self.actions_executed} 个动作 | 已接收: {self.actions_received} 个动作")

        except Exception as e:
            self.get_logger().error(f"[执行] 动作失败: {e}")
            self.file_logger.error(f"执行动作失败: {e}", exc_info=True)

            # 如果是Reflex错误，尝试恢复
            if "Reflex" in str(e):
                self.get_logger().warn("[恢复] 检测到Reflex模式，尝试恢复...")
                try:
                    self.robot.recover_from_errors()
                    self.get_logger().info("[恢复] 成功 ✓")
                except Exception as recover_error:
                    self.get_logger().error(f"[恢复] 失败: {recover_error}")
                    self.file_logger.error(f"恢复失败: {recover_error}", exc_info=True)

    def control_loop(self):
        """控制循环 - 在独立线程中运行"""
        control_period = 1.0 / self.args.control_frequency
        empty_loops = 0  # 连续空循环计数

        while self.control_running:
            loop_start = time.time()

            # 从队列中取出动作
            action = None
            queue_len = 0
            with self.queue_lock:
                queue_len = len(self.action_queue)
                if queue_len > 0:
                    action = self.action_queue.popleft()

            # 执行动作
            if action is not None:
                empty_loops = 0
                # 记录执行前的队列状态
                was_empty_before = self.queue_was_empty
                self.execute_action(action)
                
                # 执行动作后，检查队列是否从非空变为空
                queue_empty = False
                with self.queue_lock:
                    queue_empty = len(self.action_queue) == 0
                    self.queue_was_empty = queue_empty
                
                # 如果队列从非空变为空，等待机器人动作执行完毕
                if queue_empty and not was_empty_before:
                    self.get_logger().info("[队列] 清空 | 等待动作执行完毕...")
                    # 循环检查机器人是否还在执行动作
                    max_wait_time = 5.0  # 最大等待时间（秒）
                    check_interval = 0.1  # 检查间隔（秒）
                    start_wait_time = time.time()
                    
                    while self.robot.is_in_control:  # 注意：is_in_control 是属性，不是方法
                        elapsed = time.time() - start_wait_time
                        if elapsed > max_wait_time:
                            self.get_logger().warn(f"[队列] 等待超时 ({max_wait_time}s)，强制发布完成状态")
                            break
                        time.sleep(check_interval)
                    
                    wait_time = time.time() - start_wait_time
                    self.get_logger().info(f"[队列] 动作执行完毕 | 等待时间: {wait_time:.2f}s")
                    
                    # 发布队列完成状态和允许推理信号
                    self._publish_bool(self.queue_status_pub, True)
                    self._publish_bool(self.allow_inference_pub, True)
                    self.get_logger().info("[队列] 完成状态已发布 ✓ | [推理控制] 允许推理 ✓")
            else:
                empty_loops += 1
                # 队列为空时，定期发送完成状态（确保推理节点能收到）
                # 每10次循环（约1秒）发送一次，确保推理节点能收到
                if empty_loops % 10 == 0:
                    # 如果队列为空，持续发送完成状态
                    if self.queue_was_empty:
                        # 定期发送，确保推理节点收到
                        self._publish_bool(self.queue_status_pub, True)
                        self._publish_bool(self.allow_inference_pub, True)
                    else:
                        # 队列从非空变为空，标记并发送
                        self.queue_was_empty = True
                        self.get_logger().info("[控制] 队列为空，发布完成状态...")
                        self._publish_bool(self.queue_status_pub, True)
                        self._publish_bool(self.allow_inference_pub, True)
                        self.get_logger().info("[控制] 完成状态已发布 ✓")
                
                # 每50次空循环（约10秒）打印一次，提醒队列为空
                if empty_loops % 50 == 0:
                    self.get_logger().warn(f"[控制] 队列为空 | 已收到: {self.actions_received} | 已执行: {self.actions_executed}")

            # 控制循环频率
            elapsed = time.time() - loop_start
            if elapsed < control_period:
                time.sleep(control_period - elapsed)

    def shutdown(self):
        """清理资源"""
        self.get_logger().info("停止机器人...")
        self.file_logger.info(f"控制节点停止 - 共执行{self.actions_executed}个动作")
        self.control_running = False
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        self.robot.stop()
        self.get_logger().info("已停止")


def main(args: Args):
    """主函数"""
    rclpy.init()
    node = FrankaControlNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n收到中断信号，停止...")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(tyro.cli(Args))
