#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray
from websocket import create_connection


def _ws_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/ws"


def _send_ws(host: str, port: int, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    ws = create_connection(_ws_url(host, port), timeout=timeout)
    try:
        ws.send(json.dumps(payload))
        return json.loads(ws.recv())
    finally:
        ws.close()


def _encode_rgb_image(image_rgb: np.ndarray, quality: int = 90) -> str:
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _select_actions(actions: np.ndarray, max_actions: int, use_last_actions: bool) -> np.ndarray:
    if max_actions <= 0 or len(actions) <= max_actions:
        return actions
    return actions[-max_actions:] if use_last_actions else actions[:max_actions]


def _angle_delta(values: np.ndarray, refs: np.ndarray) -> np.ndarray:
    return (values - refs + np.pi) % (2 * np.pi) - np.pi


def _action_distance_summary(actions: np.ndarray, state: np.ndarray, limit: int = 8) -> str:
    dists = np.linalg.norm(actions[:, :3] - state[:3], axis=1)
    rpy_deltas = _angle_delta(actions[:, 3:6], state[3:6])
    rpy_dists = np.linalg.norm(rpy_deltas, axis=1)
    closest_idx = int(np.argmin(dists))
    rows = []
    for idx in range(min(len(actions), limit)):
        xyz = actions[idx, :3].round(4).tolist()
        rpy_delta = rpy_deltas[idx].round(4).tolist()
        rows.append(
            f"{idx}:pos={float(dists[idx]):.3f},"
            f"rpy={float(rpy_dists[idx]):.3f},xyz={xyz},drpy={rpy_delta}"
        )
    return (
        f"state_xyz={state[:3].round(4).tolist()} | "
        f"state_rpy={state[3:6].round(4).tolist()} | "
        f"state_gripper={float(state[6]):.3f} | "
        f"closest_pos={closest_idx}:pos={float(dists[closest_idx]):.3f},"
        f"rpy={float(rpy_dists[closest_idx]):.3f},"
        f"xyz={actions[closest_idx, :3].round(4).tolist()},"
        f"drpy={rpy_deltas[closest_idx].round(4).tolist()} | "
        f"first_{min(len(actions), limit)}=[{'; '.join(rows)}]"
    )


class MotusFrankaInferenceNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("motus_franka_inference_node")
        self.args = args
        self.ws_host = args.server_host
        self.ws_port = args.server_port
        self.should_exit = False

        if not args.dry_run and not args.confirm_ee_action_space:
            raise RuntimeError(
                "Refusing to publish actions without --confirm-ee-action-space. "
                "The connected Motus server must output absolute EE pose actions: "
                "[x, y, z, roll, pitch, yaw, gripper]."
            )

        self.latest_images: dict[str, np.ndarray | None] = {
            "top": None,
            "left_wrist": None,
            "right_wrist": None,
        }
        self.latest_ee_state: np.ndarray | None = None
        self.last_ee_state_time: float | None = None
        self.state_lock = threading.Lock()

        self.queue_is_empty = False
        self.allow_inference = True
        self.can_infer = False
        self.got_first_inference = False
        self.start_time = time.time()

        self.camera_pipelines: dict[str, rs.pipeline] = {}
        self.camera_frame_queues: dict[str, rs.frame_queue] = {}

        self._check_server()
        self._init_cameras()
        self._start_camera_threads()

        self.ee_states_sub = self.create_subscription(
            JointState, args.ee_states_topic, self.ee_states_callback, 10
        )
        self.queue_status_sub = self.create_subscription(
            Bool, args.queue_status_topic, self.queue_status_callback, 10
        )
        self.allow_inference_sub = self.create_subscription(
            Bool, args.allow_inference_topic, self.allow_inference_callback, 10
        )
        self.action_pub = self.create_publisher(Float32MultiArray, args.action_topic, 10)

        period = 1.0 / max(args.inference_frequency, 0.01)
        self.inference_timer = self.create_timer(period, self.inference_callback)

        self.get_logger().info(
            f"Motus inference node ready | server={self.ws_host}:{self.ws_port} | "
            f"max_actions={args.max_actions_to_publish} | dry_run={args.dry_run}"
        )

    def _check_server(self) -> None:
        response = _send_ws(self.ws_host, self.ws_port, {"type": "health"}, timeout=self.args.timeout)
        if response.get("type") != "health" or response.get("status") != "healthy":
            raise RuntimeError(f"Motus server is not healthy: {response}")
        self.get_logger().info(f"Motus server health: {response}")

    def _init_cameras(self) -> None:
        serials = {
            "top": self.args.top_camera_serial,
            "left_wrist": self.args.left_wrist_camera_serial,
            "right_wrist": self.args.right_wrist_camera_serial,
        }
        for name, serial in serials.items():
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color,
                self.args.image_width,
                self.args.image_height,
                rs.format.yuyv,
                30,
            )
            frame_queue = rs.frame_queue(50)
            pipeline.start(cfg, frame_queue)
            self.camera_pipelines[name] = pipeline
            self.camera_frame_queues[name] = frame_queue
            self.get_logger().info(f"Camera {name} started | serial={serial}")
            time.sleep(0.5)

    def _camera_thread(self, name: str, frame_queue: rs.frame_queue) -> None:
        height = self.args.image_height
        width = self.args.image_width
        while rclpy.ok() and not self.should_exit:
            try:
                frame = frame_queue.wait_for_frame(timeout_ms=2000)
                try:
                    color_frame = frame.as_frameset().get_color_frame()
                except Exception:
                    color_frame = frame
                if not color_frame:
                    continue
                img_yuyv = np.asanyarray(color_frame.get_data()).view(np.uint8).reshape(height, width, 2)
                img_rgb = cv2.cvtColor(img_yuyv, cv2.COLOR_YUV2RGB_YUYV)
                with self.state_lock:
                    self.latest_images[name] = img_rgb
            except Exception as exc:
                if rclpy.ok() and not self.should_exit:
                    self.get_logger().warn(f"Camera {name} frame failed: {exc}")

    def _start_camera_threads(self) -> None:
        for name, frame_queue in self.camera_frame_queues.items():
            threading.Thread(
                target=self._camera_thread,
                args=(name, frame_queue),
                daemon=True,
                name=f"camera_{name}",
            ).start()

    def ee_states_callback(self, msg: JointState) -> None:
        if len(msg.position) < 7:
            return
        with self.state_lock:
            state = np.asarray(msg.position[:7], dtype=np.float32)
            state[6] = 1.0 if state[6] <= self.args.gripper_closed_width else 0.0
            self.latest_ee_state = state
            self.last_ee_state_time = time.time()

    def queue_status_callback(self, msg: Bool) -> None:
        self.queue_is_empty = bool(msg.data)
        self.can_infer = self.queue_is_empty and self.allow_inference

    def allow_inference_callback(self, msg: Bool) -> None:
        self.allow_inference = bool(msg.data)
        self.can_infer = self.queue_is_empty and self.allow_inference

    def _get_observation(self) -> tuple[list[str], np.ndarray] | None:
        with self.state_lock:
            if any(image is None for image in self.latest_images.values()):
                return None
            if self.latest_ee_state is None:
                return None
            if self.last_ee_state_time is None:
                return None
            if time.time() - self.last_ee_state_time > self.args.max_state_age:
                return None
            images = [
                self.latest_images["top"].copy(),
                self.latest_images["left_wrist"].copy(),
                self.latest_images["right_wrist"].copy(),
            ]
            state = self.latest_ee_state.copy()

        return [_encode_rgb_image(image) for image in images], state

    def inference_callback(self) -> None:
        if time.time() - self.start_time < self.args.warmup_delay_s:
            return
        if not self.can_infer and not self.args.ignore_queue_status:
            return

        observation = self._get_observation()
        if observation is None:
            return

        images_b64, state = observation
        payload: dict[str, Any] = {
            "type": "inference",
            "images": images_b64,
            "proprio_data": [state.astype(float).tolist()],
        }
        if self.args.instruction:
            payload["instruction"] = self.args.instruction

        try:
            start = time.time()
            response = _send_ws(self.ws_host, self.ws_port, payload, timeout=self.args.timeout)
            inference_ms = (time.time() - start) * 1000.0
        except Exception as exc:
            self.get_logger().error(f"Motus inference request failed: {exc}")
            return

        if response.get("type") == "error":
            self.get_logger().error(f"Motus server error: {response.get('detail')}")
            return
        if response.get("type") != "inference":
            self.get_logger().error(f"Unexpected Motus response: {response}")
            return

        actions = np.asarray(response.get("predicted_actions", []), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
            self.get_logger().error(f"Bad action shape from Motus: {actions.shape}")
            return
        if self.args.expected_action_steps > 0 and actions.shape[0] != self.args.expected_action_steps:
            self.get_logger().warn(
                f"Unexpected Motus action steps: got {actions.shape[0]}, "
                f"expected {self.args.expected_action_steps}"
            )

        action_summary = _action_distance_summary(actions, state)
        if self.args.dry_run or self.args.log_action_summary:
            self.get_logger().info(f"Action distance summary | {action_summary}")

        actions_to_publish = _select_actions(
            actions,
            self.args.max_actions_to_publish,
            self.args.use_last_actions,
        )

        first_position_jump = float(np.linalg.norm(actions_to_publish[0, :3] - state[:3]))
        first_rpy_delta = _angle_delta(actions_to_publish[0, 3:6], state[3:6])
        first_orientation_jump = float(np.linalg.norm(first_rpy_delta))

        if self.args.max_position_jump > 0 and first_position_jump > self.args.max_position_jump:
            jump_msg = (
                f"first position jump {first_position_jump:.3f}m "
                f"> limit {self.args.max_position_jump:.3f}m | "
                f"state_xyz={state[:3].round(4).tolist()} "
                f"action_xyz={actions_to_publish[0, :3].round(4).tolist()} | "
                f"{action_summary}"
            )
            if self.args.dry_run:
                self.get_logger().warn(f"dry-run: would refuse to publish action: {jump_msg}")
            else:
                self.get_logger().error(f"Refusing to publish action: {jump_msg}")
                if self.args.once:
                    self.should_exit = True
                return
        if self.args.max_orientation_jump > 0 and first_orientation_jump > self.args.max_orientation_jump:
            jump_msg = (
                f"first orientation jump {first_orientation_jump:.3f}rad "
                f"> limit {self.args.max_orientation_jump:.3f}rad | "
                f"state_rpy={state[3:6].round(4).tolist()} "
                f"action_rpy={actions_to_publish[0, 3:6].round(4).tolist()} "
                f"drpy={first_rpy_delta.round(4).tolist()} | "
                f"{action_summary}"
            )
            if self.args.dry_run:
                self.get_logger().warn(f"dry-run: would refuse to publish action: {jump_msg}")
            else:
                self.get_logger().error(f"Refusing to publish action: {jump_msg}")
                if self.args.once:
                    self.should_exit = True
                return

        self.get_logger().info(
            f"Inference ok | server_ms={response.get('processing_time_ms', 0):.1f} | "
            f"wall_ms={inference_ms:.1f} | action_shape={actions.shape} | "
            f"publish={len(actions_to_publish)} | "
            f"first_pos_jump={first_position_jump:.3f}m | "
            f"first_rpy_jump={first_orientation_jump:.3f}rad | "
            f"first={actions_to_publish[0].round(4).tolist()}"
        )

        if not self.args.dry_run:
            for idx, action in enumerate(actions_to_publish):
                msg = Float32MultiArray()
                msg.data = action.astype(float).tolist()
                self.action_pub.publish(msg)
                if idx < len(actions_to_publish) - 1:
                    time.sleep(self.args.action_publish_interval)
            self.can_infer = False
        else:
            self.get_logger().warn("dry-run enabled: actions were not published to /franka/action_command")

        self.got_first_inference = True
        if self.args.once:
            self.should_exit = True

    def destroy_node(self) -> None:
        for pipeline in self.camera_pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        super().destroy_node()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motus WebSocket inference node for Franka control")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--timeout", type=int, default=300)

    parser.add_argument("--top-camera-serial", default="405622072640")
    parser.add_argument("--left-wrist-camera-serial", default="244222073051")
    parser.add_argument("--right-wrist-camera-serial", default="241222077298")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)

    parser.add_argument("--ee-states-topic", default="/franka/ee_states")
    parser.add_argument("--action-topic", default="/franka/action_command")
    parser.add_argument("--queue-status-topic", default="/franka/queue_status")
    parser.add_argument("--allow-inference-topic", default="/franka/allow_inference")

    parser.add_argument("--inference-frequency", type=float, default=1.0)
    parser.add_argument("--max-actions-to-publish", type=int, default=1)
    parser.add_argument("--use-last-actions", action="store_true")
    parser.add_argument("--action-publish-interval", type=float, default=0.05)
    parser.add_argument("--warmup-delay-s", type=float, default=3.0)
    parser.add_argument("--max-state-age", type=float, default=0.2)
    parser.add_argument("--gripper-closed-width", type=float, default=0.028)
    parser.add_argument("--max-position-jump", type=float, default=0.08)
    parser.add_argument("--max-orientation-jump", type=float, default=0.8)
    parser.add_argument("--expected-action-steps", type=int, default=16)
    parser.add_argument("--log-action-summary", action="store_true")
    parser.add_argument("--ignore-queue-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--confirm-ee-action-space",
        action="store_true",
        help="Required for non-dry-run publishing. Use only if the server output is confirmed to be EE [x,y,z,rpy,gripper].",
    )
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    rclpy.init()
    node = MotusFrankaInferenceNode(args)
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
