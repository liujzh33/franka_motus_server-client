#!/usr/bin/env python3
"""
Motus Inference API Server

Provides REST endpoints for real-world Motus inference without requiring a robot
environment. The implementation follows the policy-style loading and inference
flow from `inference/robotwin/Motus/deploy_policy.py` and wraps it with a
FastAPI server.
"""

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from transformers import AutoProcessor

PROJ_ROOT = str(Path(__file__).resolve().parents[3])
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

LOCAL_BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if LOCAL_BAK_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_BAK_ROOT)

ROOT_BAK_ROOT = str((Path(PROJ_ROOT) / "bak").resolve())
if ROOT_BAK_ROOT not in sys.path:
    sys.path.insert(0, ROOT_BAK_ROOT)

from models.motus import Motus, MotusConfig
from wan.modules.t5 import T5EncoderModel
from data.utils.image_utils import resize_with_padding


log = logging.getLogger("motus_api_server")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


app = FastAPI(
    title="Motus Inference API",
    description="REST API for real-world Motus model inference",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    instruction: str = Field(..., description="Task instruction used by both VLM and T5.")
    image: Optional[str] = Field(default=None, description="Base64 encoded RGB image.")
    image_path: Optional[str] = Field(default=None, description="Local path to RGB image.")
    images: Optional[List[str]] = Field(
        default=None,
        description="VLA-compatible list of base64 encoded images. Supports 1, 2, or 3 images.",
    )
    image_paths: Optional[List[str]] = Field(
        default=None,
        description="List of local image paths. Supports 1, 2, or 3 images.",
    )
    state: Optional[List[float]] = Field(default=None, description="Robot state vector. Defaults to zeros.")
    proprio_data: Optional[List[List[float]]] = Field(
        default=None,
        description="VLA-compatible proprio input. Expected shape [[state_dim]].",
    )
    t5_embeddings: Optional[Any] = Field(
        default=None,
        description="Pre-encoded T5 embeddings. Supports [[...]] or [[[...]], ...] JSON structure.",
    )
    t5_embeddings_path: Optional[str] = Field(
        default=None,
        description="Local path to a .pt file containing pre-encoded T5 embeddings.",
    )
    t5_embeddings_dir: Optional[str] = Field(
        default=None,
        description="Directory used for automatic T5 embedding lookup by instruction.",
    )
    auto_find_t5_embeddings: bool = Field(
        default=True,
        description="Automatically search a T5 embedding directory before falling back to online T5.",
    )
    num_inference_steps: Optional[int] = Field(default=None, ge=1, description="Overrides YAML inference steps.")
    instruction_prefix: Optional[str] = Field(
        default=None,
        description="Optional prefix prepended before instruction when computing T5 embeddings and VLM input.",
    )
    return_frame_grid: bool = Field(default=False, description="If true, return a base64 PNG frame grid.")
    save_output_path: Optional[str] = Field(default=None, description="Optional output path for the frame grid PNG.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    t5_loaded: bool
    device: str
    timestamp: str


class InferenceResponse(BaseModel):
    instruction: str
    effective_instruction: str
    predicted_actions: List[List[float]]
    action_head_type: str
    action_shape: List[int]
    predicted_frames_shape: List[int]
    frame_grid_image: Optional[str]
    processing_time_ms: float
    model_info: Dict[str, Any]
    timestamp: str


class ServerState:
    def __init__(self) -> None:
        self.policy: Optional["RealWorldMotusPolicy"] = None
        self.model: Optional[Motus] = None
        self.processor: Optional[AutoProcessor] = None
        self.t5_encoder: Optional[T5EncoderModel] = None
        self.config_dict: Optional[Dict[str, Any]] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_steps: int = 10
        self.lock = threading.Lock()
        self.model_config_path: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self.wan_path: Optional[str] = None
        self.vlm_path: Optional[str] = None
        self.use_t5: bool = False
        self.default_instruction_prefix: str = ""
        self.t5_embeddings_dir: Optional[str] = None


SERVER_STATE = ServerState()
DEFAULT_SCENE_PREFIX = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: "
)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def decode_base64_image(image_base64: str) -> Image.Image:
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def resize_image_with_padding(image: Image.Image, size_hw: tuple[int, int]) -> Image.Image:
    image_np = np.array(image).astype(np.uint8)
    resized_np = resize_with_padding(image_np, size_hw)
    return Image.fromarray(resized_np)


def image_to_tensor(image: Image.Image, size_hw: tuple[int, int]) -> torch.Tensor:
    image = resize_image_with_padding(image, size_hw)
    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)


def compose_multiview_image(images: List[Image.Image]) -> Image.Image:
    if len(images) == 1:
        return images[0].convert("RGB")

    if len(images) not in [2, 3]:
        raise ValueError(f"Expected 1, 2, or 3 images, got {len(images)}.")

    top = np.array(images[0].convert("RGB")).astype(np.uint8)
    top_h, top_w = top.shape[:2]
    bottom_h = max(1, top_h // 2)
    left_w = max(1, top_w // 2)
    right_w = top_w - left_w

    if len(images) == 2:
        left_img = np.array(images[1].convert("RGB")).astype(np.uint8)
        right_img = left_img
    else:
        left_img = np.array(images[1].convert("RGB")).astype(np.uint8)
        right_img = np.array(images[2].convert("RGB")).astype(np.uint8)

    left_resized = np.array(Image.fromarray(left_img).resize((left_w, bottom_h), Image.BICUBIC))
    right_resized = np.array(Image.fromarray(right_img).resize((right_w, bottom_h), Image.BICUBIC))
    bottom_row = np.concatenate([left_resized, right_resized], axis=1)
    composed = np.concatenate([top, bottom_row], axis=0)
    return Image.fromarray(composed)


def build_vlm_inputs(
    processor: AutoProcessor,
    instruction: str,
    image: Image.Image,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    encoded = processor(text=[text], images=[image], return_tensors="pt")

    vlm_inputs: Dict[str, torch.Tensor] = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "pixel_values": encoded["pixel_values"].to(device),
    }
    if encoded.get("image_grid_thw") is not None:
        vlm_inputs["image_grid_thw"] = encoded["image_grid_thw"].to(device)
    return vlm_inputs


def resolve_wan_model_dir(path: str) -> str:
    raw_path = Path(path).expanduser().resolve()
    if (raw_path / "Wan2.2_VAE.pth").exists():
        return str(raw_path)

    nested_path = raw_path / "Wan2.2-TI2V-5B"
    if (nested_path / "Wan2.2_VAE.pth").exists():
        return str(nested_path)

    raise ValueError(
        f"Cannot resolve WAN model directory from {path}. Expected Wan2.2_VAE.pth in the directory "
        "or in a Wan2.2-TI2V-5B subdirectory."
    )


class RealWorldMotusPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        wan_path: str,
        vlm_path: str,
        device: torch.device,
        use_t5: bool,
    ) -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = resolve_wan_model_dir(wan_path)
        self.vlm_path = str(Path(vlm_path).expanduser().resolve())
        self.use_t5 = use_t5
        self.config_path = str(Path(config_path).expanduser().resolve())
        self.config_dict = load_yaml_config(self.config_path)

        self.model = self._load_model()
        self.t5_encoder = self._load_t5_encoder() if use_t5 else None
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)

    def _create_model_config(self) -> MotusConfig:
        common = self.config_dict["common"]
        model_cfg = self.config_dict["model"]
        hidden_size = model_cfg["action_expert"]["hidden_size"]
        ffn_multiplier = model_cfg["action_expert"]["ffn_dim_multiplier"]

        return MotusConfig(
            wan_checkpoint_path=self.wan_path,
            vae_path=os.path.join(self.wan_path, "Wan2.2_VAE.pth"),
            wan_config_path=self.wan_path,
            video_precision="bfloat16",
            vlm_checkpoint_path=self.vlm_path,
            und_expert_hidden_size=model_cfg.get("und_expert", {}).get("hidden_size", 512),
            und_expert_ffn_dim_multiplier=model_cfg.get("und_expert", {}).get("ffn_dim_multiplier", 4),
            und_expert_norm_eps=model_cfg.get("und_expert", {}).get("norm_eps", 1e-5),
            und_layers_to_extract=None,
            vlm_adapter_input_dim=model_cfg.get("und_expert", {}).get("vlm", {}).get("input_dim", 2048),
            vlm_adapter_projector_type=model_cfg.get("und_expert", {}).get("vlm", {}).get("projector_type", "mlp3x_silu"),
            num_layers=30,
            action_state_dim=common["state_dim"],
            action_dim=common["action_dim"],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=model_cfg["action_expert"].get("norm_eps", 1e-6),
            global_downsample_rate=common["global_downsample_rate"],
            video_action_freq_ratio=common["video_action_freq_ratio"],
            num_video_frames=common["num_video_frames"],
            video_loss_weight=model_cfg["loss_weights"]["video_loss_weight"],
            action_loss_weight=model_cfg["loss_weights"]["action_loss_weight"],
            batch_size=1,
            video_height=common["video_height"],
            video_width=common["video_width"],
            load_pretrained_backbones=False,
            training_mode="finetune",
        )

    def _load_model(self) -> Motus:
        log.info("Initializing Motus model from deploy_policy-style config")
        model = Motus(self._create_model_config()).to(self.device)
        model.load_checkpoint(self.checkpoint_path, strict=False)
        model.eval()
        log.info("Loaded Motus checkpoint from %s", self.checkpoint_path)
        return model

    def _load_t5_encoder(self) -> T5EncoderModel:
        t5_ckpt = os.path.join(self.wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
        t5_tokenizer = os.path.join(self.wan_path, "google", "umt5-xxl")
        log.info("Loading T5 encoder from %s", t5_ckpt)
        return T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device=str(self.device),
            checkpoint_path=t5_ckpt,
            tokenizer_path=t5_tokenizer,
        )

    def predict(
        self,
        input_image: Image.Image,
        state: torch.Tensor,
        instruction: str,
        num_inference_steps: int,
        language_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        common = self.config_dict["common"]
        image_hw = (common["video_height"], common["video_width"])
        resized_pil = resize_image_with_padding(input_image, image_hw)
        first_frame = image_to_tensor(input_image, image_hw).to(self.device)
        vlm_inputs = build_vlm_inputs(self.vlm_processor, instruction, resized_pil, self.device)

        if language_embeddings is None:
            if self.t5_encoder is None:
                raise ValueError(
                    "No T5 encoder loaded. Start server with --use_t5, provide t5_embeddings / "
                    "t5_embeddings_path, or configure a searchable t5_embeddings_dir."
                )
            t5_out = self.t5_encoder([instruction], str(self.device))
            if isinstance(t5_out, torch.Tensor):
                if t5_out.dim() == 3:
                    language_embeddings = [t5_out.squeeze(0)]
                else:
                    language_embeddings = [t5_out]
            elif isinstance(t5_out, list):
                language_embeddings = t5_out
            else:
                raise ValueError("Unexpected T5 encoder output format.")

        with torch.inference_mode():
            result = self.model.inference_step(
                first_frame=first_frame,
                state=state,
                num_inference_steps=num_inference_steps,
                language_embeddings=language_embeddings,
                vlm_inputs=[vlm_inputs],
            )
            # Handle different return tuple sizes - always return (frames, actions)
            if isinstance(result, tuple) and len(result) >= 2:
                return result[0], result[1]  # (frames, actions), discard progress/subtask
            return result


def get_embedding_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def parse_t5_embeddings(payload: Any, device: torch.device) -> List[torch.Tensor]:
    if isinstance(payload, torch.Tensor):
        return [payload.to(device=device, dtype=get_embedding_dtype(device))]

    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("t5_embeddings must be a non-empty list or tensor-like object.")

    if isinstance(payload[0], list) and payload[0] and isinstance(payload[0][0], (int, float)):
        tensor = torch.tensor(payload, dtype=get_embedding_dtype(device), device=device)
        return [tensor]

    if isinstance(payload[0], list) and payload[0] and isinstance(payload[0][0], list):
        tensors = []
        for item in payload:
            tensors.append(torch.tensor(item, dtype=get_embedding_dtype(device), device=device))
        return tensors

    raise ValueError("Unsupported t5_embeddings format.")


def load_t5_embeddings_from_path(path: str, device: torch.device) -> List[torch.Tensor]:
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, torch.Tensor):
        return [loaded.to(device=device, dtype=get_embedding_dtype(device))]
    if isinstance(loaded, list):
        return [tensor.to(device=device, dtype=get_embedding_dtype(device)) for tensor in loaded]
    raise ValueError("Unsupported T5 embedding file format, expected Tensor or List[Tensor].")


def slugify_instruction(text: str, max_length: int = 120) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    if not slug:
        slug = "instruction"
    return slug[:max_length]


def get_t5_index_candidates(directory: Path) -> List[Path]:
    return [
        directory / "t5_embeddings_index.json",
        directory / "instruction_to_t5.json",
        directory / "index.json",
    ]


def get_t5_filename_candidates(text: str) -> List[str]:
    slug = slugify_instruction(text)
    sha256_name = hashlib.sha256(text.encode("utf-8")).hexdigest()
    md5_name = hashlib.md5(text.encode("utf-8")).hexdigest()
    return [
        f"{slug}.pt",
        f"{slug}.pth",
        f"{slug}.embed.pt",
        f"{slug}.embedding.pt",
        f"{sha256_name}.pt",
        f"{md5_name}.pt",
    ]


def search_t5_embedding_path(directory: str, instruction_candidates: List[str]) -> Optional[Path]:
    base_dir = Path(directory)
    if not base_dir.exists() or not base_dir.is_dir():
        raise ValueError(f"T5 embedding directory does not exist or is not a directory: {directory}")

    for index_path in get_t5_index_candidates(base_dir):
        if not index_path.exists():
            continue
        with open(index_path, "r") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            continue
        for instruction in instruction_candidates:
            mapped_path = mapping.get(instruction)
            if mapped_path:
                resolved = Path(mapped_path)
                if not resolved.is_absolute():
                    resolved = (base_dir / mapped_path).resolve()
                if resolved.exists():
                    return resolved

    seen_names = set()
    for instruction in instruction_candidates:
        for filename in get_t5_filename_candidates(instruction):
            if filename in seen_names:
                continue
            seen_names.add(filename)
            direct_path = base_dir / filename
            if direct_path.exists():
                return direct_path

    pt_files = sorted(base_dir.rglob("*.pt"))
    slug_candidates = {slugify_instruction(text) for text in instruction_candidates if text.strip()}
    for path in pt_files:
        stem_lower = path.stem.lower()
        for slug in slug_candidates:
            if slug and slug in stem_lower:
                return path

    return None


def get_effective_instruction(instruction: str, instruction_prefix: Optional[str]) -> str:
    prefix = instruction_prefix if instruction_prefix is not None else SERVER_STATE.default_instruction_prefix
    prefix = prefix or ""
    return f"{prefix}{instruction}"


def resolve_request_language_embeddings(
    request: InferenceRequest,
    effective_instruction: str,
) -> Optional[List[torch.Tensor]]:
    if request.t5_embeddings is not None:
        return parse_t5_embeddings(request.t5_embeddings, SERVER_STATE.device)

    if request.t5_embeddings_path:
        return load_t5_embeddings_from_path(request.t5_embeddings_path, SERVER_STATE.device)

    if request.auto_find_t5_embeddings:
        t5_embeddings_dir = request.t5_embeddings_dir or SERVER_STATE.t5_embeddings_dir
        if t5_embeddings_dir:
            instruction_candidates = []
            for text in [effective_instruction, request.instruction]:
                if text and text not in instruction_candidates:
                    instruction_candidates.append(text)

            found_path = search_t5_embedding_path(t5_embeddings_dir, instruction_candidates)
            if found_path is not None:
                log.info("Auto-loaded T5 embeddings from %s", found_path)
                return load_t5_embeddings_from_path(str(found_path), SERVER_STATE.device)

    return None


def get_input_image(request: InferenceRequest) -> Image.Image:
    if request.images is not None:
        decoded_images = [decode_base64_image(image_b64) for image_b64 in request.images]
        return compose_multiview_image(decoded_images)
    if request.image_paths is not None:
        loaded_images = [Image.open(path).convert("RGB") for path in request.image_paths]
        return compose_multiview_image(loaded_images)
    if request.image is not None:
        return decode_base64_image(request.image)
    if request.image_path is not None:
        return Image.open(request.image_path).convert("RGB")
    raise ValueError("Provide image/image_path or VLA-compatible images/image_paths.")


def get_state_values(request: InferenceRequest, state_dim: int) -> Optional[List[float]]:
    if request.state is not None:
        return request.state

    if request.proprio_data is not None:
        if len(request.proprio_data) == 0:
            return None
        first_row = request.proprio_data[0]
        if len(first_row) != state_dim:
            raise ValueError(f"Proprio length mismatch: expected {state_dim}, got {len(first_row)}.")
        return [float(x) for x in first_row]

    return None


def get_state_tensor(state_values: Optional[List[float]], state_dim: int, device: torch.device) -> torch.Tensor:
    state_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if state_values is None:
        return torch.zeros((1, state_dim), dtype=state_dtype, device=device)
    if len(state_values) != state_dim:
        raise ValueError(f"State length mismatch: expected {state_dim}, got {len(state_values)}.")
    return torch.tensor(state_values, dtype=state_dtype, device=device).unsqueeze(0)


def frames_to_tchw(predicted_frames: torch.Tensor) -> torch.Tensor:
    if predicted_frames.dim() != 5:
        raise ValueError(f"Unexpected predicted_frames shape: {tuple(predicted_frames.shape)}")

    if predicted_frames.shape[1] == 3:
        return predicted_frames.permute(0, 2, 1, 3, 4).squeeze(0)
    return predicted_frames.squeeze(0)


def render_frame_grid(condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
    condition_np = (
        condition_frame.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).astype(np.uint8)

    frame_images = [condition_np]
    for frame in predicted_frames:
        frame_np = (frame.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        frame_images.append(frame_np)

    grid = np.concatenate(frame_images, axis=1)
    return Image.fromarray(grid)


def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_model_inference(request: InferenceRequest) -> InferenceResponse:
    if SERVER_STATE.policy is None or SERVER_STATE.config_dict is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    cfg = SERVER_STATE.config_dict
    common = cfg["common"]
    num_inference_steps = request.num_inference_steps or SERVER_STATE.default_steps
    effective_instruction = get_effective_instruction(request.instruction, request.instruction_prefix)

    start_time = datetime.now()

    input_image = get_input_image(request)
    state_values = get_state_values(request, int(common["state_dim"]))
    state = get_state_tensor(state_values, int(common["state_dim"]), SERVER_STATE.device)
    language_embeddings = resolve_request_language_embeddings(request, effective_instruction)

    with SERVER_STATE.lock:
        predicted_frames, predicted_actions = SERVER_STATE.policy.predict(
            input_image=input_image,
            state=state,
            instruction=effective_instruction,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
        )

    image_hw = (common["video_height"], common["video_width"])
    first_frame = image_to_tensor(input_image, image_hw).to(SERVER_STATE.device)

    end_time = datetime.now()
    processing_time_ms = (end_time - start_time).total_seconds() * 1000.0

    frames_tchw = frames_to_tchw(predicted_frames)
    frame_grid = render_frame_grid(first_frame.squeeze(0), frames_tchw)
    if request.save_output_path:
        output_path = Path(request.save_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_grid.save(output_path)

    predicted_actions_cpu = predicted_actions.detach().cpu().float()
    if predicted_actions_cpu.dim() == 3:
        predicted_actions_cpu = predicted_actions_cpu.squeeze(0)
    if predicted_actions_cpu.dim() == 1:
        predicted_actions_cpu = predicted_actions_cpu.unsqueeze(0)
    frame_grid_b64 = image_to_base64_png(frame_grid) if request.return_frame_grid else None

    return InferenceResponse(
        instruction=request.instruction,
        effective_instruction=effective_instruction,
        predicted_actions=predicted_actions_cpu.tolist(),
        action_head_type="motus_diffusion",
        action_shape=list(predicted_actions_cpu.shape),
        predicted_frames_shape=list(frames_tchw.shape),
        frame_grid_image=frame_grid_b64,
        processing_time_ms=processing_time_ms,
        model_info={
            "device": str(SERVER_STATE.device),
            "num_inference_steps": num_inference_steps,
            "state_dim": int(common["state_dim"]),
            "action_dim": int(common["action_dim"]),
            "video_size": [int(common["video_height"]), int(common["video_width"])],
            "num_video_frames": int(common["num_video_frames"]),
            "video_action_freq_ratio": int(common["video_action_freq_ratio"]),
            "t5_mode": "online" if SERVER_STATE.t5_encoder is not None else "request_provided",
            "policy_style": "deploy_policy",
        },
        timestamp=end_time.isoformat(),
    )


def load_server_components(
    model_config_path: str,
    ckpt_dir: str,
    wan_path: str,
    use_t5: bool,
    device: Optional[str],
    instruction_prefix: str,
    t5_embeddings_dir: Optional[str],
    vlm_path: Optional[str],
) -> None:
    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config_dict = load_yaml_config(model_config_path)
    resolved_vlm_path = vlm_path or config_dict["model"]["vlm"]["checkpoint_path"]

    policy = RealWorldMotusPolicy(
        checkpoint_path=ckpt_dir,
        config_path=model_config_path,
        wan_path=wan_path,
        vlm_path=resolved_vlm_path,
        device=resolved_device,
        use_t5=use_t5,
    )

    SERVER_STATE.policy = policy
    SERVER_STATE.model = policy.model
    SERVER_STATE.processor = policy.vlm_processor
    SERVER_STATE.t5_encoder = policy.t5_encoder
    SERVER_STATE.config_dict = config_dict
    SERVER_STATE.device = resolved_device
    SERVER_STATE.default_steps = int(config_dict["model"]["inference"]["num_inference_timesteps"])
    SERVER_STATE.model_config_path = model_config_path
    SERVER_STATE.checkpoint_path = ckpt_dir
    SERVER_STATE.wan_path = policy.wan_path
    SERVER_STATE.vlm_path = policy.vlm_path
    SERVER_STATE.use_t5 = use_t5
    SERVER_STATE.default_instruction_prefix = instruction_prefix
    SERVER_STATE.t5_embeddings_dir = t5_embeddings_dir


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Motus Inference API Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model_info",
            "inference": "/inference",
            "inference_upload": "/inference/upload",
        },
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=SERVER_STATE.model is not None,
        t5_loaded=SERVER_STATE.t5_encoder is not None,
        device=str(SERVER_STATE.device),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model_info")
def model_info() -> Dict[str, Any]:
    if SERVER_STATE.model is None or SERVER_STATE.config_dict is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    common = SERVER_STATE.config_dict["common"]
    model_cfg = SERVER_STATE.config_dict["model"]
    return {
        "model_loaded": True,
        "device": str(SERVER_STATE.device),
        "checkpoint_path": SERVER_STATE.checkpoint_path,
        "model_config_path": SERVER_STATE.model_config_path,
        "wan_path": SERVER_STATE.wan_path,
        "vlm_path": SERVER_STATE.vlm_path,
        "t5_loaded": SERVER_STATE.t5_encoder is not None,
        "t5_embeddings_dir": SERVER_STATE.t5_embeddings_dir,
        "default_inference_steps": SERVER_STATE.default_steps,
        "default_instruction_prefix": SERVER_STATE.default_instruction_prefix,
        "common": {
            "state_dim": int(common["state_dim"]),
            "action_dim": int(common["action_dim"]),
            "video_height": int(common["video_height"]),
            "video_width": int(common["video_width"]),
            "num_video_frames": int(common["num_video_frames"]),
            "video_action_freq_ratio": int(common["video_action_freq_ratio"]),
        },
        "model": {
            "vlm_checkpoint_path": SERVER_STATE.vlm_path or model_cfg["vlm"]["checkpoint_path"],
            "wan_checkpoint_path": SERVER_STATE.wan_path or model_cfg["wan"]["checkpoint_path"],
            "wan_vae_path": os.path.join(SERVER_STATE.wan_path, "Wan2.2_VAE.pth")
            if SERVER_STATE.wan_path
            else model_cfg["wan"]["vae_path"],
            "precision": "bfloat16",
        },
        "config": {
            "action_head": "motus_diffusion",
            "use_proprio": True,
            "vision_backbone": SERVER_STATE.vlm_path or model_cfg["vlm"]["checkpoint_path"],
        },
    }


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest) -> InferenceResponse:
    try:
        return run_model_inference(request)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/inference/mock")
def inference_mock() -> Dict[str, Any]:
    common = SERVER_STATE.config_dict["common"] if SERVER_STATE.config_dict is not None else {}
    action_chunk_size = int(common.get("num_video_frames", 8)) * int(common.get("video_action_freq_ratio", 6))
    action_dim = int(common.get("action_dim", 14))
    predicted_actions = np.random.randn(action_chunk_size, action_dim).astype(np.float32).tolist()
    now = datetime.now().isoformat()
    return {
        "predicted_actions": predicted_actions,
        "instruction": "mock inference",
        "action_head_type": "mock",
        "processing_time_ms": 0.0,
        "model_info": {
            "device": str(SERVER_STATE.device),
        },
        "timestamp": now,
    }


@app.post("/inference/upload", response_model=InferenceResponse)
async def inference_upload(
    instruction: str = Form(...),
    image: UploadFile = File(...),
    state_json: Optional[str] = Form(default=None),
    t5_embeddings_path: Optional[str] = Form(default=None),
    t5_embeddings_dir: Optional[str] = Form(default=None),
    auto_find_t5_embeddings: bool = Form(default=True),
    num_inference_steps: Optional[int] = Form(default=None),
    instruction_prefix: Optional[str] = Form(default=None),
    return_frame_grid: bool = Form(default=False),
    save_output_path: Optional[str] = Form(default=None),
) -> InferenceResponse:
    try:
        image_bytes = await image.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        state = json.loads(state_json) if state_json else None
        request = InferenceRequest(
            instruction=instruction,
            image=image_b64,
            state=state,
            t5_embeddings_path=t5_embeddings_path,
            t5_embeddings_dir=t5_embeddings_dir,
            auto_find_t5_embeddings=auto_find_t5_embeddings,
            num_inference_steps=num_inference_steps,
            instruction_prefix=instruction_prefix,
            return_frame_grid=return_frame_grid,
            save_output_path=save_output_path,
        )
        return run_model_inference(request)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Upload inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motus Inference API Server")
    parser.add_argument(
        "--model_config",
        required=True,
        help="Path to real-world YAML config, e.g. inference/real_world/Motus/utils/ac_one.yaml",
    )
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Path to checkpoint directory containing the Motus weights.",
    )
    parser.add_argument(
        "--wan_path",
        required=True,
        help="WAN model path. Supports either the Wan2.2-TI2V-5B directory or its parent directory.",
    )
    parser.add_argument(
        "--vlm_path",
        default=None,
        help="Optional VLM path override. Defaults to model.vlm.checkpoint_path from YAML.",
    )
    parser.add_argument("--use_t5", action="store_true", help="Load T5 encoder and encode instruction online.")
    parser.add_argument(
        "--t5_embeddings_dir",
        default=None,
        help="Optional directory for automatic T5 embedding lookup by instruction.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--instruction_prefix",
        default=DEFAULT_SCENE_PREFIX,
        help="Instruction prefix prepended before VLM/T5 preprocessing. Defaults to deploy_policy scene prefix.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", default=6789, type=int, help="Server port.")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload in development.")
    parser.add_argument("--workers", default=1, type=int, help="Number of worker processes.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_argparser().parse_args()

    load_server_components(
        model_config_path=args.model_config,
        ckpt_dir=args.ckpt_dir,
        wan_path=args.wan_path,
        use_t5=args.use_t5,
        device=args.device,
        instruction_prefix=args.instruction_prefix,
        t5_embeddings_dir=args.t5_embeddings_dir,
        vlm_path=args.vlm_path,
    )

    if args.reload:
        log.warning("--reload is disabled because the model is loaded into the current process.")
    if args.workers != 1:
        log.warning("workers>1 is not supported with in-process model loading. Forcing workers=1.")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
