"""
ComfyUI-AudioX
==============
ComfyUI custom nodes for AudioX / AudioX-MAF / AudioX-MAF-MMDiT.
Core audiox source is embedded — no 'pip install audiox' required.

Nodes
-----
  AudioX Model Loader          : Load a local AudioX model
  AudioX Video to Audio        : Generate audio from a VIDEO input
  AudioX Images to Audio (VHS) : Generate audio from an IMAGE sequence

Model directory layout
----------------------
  ComfyUI/models/AudioX/
  ├── clip-vit-base-patch32/          (shared CLIP, recommended)
  │   ├── config.json
  │   └── pytorch_model.bin
  ├── AudioX/
  │   ├── config.json
  │   └── model.ckpt
  ├── AudioX-MAF/                     (recommended)
  │   ├── config.json
  │   ├── model.ckpt
  │   └── synchformer_state_dict.pth
  └── AudioX-MAF-MMDiT/
      ├── config.json
      ├── model.ckpt
      └── synchformer_state_dict.pth

Download commands
-----------------
  huggingface-cli download HKUSTAudio/AudioX-MAF \
      --local-dir "ComfyUI/models/AudioX/AudioX-MAF"
  huggingface-cli download openai/clip-vit-base-patch32 \
      --local-dir "ComfyUI/models/AudioX/clip-vit-base-patch32"
"""

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import folder_paths
import numpy as np
import torch
from einops import rearrange

# ---------------------------------------------------------------------------
# Prepend node directory to sys.path so the embedded audiox package is found
# before any other audiox installation on the system.
# ---------------------------------------------------------------------------
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

from audiox.data.utils import read_video
from audiox.inference.generation import generate_diffusion_cond
from audiox.models.factory import create_model_from_config
from audiox.models.utils import load_ckpt_state_dict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIOX_MODELS_DIR = os.path.join(folder_paths.models_dir, "AudioX")
os.makedirs(AUDIOX_MODELS_DIR, exist_ok=True)

_SUPPORTED_MODELS = ["AudioX", "AudioX-MAF", "AudioX-MAF-MMDiT"]

# ---------------------------------------------------------------------------
# Model cache — one model kept in GPU memory; reloads only on name change
# ---------------------------------------------------------------------------
_cached_model:       object = None
_cached_model_name:  str    = None
_cached_config:      dict   = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _model_dir(model_name: str) -> str:
    return os.path.join(AUDIOX_MODELS_DIR, model_name)


def _check_model_files(model_name: str) -> tuple:
    """Return (ok: bool, error_message: str)."""
    d = _model_dir(model_name)
    missing = [f for f in ("model.ckpt", "config.json")
               if not os.path.exists(os.path.join(d, f))]
    if "MAF" in model_name:
        if not os.path.exists(os.path.join(d, "synchformer_state_dict.pth")):
            missing.append("synchformer_state_dict.pth")
    if missing:
        return False, (
            f"[AudioX] Missing model files: {missing}\n"
            f"Target directory: {d}\n"
            f"Download command: huggingface-cli download HKUSTAudio/{model_name} "
            f"--local-dir \"{d}\""
        )
    return True, ""


def _scan_available_models() -> list:
    """Return locally available model names, or all names if none are found."""
    available = [n for n in _SUPPORTED_MODELS if _check_model_files(n)[0]]
    return available if available else _SUPPORTED_MODELS


def _set_clip_env(model_name: str) -> None:
    """
    Set AUDIOX_CLIP_MODEL_PATH so conditioners.py loads CLIP from disk
    instead of downloading it.

    Search order:
      1. <models>/AudioX/<model_name>/clip-vit-base-patch32/
      2. <models>/AudioX/clip-vit-base-patch32/   (shared across models)
    """
    candidates = [
        os.path.join(_model_dir(model_name), "clip-vit-base-patch32"),
        os.path.join(AUDIOX_MODELS_DIR, "clip-vit-base-patch32"),
    ]
    for p in candidates:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            os.environ["AUDIOX_CLIP_MODEL_PATH"] = p
            print(f"[AudioX] Using local CLIP model: {p}")
            return
    os.environ.pop("AUDIOX_CLIP_MODEL_PATH", None)
    print(
        "[AudioX] Local CLIP model not found — will download from HuggingFace.\n"
        f"[AudioX] To avoid repeated downloads, run:\n"
        f"  huggingface-cli download openai/clip-vit-base-patch32 "
        f"--local-dir \"{candidates[1]}\""
    )


def _load_model(model_name: str):
    """Load the requested model, returning (model, config). Results are cached."""
    global _cached_model, _cached_model_name, _cached_config

    if _cached_model_name == model_name and _cached_model is not None:
        print(f"[AudioX] Using cached model: {model_name}")
        return _cached_model, _cached_config

    ok, msg = _check_model_files(model_name)
    if not ok:
        raise RuntimeError(msg)

    _set_clip_env(model_name)

    config_path = os.path.join(_model_dir(model_name), "config.json")
    ckpt_path   = os.path.join(_model_dir(model_name), "model.ckpt")

    print(f"[AudioX] Loading model: {model_name}")
    with open(config_path) as f:
        model_config = json.load(f)

    model  = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(ckpt_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()

    _cached_model      = model
    _cached_model_name = model_name
    _cached_config     = model_config
    print(f"[AudioX] Model ready: {model_name} ({device})")
    return model, model_config


def _resolve_video_path(video) -> str:
    """
    Resolve a ComfyUI VIDEO value to a filesystem path string.

    Handles multiple representations used across ComfyUI versions:
      - str or pathlib.Path
      - dict with path/video_path/filename/url key
      - comfy_api VideoFromFile: get_stream_source(), name-mangled
        _VideoFromFile__file attribute, or save_to() fallback
    """
    if isinstance(video, (str, pathlib.Path)):
        return str(video)

    if isinstance(video, dict):
        for key in ("path", "video_path", "filename", "url"):
            if video.get(key):
                return str(video[key])

    # comfy_api.latest VideoFromFile — try in order of preference

    if callable(getattr(video, "get_stream_source", None)):
        try:
            src = video.get_stream_source()
            if src and os.path.exists(str(src)):
                return str(src)
        except Exception:
            pass

    private = getattr(video, "_VideoFromFile__file", None)
    if private is not None and os.path.exists(str(private)):
        return str(private)

    if callable(getattr(video, "save_to", None)):
        try:
            tmp = os.path.join(tempfile.gettempdir(), f"audiox_{id(video)}.mp4")
            video.save_to(tmp)
            if os.path.exists(tmp):
                return tmp
        except Exception as e:
            print(f"[AudioX] VideoFromFile.save_to() failed: {e}")

    raise RuntimeError(
        f"[AudioX] Cannot resolve video path from {type(video)}.\n"
        f"Attributes: {[a for a in dir(video) if not a.startswith('__')]}"
    )


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using decord."""
    try:
        from decord import VideoReader
        from decord import cpu as dcpu
        vr = VideoReader(video_path, ctx=dcpu(0))
        duration = round(len(vr) / vr.get_avg_fps(), 3)
        del vr
        return duration
    except Exception as e:
        print(f"[AudioX] Could not read video duration ({e}), defaulting to 10s")
        return 10.0


def _encode_synchformer(video_path: str, synchformer_path: str, device: str):
    """
    Run the Synchformer visual encoder required by AudioX-MAF variants.
    Input is always resampled to 10 s at 25 fps to match training conditions.
    """
    from torchvision.transforms import v2
    from audiox.models.synchformer.features_utils import FeaturesUtils

    extractor = FeaturesUtils(
        tod_vae_ckpt="",
        enable_conditions=True,
        bigvgan_vocoder_ckpt="",
        synchformer_ckpt=synchformer_path,
        mode="44k",
    ).eval().to(device)

    frames = read_video(video_path, seek_time=0.0, duration=10.0, target_fps=25)
    transform = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    video_in = transform(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        return extractor.encode_video_with_sync(video_in)


def _run_generation(audiox_model: dict, video_path: str, task: str,
                    steps: int, cfg_scale: float,
                    sigma_min: float, sigma_max: float,
                    sampler_type: str, seed: int, custom_prompt: str):
    """
    Core generation logic shared by VideoToAudio and ImagesToAudio.
    Returns (audio_dict, duration_seconds).

    The model is fixed at 10 s; shorter videos are padded with their last
    frame, and the output waveform is trimmed to the actual video duration.
    """
    model        = audiox_model["model"]
    model_config = audiox_model["config"]
    model_name   = audiox_model["name"]
    device       = next(model.parameters()).device

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]  # 10 s worth of samples
    target_fps  = model_config["video_fps"]

    actual_duration = _get_video_duration(video_path)
    print(f"[AudioX] Video: {os.path.basename(video_path)} | "
          f"duration: {actual_duration}s")

    # Resolve prompt
    preset = _TASK_PRESETS[task]
    if preset == "__custom__":
        if not custom_prompt.strip():
            raise RuntimeError(
                "[AudioX] A custom_prompt is required for TV2A / TV2M tasks."
            )
        text_prompt = custom_prompt.strip()
    else:
        text_prompt = preset
    print(f"[AudioX] Prompt: {text_prompt}")

    # Seed
    if seed == -1:
        seed = int(torch.randint(0, 2**31, (1,)).item())
    torch.manual_seed(seed)
    print(f"[AudioX] Seed: {seed}")

    MODEL_SECONDS = 10.0
    video_tensor  = read_video(video_path, seek_time=0.0,
                               duration=MODEL_SECONDS, target_fps=target_fps)
    audio_tensor  = torch.zeros((2, int(sample_rate * MODEL_SECONDS)))

    # Synchformer (AudioX-MAF and AudioX-MAF-MMDiT only)
    video_sync_frames = None
    if "MAF" in model_name:
        print("[AudioX] Running Synchformer encoder...")
        sync_ckpt = os.path.join(_model_dir(model_name),
                                 "synchformer_state_dict.pth")
        video_sync_frames = _encode_synchformer(video_path, sync_ckpt,
                                                str(device))

    conditioning = [{
        "video_prompt": {
            "video_tensors":     video_tensor.unsqueeze(0),
            "video_sync_frames": video_sync_frames,
        },
        "text_prompt":   text_prompt,
        "audio_prompt":  audio_tensor.unsqueeze(0),
        "seconds_start": 0.0,
        "seconds_total": MODEL_SECONDS,
    }]

    print(f"[AudioX] Generating — steps={steps}  cfg={cfg_scale}  "
          f"sampler={sampler_type}")
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=str(device),
    )

    # Reshape (b d n) → (d, b*n), normalise, trim
    output = rearrange(output, "b d n -> d (b n)").to(torch.float32)
    peak = torch.max(torch.abs(output))
    if peak > 0:
        output = output / peak
    output = output.clamp(-1.0, 1.0)
    output = output[:, : int(sample_rate * actual_duration)]
    print(f"[AudioX] Done — output trimmed to {actual_duration}s")

    return (
        {"waveform": output.unsqueeze(0).cpu(), "sample_rate": sample_rate},
        float(actual_duration),
    )


# ---------------------------------------------------------------------------
# Shared task presets and sampler list
# ---------------------------------------------------------------------------
_TASK_PRESETS = {
    "V2A  — Video to Audio":        "Generate general audio for the video",
    "V2M  — Video to Music":        "Generate music for the video",
    "TV2A — Text + Video to Audio": "__custom__",
    "TV2M — Text + Video to Music": "__custom__",
}

_SAMPLER_CHOICES = ["dpmpp-3m-sde", "dpmpp-2m-sde", "k-heun", "k-dpm-fast"]


# ===========================================================================
# Node 1 — AudioX Model Loader
# ===========================================================================
class AudioXModelLoader:
    """Load a local AudioX model into GPU memory."""

    @classmethod
    def INPUT_TYPES(cls):
        available = _scan_available_models()
        return {
            "required": {
                "model_name": (available, {"default": available[0]}),
            }
        }

    RETURN_TYPES  = ("AUDIOX_MODEL",)
    RETURN_NAMES  = ("audiox_model",)
    FUNCTION      = "load_model"
    CATEGORY      = "AudioX"

    def load_model(self, model_name):
        model, config = _load_model(model_name)
        return ({"model": model, "config": config, "name": model_name},)


# ===========================================================================
# Node 2 — AudioX Video to Audio
# ===========================================================================
class AudioXVideoToAudio:
    """
    Generate audio from a video file.
    Connects to ComfyUI's built-in Load Video node (VIDEO type).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audiox_model": ("AUDIOX_MODEL",),
                "video":        ("VIDEO",),
                "task":         (list(_TASK_PRESETS.keys()),
                                 {"default": "V2M  — Video to Music"}),
                "steps":        ("INT",   {"default": 250, "min": 50,   "max": 500,  "step": 10}),
                "cfg_scale":    ("FLOAT", {"default": 7.0, "min": 1.0,  "max": 15.0, "step": 0.5}),
                "sigma_min":    ("FLOAT", {"default": 0.3, "min": 0.01, "max": 1.0,  "step": 0.01}),
                "sigma_max":    ("FLOAT", {"default": 500, "min": 100,  "max": 1000, "step": 50}),
                "sampler_type": (_SAMPLER_CHOICES, {"default": "dpmpp-3m-sde"}),
                "seed":         ("INT",   {"default": -1,  "min": -1,   "max": 2**31 - 1}),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default":     "",
                    "multiline":   True,
                    "placeholder": "Required for TV2A / TV2M tasks. "
                                   "Example: relaxing piano music",
                }),
            },
        }

    RETURN_TYPES  = ("AUDIO", "FLOAT")
    RETURN_NAMES  = ("audio", "duration_seconds")
    FUNCTION      = "generate"
    CATEGORY      = "AudioX"

    def generate(self, audiox_model, video, task, steps, cfg_scale,
                 sigma_min, sigma_max, sampler_type, seed, custom_prompt=""):

        video_path = _resolve_video_path(video)
        if not os.path.exists(video_path):
            raise RuntimeError(f"[AudioX] Video file not found: {video_path}")

        return _run_generation(
            audiox_model, video_path, task,
            steps, cfg_scale, sigma_min, sigma_max,
            sampler_type, seed, custom_prompt,
        )


# ===========================================================================
# Node 3 — AudioX Images to Audio  (VHS-compatible)
# ===========================================================================
class AudioXImagesToAudio:
    """
    Generate audio from an IMAGE tensor sequence.
    Compatible with VideoHelperSuite (VHS) and any node that outputs IMAGE frames.
    Requires ffmpeg on the system PATH to assemble frames into a temporary video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audiox_model": ("AUDIOX_MODEL",),
                "images":       ("IMAGE",),
                "fps":          ("FLOAT", {"default": 24.0, "min": 1.0,
                                           "max": 60.0, "step": 1.0}),
                "task":         (list(_TASK_PRESETS.keys()),
                                 {"default": "V2M  — Video to Music"}),
                "steps":        ("INT",   {"default": 250, "min": 50,   "max": 500,  "step": 10}),
                "cfg_scale":    ("FLOAT", {"default": 7.0, "min": 1.0,  "max": 15.0, "step": 0.5}),
                "sigma_min":    ("FLOAT", {"default": 0.3, "min": 0.01, "max": 1.0,  "step": 0.01}),
                "sigma_max":    ("FLOAT", {"default": 500, "min": 100,  "max": 1000, "step": 50}),
                "sampler_type": (_SAMPLER_CHOICES, {"default": "dpmpp-3m-sde"}),
                "seed":         ("INT",   {"default": -1,  "min": -1,   "max": 2**31 - 1}),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default":     "",
                    "multiline":   True,
                    "placeholder": "Required for TV2A / TV2M tasks. "
                                   "Example: relaxing piano music",
                }),
            },
        }

    RETURN_TYPES  = ("AUDIO", "FLOAT")
    RETURN_NAMES  = ("audio", "duration_seconds")
    FUNCTION      = "generate"
    CATEGORY      = "AudioX"

    def generate(self, audiox_model, images, fps, task, steps, cfg_scale,
                 sigma_min, sigma_max, sampler_type, seed, custom_prompt=""):

        video_path = self._frames_to_video(images, fps)
        return _run_generation(
            audiox_model, video_path, task,
            steps, cfg_scale, sigma_min, sigma_max,
            sampler_type, seed, custom_prompt,
        )

    @staticmethod
    def _frames_to_video(images: torch.Tensor, fps: float) -> str:
        """
        Write IMAGE tensor (B, H, W, C), float32 [0, 1] to a temporary
        H.264 MP4 file using ffmpeg.
        """
        from PIL import Image

        frames  = (images.cpu().numpy() * 255).astype(np.uint8)
        tmp_dir = tempfile.mkdtemp(prefix="audiox_")

        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(
                os.path.join(tmp_dir, f"frame_{i:06d}.png")
            )

        out_path = os.path.join(tmp_dir, "input.mp4")
        result   = subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                out_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"[AudioX] ffmpeg failed:\n{result.stderr}\n"
                "Ensure ffmpeg is installed and available on the system PATH."
            )
        print(f"[AudioX] Temp video: {out_path} "
              f"({len(frames)} frames @ {fps} fps)")
        return out_path


# ===========================================================================
# Node registration
# ===========================================================================
NODE_CLASS_MAPPINGS = {
    "AudioXModelLoader":   AudioXModelLoader,
    "AudioXVideoToAudio":  AudioXVideoToAudio,
    "AudioXImagesToAudio": AudioXImagesToAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioXModelLoader":   "AudioX Model Loader",
    "AudioXVideoToAudio":  "AudioX Video to Audio",
    "AudioXImagesToAudio": "AudioX Images to Audio (VHS)",
}
