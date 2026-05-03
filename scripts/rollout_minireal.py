#!/usr/bin/env python3
"""
Autoregressive rollout for MiniReal test episodes using finetuned IRASim (frame_ada).

Expects per-episode RDT outputs as numpy:
  <rdt_actions>/<episode>/candidate_<k>.npy   shape [51, 26]
or a single file:
  <rdt_actions>/<episode>.npy                  shape [K, 51, 26]

Uses test/{episode}/video.mp4 (16 frames) + joint.txt row 15 for conditioning.

Run from IRASim root:
  python scripts/rollout_minireal.py \\
    --config configs/evaluation/minireal/frame_ada.yaml \\
    --checkpoint robotdata/opensource_robotdata/minireal/checkpoints/frame_ada/0050000.pt \\
    --test-data /path/to/release/test \\
    --rdt-actions /path/to/rdt_preds \\
    --out irasim_rollouts
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from einops import rearrange

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from evaluate.generate_short_video import generate_single_video
from models import get_models
from scripts.minireal_action_util import (
    parse_csv_numeric,
    states_to_delta_actions,
    build_arm_trajectory_from_test_and_rdt,
)
from util import get_args


def _video_frames_rgb(path: Path, max_frames: int = 16):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < max_frames:
        raise RuntimeError(f"{path}: expected >= {max_frames} frames, got {len(frames)}")
    return frames[:max_frames]


def _frames_to_latent_start(
    frames_rgb, args, vae, device: torch.device
):
    """Encode frame 15 (index 15) to latent [1,1,C,H,W] matching training."""
    from torchvision import transforms as T
    from dataset.video_transforms import Resize_Preprocess, ToTensorVideo

    preprocess = T.Compose(
        [
            ToTensorVideo(),
            Resize_Preprocess(tuple(args.video_size)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    vid = torch.stack(
        [
            torch.from_numpy(f).permute(2, 0, 1).to(torch.uint8)
            for f in frames_rgb
        ],
        dim=0,
    )
    vid = preprocess(vid).unsqueeze(0).to(device)
    # encode all frames then take last-of-prefix
    b, f, c, h, w = vid.shape
    x = rearrange(vid, "b f c h w -> (b f) c h w")
    with torch.no_grad():
        enc = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
    enc = rearrange(enc, "(b f) c h w -> b f c h w", b=b, f=f)
    return enc[:, 15:16]


def _chunk_actions(act50: np.ndarray, chunk: int = 15):
    """Pad [50,7] into chunks of length 15 (num_frames-1)."""
    out = []
    start = 0
    while start < act50.shape[0]:
        piece = act50[start : start + chunk]
        if piece.shape[0] < chunk:
            pad_n = chunk - piece.shape[0]
            piece = np.concatenate([piece, np.repeat(piece[-1:], pad_n, axis=0)], axis=0)
        out.append(piece.astype(np.float32))
        start += chunk
    return out


def _generate_fifty_frames(
    args,
    device,
    vae,
    model,
    start_latent_b1fchw: torch.Tensor,
    actions_scaled_50x7: torch.Tensor,
):
    """Autoregressive segments of length num_frames (default 16 -> 15 actions each)."""
    num_frames = args.num_frames
    step = num_frames - 1
    chunks = _chunk_actions(actions_scaled_50x7.cpu().numpy(), chunk=step)
    generated_rgb = []
    start_image = start_latent_b1fchw.clone()
    for ci, ca in enumerate(chunks):
        mask_x = start_image.to(device).float()
        actions = torch.from_numpy(ca).unsqueeze(0).to(device).float()
        seg_video, seg_latents = generate_single_video(
            args,
            mask_x,
            actions,
            device,
            vae,
            model,
        )
        # Same as evaluate/generate_short_video.generate_sample_videos
        seg_video = seg_video.detach().cpu()
        seg_video = seg_video.permute(0, 1, 3, 4, 2)
        seg_video = ((seg_video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8)
        seg_np = seg_video[0].numpy()
        remain_need = 50 - sum(x.shape[0] for x in generated_rgb)
        take = min(step, remain_need, seg_np.shape[0] - 1)
        if take <= 0:
            break
        generated_rgb.append(seg_np[1 : 1 + take])
        if seg_latents is not None:
            start_image = seg_latents[:, -1:].detach().to(device)
        else:
            raise RuntimeError("Pipeline returned no latents; cannot chain")
        if sum(x.shape[0] for x in generated_rgb) >= 50:
            break
    out = np.concatenate(generated_rgb, axis=0)[:50]
    return out


def _load_rdt_candidates(ep: str, rdt_root: Path) -> list[np.ndarray]:
    """Return list of [51,26] arrays."""
    ep_dir = rdt_root / ep
    single = rdt_root / f"{ep}.npy"
    if single.is_file():
        arr = np.load(single)
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 2 and arr.shape[0] == 51:
            return [arr]
        raise ValueError(f"Bad shape in {single}: {arr.shape}")
    if not ep_dir.is_dir():
        raise FileNotFoundError(f"No RDT data for episode {ep} under {rdt_root}")
    cands = sorted(ep_dir.glob("candidate_*.npy"))
    if not cands:
        raise FileNotFoundError(f"No candidate_*.npy in {ep_dir}")
    out = []
    for p in cands:
        a = np.load(p)
        if a.shape != (51, 26):
            raise ValueError(f"{p} expected (51,26), got {a.shape}")
        out.append(a.astype(np.float32))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluation/minireal/frame_ada.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override evaluate_checkpoint from yaml (full path or under robotdata/...)",
    )
    parser.add_argument("--test-data", type=str, required=True, help="Folder containing <episode>/ video joint action")
    parser.add_argument("--rdt-actions", type=str, required=True, help="Root of RDT numpy predictions")
    parser.add_argument("--out", type=str, default="irasim_rollouts")
    parser.add_argument("--episodes", type=str, default=None, help="Comma-separated episode ids; default all under test-data")
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="torch device index cuda:N (default 0). For multi-GPU, prefer one process per GPU via CUDA_VISIBLE_DEVICES=K so each process uses cuda:0 on that physical card.",
    )
    args_ns = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for IRASim rollout.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(f"cuda:{args_ns.cuda_device}")
    cfg = argparse.Namespace(config=args_ns.config)
    args = get_args(cfg)
    if args_ns.checkpoint:
        ck = Path(args_ns.checkpoint)
        if ck.is_file():
            args.evaluate_checkpoint = str(ck.resolve())
        else:
            args.evaluate_checkpoint = str((_ROOT / args_ns.checkpoint).resolve())
    test_root = Path(args_ns.test_data).resolve()
    rdt_root = Path(args_ns.rdt_actions).resolve()
    out_root = Path(args_ns.out)
    if not out_root.is_absolute():
        out_root = (_ROOT / out_root).resolve()

    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device)
    args.latent_size = [t // 8 for t in args.video_size]
    model = get_models(args).to(device)

    ck_path = args.evaluate_checkpoint
    if not ck_path or not Path(ck_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    checkpoint = torch.load(ck_path, map_location="cpu", weights_only=False)
    raw_sd = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in raw_sd.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v
        else:
            mv = model_dict[k].shape if k in model_dict else "missing"
            logger.info(
                "Skipping %s: ckpt %s vs model %s",
                k,
                tuple(v.shape),
                tuple(mv) if mv != "missing" else mv,
            )
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    c_act = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0], dtype=np.float32)

    ep_ids = (
        [e.strip() for e in args_ns.episodes.split(",") if e.strip()]
        if args_ns.episodes
        else sorted([p.name for p in test_root.iterdir() if p.is_dir()])
    )

    for ep in ep_ids:
        ep_dir = test_root / ep
        vid_p = ep_dir / "video.mp4"
        joint_p = ep_dir / "joint.txt"
        if not vid_p.is_file() or not joint_p.is_file():
            print(f"[skip] missing files in {ep_dir}")
            continue
        header_j, joint_mat = parse_csv_numeric(joint_p)
        if joint_mat.shape[0] < 16:
            print(f"[skip] {ep}: need >=16 joint rows")
            continue
        joint_row_15 = joint_mat[15]
        frames_rgb = _video_frames_rgb(vid_p, 16)
        start_latent = _frames_to_latent_start(frames_rgb, args, vae, device)

        try:
            cand_list = _load_rdt_candidates(ep, rdt_root)
        except FileNotFoundError as e:
            print(f"[skip] {ep}: {e}")
            continue

        for k, rdt in enumerate(cand_list):
            arm, grip = build_arm_trajectory_from_test_and_rdt(joint_row_15, rdt)
            d_all = states_to_delta_actions(arm, grip, accumulate_action=False)
            if d_all.shape[0] < 50:
                raise RuntimeError(f"{ep}: internal action length {d_all.shape}")
            d50 = d_all[:50]
            actions_scaled = torch.from_numpy(d50 * c_act).float()

            rgb50 = _generate_fifty_frames(args, device, vae, model, start_latent, actions_scaled)
            dest = out_root / ep / f"candidate_{k}"
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest / "frames.npy", rgb50.astype(np.uint8))
            np.save(dest / "action_51x26.npy", rdt.astype(np.float32))
            meta = {
                "episode": ep,
                "candidate_id": k,
                "checkpoint": str(ck_path),
                "video_shape": list(rgb50.shape),
            }
            (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
