"""Explicit model identities and prompt contracts, without loading ML runtimes."""
import json
import math
from pathlib import Path

MODEL_REPOS = {
    'dinov2b': 'facebook/dinov2-base',
    'dinov2l': 'facebook/dinov2-large',
    'sam2': 'facebook/sam2.1-hiera-small',
    'dinov3b': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'dinov3l': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'sam3': 'facebook/sam3',
    'qwen': 'Qwen/Qwen3.5-4B',
}
PUBLIC_MODELS = frozenset(('dinov2b', 'dinov2l', 'sam2', 'qwen'))
ENCODERS = frozenset(('dinov2b', 'dinov2l', 'dinov3b', 'dinov3l'))
PUBLIC_REVISIONS = {
    'qwen': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'dinov2b': 'f9e44c814b77203eaa57a6bdbbd535f21ede1415',
    'dinov2l': '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c',
    'sam2': 'ee5bba1d82bb8749febdf90f45e84b687142ba03',
}

def encoder_name(args):
    name = args.get('model', 'dinov2b')
    if name not in ENCODERS:
        raise ValueError('Unknown visual encoder')
    return name

def weights_ready(root, name):
    folder = Path(root) / name
    if name not in MODEL_REPOS:
        return False
    if name == 'sam3':
        checkpoint = folder / 'sam3.pt'
        return checkpoint.is_file() and checkpoint.stat().st_size > 0
    if not all((folder / f).is_file() for f in ('config.json', 'preprocessor_config.json')):
        return False
    weight = folder / 'model.safetensors'
    if weight.is_file() and weight.stat().st_size > 0:
        return True
    index = folder / 'model.safetensors.index.json'
    try:
        shards = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
        return bool(shards) and all(Path(f).name == f and (folder / f).is_file()
                                    and (folder / f).stat().st_size > 0 for f in shards)
    except (OSError, ValueError, KeyError, TypeError):
        return False

def normalized_box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Select an image region first: normalized center_x, center_y, width, height')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError('Box coordinates must be finite numbers')
    cx, cy, width, height = map(float, value)
    if not (0 < width <= 1 and 0 < height <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
        raise ValueError('Box size must be positive and normalized to the image')
    if cx - width / 2 < -1e-7 or cy - height / 2 < -1e-7 or cx + width / 2 > 1 + 1e-7 or cy + height / 2 > 1 + 1e-7:
        raise ValueError('Box extends outside the image')
    return [cx, cy, width, height]

def box_pixels(value, width, height):
    cx, cy, bw, bh = normalized_box(value)
    return [max(0., (cx - bw / 2) * width), max(0., (cy - bh / 2) * height),
            min(float(width), (cx + bw / 2) * width), min(float(height), (cy + bh / 2) * height)]
