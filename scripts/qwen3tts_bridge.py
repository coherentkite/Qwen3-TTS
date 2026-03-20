#!/usr/bin/env python3
# coding: utf-8

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse

import torch

try:
    import soundfile as sf
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency `soundfile`. Install project dependencies (for example `pip install -e .`)."
    ) from exc

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen3tts_bridge.py",
        description="Python bridge for Node CLI wrapper around Qwen3-TTS voice clone.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    clone = sub.add_parser("clone", help="Create reusable ICL voice clone prompt and save to .pt.")
    add_common_model_args(clone)
    clone.add_argument("--ref-audio", required=True, help="Reference audio path/URL/base64.")
    clone_text = clone.add_mutually_exclusive_group(required=True)
    clone_text.add_argument("--ref-text", help="Reference transcript text.")
    clone_text.add_argument("--ref-text-file", help="Path to file with reference transcript.")
    clone.add_argument("--clone-file", required=True, help="Output .pt clone file path.")

    speak = sub.add_parser("speak", help="Generate speech from text using saved clone .pt.")
    add_common_model_args(speak)
    speak.add_argument("--clone-file", required=True, help="Input clone .pt file path.")
    speak_text = speak.add_mutually_exclusive_group(required=True)
    speak_text.add_argument("--text", help="Text to synthesize.")
    speak_text.add_argument("--text-file", help="Path to file with text to synthesize.")
    speak.add_argument("--out", required=True, help="Output wav file path.")
    speak.add_argument("--language", default="Auto", help="Target language (default: Auto).")
    speak.add_argument("--max-new-tokens", type=int, default=None)
    speak.add_argument("--top-k", type=int, default=None)
    speak.add_argument("--top-p", type=float, default=None)
    speak.add_argument("--temperature", type=float, default=None)
    speak.add_argument("--repetition-penalty", type=float, default=None)

    return parser.parse_args()


def add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id/path (default: {DEFAULT_MODEL}).")
    p.add_argument("--device", default="cuda:0", help='device_map value, e.g. "cuda:0" or "cpu".')
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for model loading.",
    )
    flash_group = p.add_mutually_exclusive_group()
    flash_group.add_argument(
        "--flash-attn",
        dest="flash_attn",
        action="store_true",
        default=True,
        help="Enable FlashAttention-2 (default: enabled).",
    )
    flash_group.add_argument(
        "--no-flash-attn",
        dest="flash_attn",
        action="store_false",
        help="Disable FlashAttention-2.",
    )


def dtype_from_str(name: str) -> torch.dtype:
    lowered = (name or "").strip().lower()
    if lowered in ("bfloat16", "bf16"):
        return torch.bfloat16
    if lowered in ("float16", "fp16"):
        return torch.float16
    if lowered in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model(model_id: str, device: str, dtype_name: str, flash_attn: bool) -> Qwen3TTSModel:
    kwargs: Dict[str, Any] = {
        "device_map": device,
        "dtype": dtype_from_str(dtype_name),
    }
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def read_text_arg(raw_text: Optional[str], text_file: Optional[str], field_name: str) -> str:
    if raw_text is not None:
        text = raw_text
    else:
        if not text_file:
            raise ValueError(f"{field_name} is required.")
        text = Path(text_file).read_text(encoding="utf-8")

    if not text or not text.strip():
        raise ValueError(f"{field_name} must be non-empty.")
    return text.strip()


def is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def is_probably_base64_audio(value: str) -> bool:
    if value.startswith("data:audio"):
        return True
    # Heuristic reused from model wrapper style: long string without path separators.
    return len(value) > 256 and ("/" not in value and "\\" not in value)


def maybe_convert_local_audio_to_wav(ref_audio: str) -> tuple[str, Optional[Path]]:
    """
    If `ref_audio` is a local non-wav file path (e.g. .m4a), convert it to a temporary wav with ffmpeg.
    Returns (audio_path_for_model, temp_file_path_or_none).
    """
    if is_url(ref_audio) or is_probably_base64_audio(ref_audio):
        return ref_audio, None

    source = Path(ref_audio)
    if not source.exists() or not source.is_file():
        # Let downstream loader handle path errors for consistency.
        return ref_audio, None

    if source.suffix.lower() == ".wav":
        return str(source), None

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Reference audio is not WAV and ffmpeg is not installed. "
            "Install ffmpeg or convert the file manually to .wav."
        )

    with tempfile.NamedTemporaryFile(prefix="qwen3tts_ref_", suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "pcm_s16le",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        stderr_tail = (proc.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError(
            "ffmpeg conversion failed while converting reference audio to WAV.\n"
            + "\n".join(stderr_tail)
        )

    print(f"Converted reference audio to WAV via ffmpeg: {tmp_path}")
    return str(tmp_path), tmp_path


def item_to_payload_dict(item: VoiceClonePromptItem) -> Dict[str, Any]:
    return {
        "ref_code": None if item.ref_code is None else item.ref_code.detach().cpu(),
        "ref_spk_embedding": item.ref_spk_embedding.detach().cpu(),
        "x_vector_only_mode": bool(item.x_vector_only_mode),
        "icl_mode": bool(item.icl_mode),
        "ref_text": item.ref_text,
    }


def validate_clone_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid clone file: expected a dict payload.")

    required_keys = {"schema_version", "model_id", "tts_model_type", "tokenizer_type", "created_at", "items"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise ValueError(f"Invalid clone file: missing keys: {missing}")

    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported clone schema_version={payload['schema_version']}. Expected {SCHEMA_VERSION}."
        )
    if payload["tts_model_type"] != "base":
        raise ValueError(
            f"Clone file was built with unsupported tts_model_type={payload['tts_model_type']!r}; expected 'base'."
        )
    if not isinstance(payload["items"], list) or len(payload["items"]) == 0:
        raise ValueError("Invalid clone file: `items` must be a non-empty list.")

    return payload


def payload_to_items(payload: Dict[str, Any]) -> List[VoiceClonePromptItem]:
    out: List[VoiceClonePromptItem] = []
    for idx, item in enumerate(payload["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid clone file: items[{idx}] must be a dict.")

        if bool(item.get("x_vector_only_mode", False)):
            raise ValueError(
                f"Invalid clone file: items[{idx}] has x_vector_only_mode=True. "
                "This CLI only supports ICL clone files."
            )
        if not bool(item.get("icl_mode", False)):
            raise ValueError(
                f"Invalid clone file: items[{idx}] has icl_mode=False. This CLI requires ICL mode."
            )

        ref_text = item.get("ref_text")
        if ref_text is None or not str(ref_text).strip():
            raise ValueError(
                f"Invalid clone file: items[{idx}] is missing ref_text required for ICL mode."
            )

        ref_spk_embedding = item.get("ref_spk_embedding")
        if not torch.is_tensor(ref_spk_embedding):
            raise ValueError(f"Invalid clone file: items[{idx}].ref_spk_embedding must be a torch.Tensor.")

        ref_code = item.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            raise ValueError(f"Invalid clone file: items[{idx}].ref_code must be a torch.Tensor or None.")

        out.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text=str(ref_text),
            )
        )
    return out


def safe_torch_load(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_clone(args: argparse.Namespace) -> None:
    ref_text = read_text_arg(args.ref_text, args.ref_text_file, field_name="ref_text")
    model = load_model(args.model, args.device, args.dtype, args.flash_attn)

    if model.model.tts_model_type != "base":
        raise ValueError(
            f"Model {args.model!r} has tts_model_type={model.model.tts_model_type!r}; expected 'base'."
        )

    normalized_ref_audio, temp_wav = maybe_convert_local_audio_to_wav(args.ref_audio)
    try:
        items = model.create_voice_clone_prompt(
            ref_audio=normalized_ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
    finally:
        if temp_wav is not None:
            temp_wav.unlink(missing_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model,
        "tts_model_type": model.model.tts_model_type,
        "tokenizer_type": model.model.tokenizer_type,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "items": [item_to_payload_dict(x) for x in items],
    }

    clone_path = Path(args.clone_file)
    clone_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(clone_path))
    print(f"Saved clone file: {clone_path} (items={len(items)})")


def run_speak(args: argparse.Namespace) -> None:
    text = read_text_arg(args.text, args.text_file, field_name="text")
    payload = validate_clone_payload(safe_torch_load(args.clone_file))

    if payload["model_id"] != args.model:
        raise ValueError(
            f"Model mismatch: clone file was created with {payload['model_id']!r}, "
            f"but current --model is {args.model!r}."
        )

    model = load_model(args.model, args.device, args.dtype, args.flash_attn)

    if model.model.tts_model_type != payload["tts_model_type"]:
        raise ValueError(
            f"Model type mismatch: clone expects {payload['tts_model_type']!r}, loaded model is "
            f"{model.model.tts_model_type!r}."
        )
    if model.model.tokenizer_type != payload["tokenizer_type"]:
        raise ValueError(
            f"Tokenizer mismatch: clone expects {payload['tokenizer_type']!r}, loaded model has "
            f"{model.model.tokenizer_type!r}."
        )

    prompt_items = payload_to_items(payload)

    gen_kwargs = {}
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=args.language,
        voice_clone_prompt=prompt_items,
        **gen_kwargs,
    )
    if not wavs:
        raise RuntimeError("Model returned empty audio output.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wavs[0], sr)
    print(f"Saved wav: {out_path} (sr={sr}, samples={len(wavs[0])})")


def main() -> None:
    args = parse_args()
    if args.command == "clone":
        run_clone(args)
    elif args.command == "speak":
        run_speak(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
