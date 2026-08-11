"""Lazy Llama chat inference for chapterwise Kaggle GPU server."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from typing import Any

_model = None
_tokenizer = None
_load_error: str | None = None
_load_lock = threading.Lock()
_gen_lock = threading.Lock()

_PIP_PACKAGES: list[tuple[str, list[str]]] = [
    ("torch", ["torch"]),
    ("transformers", ["transformers", "accelerate", "sentencepiece"]),
]


def model_id() -> str:
    return os.environ.get(
        "CHAPTERWISE_LLM_MODEL",
        "meta-llama/Llama-3.2-1B-Instruct",
    ).strip()


def max_new_tokens() -> int:
    return int(os.environ.get("CHAPTERWISE_LLM_MAX_NEW_TOKENS", "1024"))


def max_input_chars() -> int:
    return int(os.environ.get("CHAPTERWISE_LLM_MAX_INPUT_CHARS", "12000"))


_TRUNC_SUFFIX = "\n\n[truncated for model context limit]"
_ASSISTANT_CAP_CHARS = int(os.environ.get("CHAPTERWISE_LLM_ASSISTANT_CAP_CHARS", "2000"))


def _quiet_pip(*packages: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        env={**os.environ, "PIP_PROGRESS_BAR": "off", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )


def _ensure_packages() -> None:
    for import_name, pip_args in _PIP_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            print(f"chapterwise LLM: installing {import_name} …")
            _quiet_pip(*pip_args)

    if os.environ.get("CHAPTERWISE_LLM_4BIT", "1") == "1":
        try:
            __import__("bitsandbytes")
        except ImportError:
            print("chapterwise LLM: installing bitsandbytes …")
            _quiet_pip("bitsandbytes")


def _hf_token() -> str | None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()
    return token or None


def _device_name() -> str | None:
    if _model is None:
        return None
    try:
        return str(next(_model.parameters()).device)
    except Exception:
        return None


def status() -> dict[str, Any]:
    return {
        "model": model_id(),
        "loaded": _model is not None,
        "load_error": _load_error,
        "device": _device_name(),
    }


def _message_total_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages)


def _truncate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim conversation to max_input_chars across all roles (not only the last user)."""
    limit = max_input_chars()
    working: list[dict[str, str]] = [
        {
            "role": str(message.get("role") or "user"),
            "content": str(message.get("content") or ""),
        }
        for message in messages
    ]
    if _message_total_chars(working) <= limit:
        return working

    for message in working:
        if message["role"] == "assistant" and len(message["content"]) > _ASSISTANT_CAP_CHARS:
            message["content"] = message["content"][:_ASSISTANT_CAP_CHARS] + _TRUNC_SUFFIX
            print(
                f"chapterwise LLM: truncated assistant message to "
                f"{_ASSISTANT_CAP_CHARS} chars"
            )

    trim_order: list[tuple[int, int]] = []
    for index, message in enumerate(working):
        if message["role"] == "assistant":
            trim_order.append((0, index))
    for index, message in enumerate(working):
        if message["role"] == "user" and index != len(working) - 1:
            trim_order.append((1, index))
    if working:
        trim_order.append((2, len(working) - 1))
    if working and working[0]["role"] == "system":
        trim_order.append((3, 0))
    trim_order.sort(key=lambda item: item[0])

    while _message_total_chars(working) > limit:
        trimmed_any = False
        overflow = _message_total_chars(working) - limit
        for _, index in trim_order:
            message = working[index]
            content = message["content"]
            if index == len(working) - 1:
                min_keep = 200
            elif message["role"] == "system":
                min_keep = 1500
            else:
                min_keep = 500
            if len(content) <= min_keep:
                continue
            new_len = max(min_keep, len(content) - overflow)
            if new_len < len(content):
                message["content"] = content[:new_len] + _TRUNC_SUFFIX
                print(
                    f"chapterwise LLM: truncated {message['role']} message "
                    f"to {new_len} chars (total budget {limit})"
                )
                trimmed_any = True
                break
        if not trimmed_any:
            break

    return working


def _load_model() -> None:
    global _model, _tokenizer, _load_error

    with _load_lock:
        if _model is not None:
            return
        if _load_error:
            raise RuntimeError(_load_error)

        try:
            _ensure_packages()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            mid = model_id()
            token = _hf_token()
            print(
                f"chapterwise LLM: HF auth token in server process: "
                f"present={bool(token)}, len={len(token or '')}"
            )
            if not token:
                print(
                    "chapterwise LLM: no HF_TOKEN in environment — "
                    "set via notebook UserSecretsClient before !python server"
                )
            else:
                try:
                    from huggingface_hub import login

                    login(token=token, add_to_git_credential=False)
                except Exception as exc:
                    print(f"chapterwise LLM: huggingface_hub.login failed: {exc}")
            print(f"chapterwise LLM: loading {mid} …")
            tokenizer = AutoTokenizer.from_pretrained(mid, token=token)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            use_4bit = os.environ.get("CHAPTERWISE_LLM_4BIT", "1") == "1"
            if use_4bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig

                    quant = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        mid,
                        quantization_config=quant,
                        device_map="auto",
                        token=token,
                    )
                except Exception as exc:
                    print(f"chapterwise LLM: 4-bit load failed ({exc}); using float16 …")
                    model = AutoModelForCausalLM.from_pretrained(
                        mid,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        token=token,
                    )
            elif torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    mid,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    token=token,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    mid,
                    torch_dtype=torch.float32,
                    token=token,
                )

            model.eval()
            _model = model
            _tokenizer = tokenizer
            print(f"chapterwise LLM ready: {mid} on {_device_name()}")
        except Exception as exc:
            _load_error = str(exc)
            print(f"chapterwise LLM load failed: {exc}")
            raise RuntimeError(_load_error) from exc


def generate_chat(messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
    _load_model()
    if _model is None or _tokenizer is None:
        raise RuntimeError(_load_error or "LLM is not loaded.")

    prepared = _truncate_messages(messages)
    with _gen_lock:
        import torch

        prompt = _tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _tokenizer(prompt, return_tensors="pt")
        device = next(_model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens(),
                temperature=max(temperature, 0.01) if do_sample else None,
                do_sample=do_sample,
                repetition_penalty=1.2,
                no_repeat_ngram_size=4,
                pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not text:
            raise RuntimeError("LLM returned an empty response.")
        return text


def preload() -> None:
    if os.environ.get("CHAPTERWISE_LLM_PRELOAD", "0") == "1":
        _load_model()
