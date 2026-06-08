"""Prompt formatting compatible with the LUTAttn LongBench path."""

from __future__ import annotations

from typing import Any

from .config import NO_CHAT_DATASETS


def infer_model_family(model_name: str | None, model_path: str | None = None) -> str:
    source = " ".join(part for part in [model_name, model_path] if part).lower()
    if "llama2-7b-80k" in source or "llama-2-7b-80k" in source:
        return "llama2-7b-80k"
    if "llama-3.2" in source or "llama3.2" in source:
        return "llama3.2"
    if "llama-3" in source or "llama3" in source:
        return "llama3"
    if "qwen" in source:
        return "qwen"
    if "glm-4" in source or "glm4" in source:
        return "glm4"
    if "chatglm3" in source:
        return "chatglm3"
    if "chatglm" in source:
        return "chatglm"
    if "longchat" in source:
        return "longchat"
    if "vicuna" in source:
        return "vicuna"
    if "llama2" in source or "llama-2" in source:
        return "llama2"
    if "mistral" in source and "instruct" in source:
        return "mistral-instruct"
    if "xgen" in source:
        return "xgen"
    if "internlm" in source:
        return "internlm"
    if "phi" in source:
        return "phi"
    return "default"


def build_chat(tokenizer: Any, prompt: str, model_family: str) -> str:
    """Apply the same chat wrapping policy as LUTAttn, with Qwen support."""
    family = model_family.lower()
    if family == "chatglm3":
        return tokenizer.build_chat_input(prompt)
    if family in {"glm4", "glm-4"}:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if family == "chatglm":
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer.build_prompt(prompt)
    if family in {"longchat", "vicuna"}:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    if family == "llama2-7b-80k":
        return f"<|im_start|> {prompt}"
    if family in {"llama3.2", "qwen", "phi"}:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if family == "llama2" or family == "mistral-instruct":
        return f"[INST]{prompt}[/INST]"
    if family == "xgen":
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        return header + f" ### Human: {prompt}\n###"
    if family == "internlm":
        return f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def middle_truncate(tokenizer: Any, prompt: str, max_tokens: int) -> str:
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) <= max_tokens:
        return prompt
    half = max_tokens // 2
    return tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
        tokenized[-(max_tokens - half) :], skip_special_tokens=True
    )


def format_longbench_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    dataset: str,
    model_family: str,
    max_model_len: int,
    prompt_token_reserve: int = 0,
) -> str:
    """Format and middle-truncate a LongBench prompt following LUTAttn."""
    effective_max = max(1, max_model_len - max(0, prompt_token_reserve))
    prompt = middle_truncate(tokenizer, prompt, effective_max)
    if dataset not in NO_CHAT_DATASETS:
        prompt = build_chat(tokenizer, prompt, model_family)
    prompt = middle_truncate(tokenizer, prompt, effective_max)
    return prompt


def post_process(response: str, model_family: str) -> str:
    family = model_family.lower()
    if family == "xgen":
        return response.strip().replace("Assistant:", "")
    if family == "internlm":
        return response.split("<eoa>")[0]
    return response
