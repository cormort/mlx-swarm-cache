---
license: apache-2.0
language:
- en
- zh
base_model: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
library_name: mlx
tags:
- mlx
- reasoning
- distilled
- qwen
- qwen3.5
- apple-silicon
- chain-of-thought
- claude-4.6-opus
- mlx-community
---

# Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-4bit-MLX

Quantized by [BeastCode](https://huggingface.co/BeastCode)

A high-performance **4-bit MLX quantization** of [Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled). Specifically optimized for Apple Silicon (M-series chips) to provide deep, agentic-level reasoning locally.

The original BF16 weights are **55.6 GB**. This conversion reduces the footprint to **14 GB**, making it runnable on any Mac with 24 GB+ of unified memory with room to spare for large context windows.

---

## 🧠 Why This Model?

Most local LLMs are "reactive" — they start generating a response before they've fully mapped out the logic. This model is **deliberative**.

Distilled from state-of-the-art **Claude 4.6 Opus** reasoning trajectories, it uses an advanced Chain-of-Thought (CoT) scaffold. Before providing its final answer, it enters an internal `<think>` state where it:

- **Deconstructs** complex, multi-layered prompts into manageable sub-tasks
- **Simulates** different solution paths and self-corrects logic errors before you see them
- **Reduces redundancy** by adopting Claude's structured thinking pattern rather than the looping often seen in base reasoning models

This makes it the premier choice for **technical planning, complex logic puzzles, and high-stakes decision support** on Apple hardware.

---

## 📊 Performance Benchmarks

> Tested on Apple M4 Pro, 64 GB · `mlx-lm 0.30.7` · macOS 15

| Metric | Result |
|--------|--------|
| Model load time | 2.4 seconds |
| Prompt ingestion | 86.5 tokens/sec |
| Generation speed | 15.7 tokens/sec |
| Peak RAM usage | 15.6 GB |
| Bit-rate | 4.501 bits/weight |
| Final size | 14 GB (3 shards) |

---

## 💻 System Requirements

| | |
|--|--|
| **Hardware** | Apple Silicon Mac (M1, M2, M3, M4 or later) |
| **Minimum RAM** | 24 GB Unified Memory |
| **Recommended RAM** | 32 GB+ (headroom for long-context reasoning) |
| **OS** | macOS 13.5 or later |
| **Python** | 3.10+ (Homebrew Python 3.12 recommended) |

---

## 🚀 Quick Start

### 1. Install the MLX library

```bash
pip install mlx-lm
```

### 2. Run in your terminal

```bash
python -m mlx_lm.chat \
  --model BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
```

### 3. Python integration — recommended approach

Use `apply_chat_template` with `enable_thinking=True`. This is the idiomatic way to trigger the reasoning mode — no manual prompt construction needed.

```python
from mlx_lm import load, generate

model, tokenizer = load("BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit")

messages = [
    {
        "role": "user",
        "content": (
            "A farmer needs to cross a river with a wolf, a goat, and a cabbage. "
            "The boat can only hold the farmer and one other item. "
            "If left alone, the wolf eats the goat, and the goat eats the cabbage. "
            "How can he get everything across safely?"
        ),
    }
]

# enable_thinking=True inserts the <think> prefix automatically
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

response = generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=True)
print(response)
```

> **Tip:** Set `enable_thinking=False` (or omit it) for fast, direct answers without the reasoning block — useful for simple lookups or chat where latency matters.

<details>
<summary>Manual prefill style (advanced)</summary>

If you want to see exactly what happens under the hood, you can construct the prompt manually and pre-fill the `<think>` token yourself. The two approaches produce identical results.

```python
prompt = (
    "<|im_start|>system\n"
    "You are a highly analytical assistant.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "A farmer needs to cross a river with a wolf, a goat, and a cabbage. "
    "The boat can only hold the farmer and one other item. "
    "If left alone, the wolf eats the goat, and the goat eats the cabbage. "
    "How can he get everything across safely?\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
)

response = generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=True)
print(response)
```

</details>

---

## ⚙️ Quantization Details

| Property | Value |
|----------|-------|
| Method | 4-bit group-wise quantization |
| Precision | Mixed (embeddings/heads kept at higher precision for stability) |
| Tooling | `mlx-lm.convert` |
| Base model | Qwen 3.5 27B (Dense Architecture) |

### Reproduce this quantization

```bash
python -m mlx_lm.convert \
  --hf-path Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled \
  --mlx-path ./Qwen3.5-27B-Claude-4.6-MLX \
  --quantize \
  --q-bits 4
```

---

## 🧩 Stripping the Reasoning Block

The `<think>` block is invaluable for verifying logic, but you may want to strip it for a cleaner UI:

```python
import re

def strip_thinking(text: str) -> str:
    """Remove the internal <think> process, returning only the final answer."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
```

---

## 🏆 Model Comparison

| Model | Size | Reasoning style | Hardware target |
|-------|------|-----------------|-----------------|
| **This model (27B)** | **14 GB** | Claude 4.6 distilled | 24 GB+ Macs |
| Qwen3.5-9B | ~5 GB | Fast / intuitive | Base 8 GB / 16 GB Macs |
| Qwen3.5-72B | ~42 GB | Deep / exhaustive | 64 GB+ Ultra/Max |

---

## 🙏 Acknowledgements

- **Core weights:** Alibaba Qwen Team — [Qwen 3.5 27B](https://huggingface.co/Qwen)
- **Reasoning SFT:** [Jackrong](https://huggingface.co/Jackrong) for the Claude 4.6 Opus distillation work
- **Inference engine:** Apple MLX Team for making high-speed local inference possible on Apple Silicon