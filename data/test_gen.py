"""Smoke-test an adapter checkpoint on an unseen topic."""
import sys

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

SYSTEM = (
    "You are Rohit Gupta, a backend/React developer who builds AI agents and "
    "posts about them on LinkedIn. Write in his voice: a bold hook line, a "
    "personal build story, em-dash bullets with concrete receipts, and a "
    "short mission close."
)
adapter = sys.argv[1] if len(sys.argv) > 1 else "adapters"
topic = sys.argv[2] if len(sys.argv) > 2 else "why I run AI models on my Mac instead of the cloud"

model, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit", adapter_path=adapter)
prompt = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": f"Write a LinkedIn post about: {topic}"}],
    add_generation_prompt=True,
)
print(generate(model, tok, prompt, max_tokens=500,
               sampler=make_sampler(temp=0.7, top_p=0.9)))
