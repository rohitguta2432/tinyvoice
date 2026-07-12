"""tinyvoice — serve base vs fine-tuned model side by side.

One FastAPI app: /api/generate streams tokens (SSE) from either the stock
Qwen2.5-0.5B or the LoRA-tuned "my voice" version of it.

MLX stream state is bound to the thread that touches it first, so ALL MLX
work (model load + generation) lives on ONE persistent inference thread;
requests hand jobs over via a queue.
"""
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

VOICE_SYSTEM = (
    "You are Rohit Gupta, a backend/React developer who builds AI agents and "
    "posts about them on LinkedIn. Write in his voice: a bold hook line, a "
    "personal build story, em-dash bullets with concrete receipts, and a "
    "short mission close."
)
BASE_SYSTEM = "You are a helpful assistant."

app = FastAPI(title="tinyvoice")

jobs: queue.Queue = queue.Queue()
models = {}
ready = threading.Event()


def inference_loop():
    """Owns every MLX call. Loads models, then serves generation jobs forever."""
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print("loading base…")
    models["base"] = load(MODEL)
    print("loading tuned…")
    models["mine"] = load(MODEL, adapter_path=str(ROOT / "adapters"))
    print("ready")
    ready.set()

    while True:
        voice, prompt, out_q = jobs.get()
        model, tok = models[voice]
        try:
            sampler = make_sampler(temp=0.7, top_p=0.9)
            for r in stream_generate(model, tok, prompt, max_tokens=600, sampler=sampler):
                out_q.put(r.text)
        except Exception as e:  # surface errors to the client, keep the loop alive
            out_q.put(f"\n[error: {e}]")
        finally:
            out_q.put(None)


threading.Thread(target=inference_loop, daemon=True).start()


class Ask(BaseModel):
    topic: str
    voice: str = "mine"  # "base" | "mine"


def sse(events):
    for e in events:
        yield f"data: {json.dumps(e)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/generate")
def generate(ask: Ask):
    ready.wait(timeout=120)
    voice = "mine" if ask.voice == "mine" else "base"
    system = VOICE_SYSTEM if voice == "mine" else BASE_SYSTEM
    tok = models[voice][1]
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Write a LinkedIn post about: {ask.topic.strip()}"},
        ],
        add_generation_prompt=True,
    )

    def stream():
        out_q: queue.Queue = queue.Queue()
        jobs.put((voice, prompt, out_q))
        while (t := out_q.get()) is not None:
            yield {"token": t}

    return StreamingResponse(sse(stream()), media_type="text/event-stream")


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)
