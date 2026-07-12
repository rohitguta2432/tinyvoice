# tinyvoice

A 0.5B-parameter model fine-tuned to write LinkedIn posts in **my** voice — trained
in an afternoon, on a laptop, on my own 28 published posts. No GPU cluster, no cloud.

The web UI puts the stock model and the tuned one side by side: give both the same
topic and watch the difference stream in.

![demo](demo.gif)

## How it works

1. **Dataset** — `data/build_dataset.py` pairs each published post's one-line brief
   with its final copy → 26 train / 2 valid chat examples.
2. **Fine-tune** — [MLX](https://github.com/ml-explore/mlx) LoRA on
   `Qwen2.5-0.5B-Instruct-4bit`, ~3 minutes on Apple silicon:

   ```sh
   uv run mlx_lm.lora --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
     --train --data data --iters 400 --batch-size 2 \
     --learning-rate 1e-4 --adapter-path adapters --max-seq-length 2048
   ```

3. **Serve** — one FastAPI file loads the model twice (with and without the
   adapter) and streams both over SSE:

   ```sh
   uv run python app/main.py   # → http://127.0.0.1:7860
   ```

## Bring your own voice

Point `data/build_dataset.py` at any folder of your own writing (posts, emails,
essays) as `prompt → text` pairs and re-run the two commands above. The whole loop —
data, train, serve — is three commands.

## Stack

- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LoRA fine-tuning + inference on Apple silicon
- FastAPI + vanilla JS — no build step, one HTML file

## License

MIT
