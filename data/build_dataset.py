"""Build MLX chat-format train/valid JSONL from Rohit's logged LinkedIn posts.

Pairs: brief (blockquote on the wiki post page) -> final copy (assets/<stem>_post.txt).
"""
import json
import random
import re
from pathlib import Path

WIKI = Path.home() / "Documents/wiki/linkedin"
OUT = Path(__file__).parent
SYSTEM = (
    "You are Rohit Gupta, a backend/React developer who builds AI agents and "
    "posts about them on LinkedIn. Write in his voice: a bold hook line, a "
    "personal build story, em-dash bullets with concrete receipts, and a "
    "short mission close."
)

def brief_for(stem: str) -> str | None:
    md = WIKI / f"{stem}.md"
    if not md.exists():
        return None
    for line in md.read_text().splitlines():
        if line.startswith("> "):
            # first sentence of the blockquote is the topic brief
            text = line[2:].strip()
            text = re.sub(r"\[\[(.+?)\]\]", r"\1", text)  # strip wikilinks
            return text.split(";")[0].split(". ")[0][:300]
    return None

pairs = []
for txt in sorted(WIKI.glob("assets/*_post.txt")):
    stem = txt.name.removesuffix("_post.txt")
    brief = brief_for(stem)
    post = txt.read_text().strip()
    if brief and post:
        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Write a LinkedIn post about: {brief}"},
                {"role": "assistant", "content": post},
            ]
        })

random.seed(42)
random.shuffle(pairs)
n_valid = max(2, len(pairs) // 10)
valid, train = pairs[:n_valid], pairs[n_valid:]

for name, rows in [("train", train), ("valid", valid)]:
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"{len(train)} train / {len(valid)} valid examples")
assert len(train) >= 20, "corpus shrank — check wiki assets"
