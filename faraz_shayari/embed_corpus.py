"""
Embed every couplet with Amazon Titan Embed v2 (multilingual) on Bedrock, and
save the vectors alongside the couplet metadata.

NOTE: We wanted Cohere Embed v4, but Cohere on Bedrock is an AWS *Marketplace*
subscription needing a valid payment instrument on the account (it failed with
INVALID_PAYMENT_INSTRUMENT). Titan is Amazon-native, works out of the box,
multilingual, 1024-dim. To switch to Cohere later, change MODEL_ID + payload.

Output: data/embeddings.npy (float32 [N, 1024]); order matches data/couplets.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import numpy as np
from dotenv import dotenv_values

HERE = Path(__file__).parent
COUPLETS = HERE / "data" / "couplets.jsonl"
OUT_EMB = HERE / "data" / "embeddings.npy"

CREDS = dotenv_values(HERE.parent / "aws-mcp-agent" / ".env")
MODEL_ID = "amazon.titan-embed-text-v2:0"      # Amazon-native, no Marketplace sub


def embed_one(rt, text: str) -> list[float]:
    body = {"inputText": text, "normalize": True, "dimensions": 1024}
    r = rt.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    return json.loads(r["body"].read())["embedding"]


def main() -> None:
    rows = [json.loads(l) for l in COUPLETS.read_text(encoding="utf-8").splitlines()]
    rt = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=CREDS["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=CREDS["AWS_SECRET_ACCESS_KEY"],
        region_name=CREDS["AWS_DEFAULT_REGION"],
    )
    print(f"embedding {len(rows)} couplets with {MODEL_ID} ...")
    vecs = []
    for i, r in enumerate(rows):
        vecs.append(embed_one(rt, r["couplet"]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}")
    emb = np.asarray(vecs, dtype=np.float32)      # Titan normalize=True -> unit vectors
    np.save(OUT_EMB, emb)
    print(f"✓ saved {emb.shape} -> {OUT_EMB}")


if __name__ == "__main__":
    main()
