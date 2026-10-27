"""Explicitly download pinned public research models, then verify their weights.

This operation downloads gigabytes for SmolLM2. Inference providers never invoke
it automatically. No API key or model API spend is needed for these public files.
"""

import argparse
import hashlib
import json
from pathlib import Path

MODELS = {
    "smollm2": {
        "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "revision": "31b70e2e869a7173562077fd711b654946d38674",
        "directory": "smollm2-1.7b-instruct",
        "weight_bytes": 3422777952,
        "weight_sha256": "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
        "files": [
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
    "minilm": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "directory": "all-MiniLM-L6-v2",
        "weight_bytes": 90868376,
        "weight_sha256": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
        "files": [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "sentence_bert_config.json",
            "modules.json",
            "1_Pooling/config.json",
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], required=True)
    parser.add_argument("--destination", type=Path, default=Path("models"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    for name in MODELS if args.model == "all" else [args.model]:
        spec = MODELS[name]
        directory = args.destination / spec["directory"]
        snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            local_dir=directory,
            allow_patterns=spec["files"],
            token=False,
        )
        weight = directory / "model.safetensors"
        value = hashlib.sha256()
        with weight.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(chunk)
        if (
            weight.stat().st_size != spec["weight_bytes"]
            or value.hexdigest() != spec["weight_sha256"]
        ):
            raise RuntimeError(
                f"{name}: downloaded weight failed its pinned size/SHA256 check"
            )
        manifest = {key: val for key, val in spec.items() if key != "files"}
        manifest["verified_weight_sha256"] = value.hexdigest()
        (directory / "verified-model.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: pinned model weights verified")


if __name__ == "__main__":
    main()
