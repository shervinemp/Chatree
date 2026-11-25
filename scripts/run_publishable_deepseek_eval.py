"""
Backward-compatible wrapper around the provider-agnostic `run_publishable_eval.py`
defaulting `--out-dir` to `results/deepseek_eval` and `--model` to `deepseek/deepseek-chat`.
Pass `--model <provider>/<model>` to run against any other provider seamlessly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_publishable_eval import run_main

if __name__ == "__main__":
    asyncio.run(run_main(default_out_dir="results/deepseek_eval"))
