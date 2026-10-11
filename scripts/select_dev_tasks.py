"""Choose a reproducible, repository-stratified SWE-bench development set."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


DATASET_NAME = "SWE-bench/SWE-bench_Verified"
SPLIT = "test"
REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
SAMPLE_SIZE = 50
SEED = 42
DEFAULT_OUTPUT = Path("data/swebench_verified_dev_50.json")


def select_development_tasks(
    rows: Iterable[Mapping[str, object]],
    *,
    sample_size: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Allocate seats near repository quotas, then sample within each repository."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    by_repo: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        repo = str(row["repo"])
        instance_id = str(row["instance_id"])
        if not repo or not instance_id or instance_id in seen_ids:
            raise ValueError(f"Missing repository/instance ID or duplicate ID: {instance_id!r}")
        seen_ids.add(instance_id)
        by_repo[repo].append(instance_id)

    population = len(seen_ids)
    if len(by_repo) > sample_size or population < sample_size:
        raise ValueError(
            f"Cannot select {sample_size} tasks from {population} tasks "
            f"across {len(by_repo)} repositories while covering every repository"
        )

    repos = sorted(by_repo)
    quotas = {repo: sample_size * len(by_repo[repo]) / population for repo in repos}
    counts = {repo: 1 for repo in repos}
    for _ in range(sample_size - len(repos)):
        # Give the next seat to the repository furthest below its proportional
        # quota. Sorted repository names break equal-deficit ties consistently.
        repo = max(
            (repo for repo in repos if counts[repo] < len(by_repo[repo])),
            key=lambda repo: quotas[repo] - counts[repo],
        )
        counts[repo] += 1

    rng = random.Random(seed)
    selected = sorted(
        instance_id
        for repo in repos
        for instance_id in rng.sample(sorted(by_repo[repo]), counts[repo])
    )
    repository_counts = {
        repo: {"verified": len(by_repo[repo]), "selected": counts[repo]}
        for repo in repos
    }
    return selected, repository_counts


def build_manifest(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    selected, repository_counts = select_development_tasks(rows)
    return {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "revision": REVISION,
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "repository_counts": repository_counts,
        "instance_ids": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=SPLIT, revision=REVISION)
    manifest = build_manifest(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(manifest['instance_ids'])} development task IDs to {args.output}")


if __name__ == "__main__":
    main()
