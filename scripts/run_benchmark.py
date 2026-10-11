import argparse
import json
import random
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from langchain_core.messages import message_to_dict
from langchain_core.rate_limiters import InMemoryRateLimiter

from codepilot.agent.runner import run_task
from codepilot.benchmark import SWEBenchAdapter
from codepilot.sandbox import DockerSandbox
from codepilot.submission import SubmissionWriter
from codepilot.trajectory import TrajectoryLogger
from scripts.run_agent import MODEL_NAME, create_model

DEFAULT_DEV_MANIFEST = (
    Path(__file__).resolve().parents[1] / "data/swebench_verified_dev_50.json"
)


def load_development_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = manifest["instance_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or len(set(ids)) != len(ids)
        or len(ids) != manifest["sample_size"]
    ):
        raise ValueError(
            "Development manifest must contain sample_size unique instance IDs"
        )
    if not all(manifest.get(key) for key in ("dataset", "split", "revision")):
        raise ValueError("Development manifest must pin dataset, split, and revision")
    return manifest


class LoggedModel:
    def __init__(self, model, logger, instance_id: str, attempt: int) -> None:
        self.model = model
        self.logger = logger
        self.instance_id = instance_id
        self.attempt = attempt
        self.step = 0

    def bind_tools(self, tools, **kwargs):
        return LoggedModel(
            self.model.bind_tools(tools, **kwargs),
            self.logger,
            self.instance_id,
            self.attempt,
        )

    def invoke(self, messages):
        self.step += 1
        details = {
            "instance_id": self.instance_id,
            "attempt": self.attempt,
            "step": self.step,
        }
        self.logger.log(
            "model_input",
            **details,
            messages=[message_to_dict(message) for message in messages],
        )
        try:
            response = self.model.invoke(messages)
        except Exception as exc:
            self.logger.log(
                "model_error", **details, error_type=type(exc).__name__, error=str(exc)
            )
            raise
        self.logger.log(
            "model_response",
            **details,
            message=message_to_dict(response),
        )
        return response


def evaluate_predictions(adapter, tasks, predictions, max_workers: int) -> dict:
    from swebench.harness.run_evaluation import main as run_evaluation

    run_id = datetime.now(timezone.utc).strftime("codepilot-%Y%m%d-%H%M%S-%f")
    selected_ids = [task.instance_id for task in tasks]
    selected_predictions = [
        predictions[instance_id]
        for instance_id in selected_ids
        if instance_id in predictions
    ]

    if not selected_predictions:
        print(f"SWE-bench resolved: 0/{len(tasks)} (no predictions)")
        return {
            "resolved_ids": [],
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": [],
            "incomplete_ids": selected_ids,
        }

    evaluation_input = Path("logs/evaluation") / run_id / "predictions.jsonl"
    SubmissionWriter.write(evaluation_input, selected_predictions)

    dataset_name = adapter.dataset_name
    if getattr(adapter, "revision", None):
        # The harness accepts a local JSON dataset, so it evaluates the same pinned rows.
        dataset_path = evaluation_input.parent / "dataset.json"
        dataset_path.write_text(
            json.dumps(
                [
                    dict(row)
                    for row in adapter._dataset
                    if row["instance_id"] in selected_ids
                ]
            ),
            encoding="utf-8",
        )
        dataset_name = str(dataset_path.resolve())

    print(f"Evaluating {len(selected_predictions)}/{len(tasks)} predictions...")
    report_path = run_evaluation(
        dataset_name=dataset_name,
        split=adapter.split,
        instance_ids=selected_ids,
        predictions_path=str(evaluation_input),
        max_workers=max_workers,
        open_file_limit=4096,
        run_id=run_id,
        timeout=1800,
        rewrite_reports=False,
        modal=False,
    )
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    statuses = {
        "resolved_ids": "RESOLVED",
        "unresolved_ids": "UNRESOLVED",
        "empty_patch_ids": "EMPTY_PATCH",
        "error_ids": "EVAL_ERROR",
        "incomplete_ids": "NO_PREDICTION",
    }
    by_id = {
        instance_id: status
        for field, status in statuses.items()
        for instance_id in report[field]
    }
    for instance_id in selected_ids:
        print(f"  {instance_id}: {by_id[instance_id]}")

    print(f"SWE-bench resolved: {len(report['resolved_ids'])}/{len(tasks)}")
    print(f"Evaluation report: {report_path}")
    return report


def main(*, development=False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--failure-log",
        default="outputs/failures.jsonl",
    )
    parser.add_argument(
        "--printlog",
        action="store_true",
        help="Log model inputs and replies to --failure-log",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--task-file",
        type=Path,
        default=DEFAULT_DEV_MANIFEST if development else None,
        help="Pinned development manifest containing instance_ids",
    )
    parser.add_argument(
        "--tool-config",
        choices=("two", "five"),
        default="five" if development else "two",
    )
    parser.add_argument(
        "--max-model-calls",
        type=int,
        default=500,
        help="Maximum normally returned model responses per experimental run",
    )
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument(
        "--model", default="Qwen/Qwen3.5-35B-A3B" if development else MODEL_NAME
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1"
        if development
        else "https://llm.onerouter.pro/v1",
    )
    parser.add_argument(
        "--api-key-env", default="" if development else "INFRON_API_KEY"
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True if development else None,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate saved predictions without running the agent",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=0.2,
    )
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.max_model_calls < 1:
        parser.error("--limit and --max-model-calls must be positive")
    if args.max_retries < 0 or args.eval_workers < 1 or args.requests_per_second <= 0:
        parser.error(
            "Retries must be nonnegative; workers and request rate must be positive"
        )
    args.output = args.output or (
        f"outputs/dev/{args.tool_config}/predictions.jsonl"
        if development
        else "outputs/predictions.jsonl"
    )
    manifest = load_development_manifest(args.task_file) if args.task_file else None
    if manifest:
        adapter = SWEBenchAdapter(
            dataset_name=manifest["dataset"],
            split=manifest["split"],
            revision=manifest["revision"],
        )
        ids = (
            manifest["instance_ids"][: args.limit]
            if args.limit
            else manifest["instance_ids"]
        )
        tasks = [adapter.load_task(instance_id) for instance_id in ids]
    else:
        adapter = SWEBenchAdapter()
        tasks = (
            list(islice(adapter.iter_tasks(), args.limit))
            if args.limit
            else list(adapter.iter_tasks())
        )

    trajectory_dir = args.trajectory_dir or (Path(args.output).parent / "trajectories")
    if development:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset": manifest["dataset"],
            "split": manifest["split"],
            "revision": manifest["revision"],
            "instance_ids": manifest["instance_ids"],
            "tool_configuration": args.tool_config,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "thinking": args.thinking,
            "max_completion_tokens": args.max_completion_tokens,
            "max_model_calls": args.max_model_calls,
        }
        metadata_path = Path(args.output).with_suffix(".run.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if args.eval_only and metadata_path.exists():
            args.tool_config = json.loads(metadata_path.read_text())[
                "tool_configuration"
            ]
        if (
            not args.eval_only
            and metadata_path.exists()
            and json.loads(metadata_path.read_text()) != metadata
        ):
            parser.error(
                "Saved run settings differ; use a new --output for changed conditions"
            )
        if not args.eval_only:
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )

    predictions = {
        prediction.instance_id: prediction
        for prediction in SubmissionWriter.read(args.output)
    }
    logger = TrajectoryLogger(args.failure_log)
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=args.requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
    model = None

    for index, task in enumerate(tasks, start=1):
        if args.eval_only:
            continue
        if task.instance_id in predictions:
            print(f"[{index}/{len(tasks)}] Skip {task.instance_id}")
            continue

        print(f"[{index}/{len(tasks)}] Run {task.instance_id}")
        patch = None
        if model is None:
            model = create_model(
                rate_limiter=rate_limiter,
                model_name=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                temperature=args.temperature,
                top_p=args.top_p,
                max_completion_tokens=args.max_completion_tokens,
                thinking=args.thinking,
            )

        for attempt in range(1, args.max_retries + 2):
            task_logger = (
                TrajectoryLogger(
                    trajectory_dir / f"{task.instance_id}.attempt-{attempt}.jsonl"
                )
                if development
                else logger
            )
            try:
                task_model = (
                    LoggedModel(model, task_logger, task.instance_id, attempt)
                    if development or args.printlog
                    else model
                )
                state, patch = run_task(
                    task,
                    task_model,
                    DockerSandbox,
                    tool_configuration=args.tool_config,
                    max_model_calls=args.max_model_calls,
                    logger=task_logger,
                )
                task_logger.log(
                    "run_completed",
                    instance_id=task.instance_id,
                    attempt=attempt,
                    tool_configuration=args.tool_config,
                    termination_reason=state.get("termination_reason"),
                    patch=patch.text,
                    changed_files=patch.changed_files,
                )
                break
            except Exception as exc:  # noqa: BLE001 - Record and retry task failures in fresh sandboxes.
                if development:
                    task_logger.log(
                        "run_failed",
                        instance_id=task.instance_id,
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                logger.log(
                    "task_failed",
                    instance_id=task.instance_id,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                print(f"  Attempt {attempt} failed: {type(exc).__name__}: {exc}")
                if attempt <= args.max_retries:
                    time.sleep(2 ** (attempt - 1) + random.random())

        if patch is None:
            print("  No prediction written.")
            continue

        print("  Changed files:")
        for path in patch.changed_files:
            print(f"    - {path}")

        predictions[task.instance_id] = SubmissionWriter.from_patch(
            task,
            patch,
            model_name_or_path=args.model,
        )
        SubmissionWriter.write(args.output, predictions.values())
        print(f"  Saved: {args.output}")

    completed = sum(task.instance_id in predictions for task in tasks)
    print(f"Completed {completed}/{len(tasks)} selected instances.")
    if tasks:
        report = evaluate_predictions(adapter, tasks, predictions, args.eval_workers)
        if development:
            result_path = Path(args.output).with_suffix(".results.json")
            result_path.write_text(
                json.dumps(
                    {
                        "instance_ids": [task.instance_id for task in tasks],
                        "tool_configuration": args.tool_config,
                        "resolved_count": len(report["resolved_ids"]),
                        "selected_count": len(tasks),
                        "resolved_at_1": len(report["resolved_ids"]) / len(tasks),
                        "report": report,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            TrajectoryLogger(trajectory_dir / "evaluation.jsonl").log(
                "evaluation_result", report=report
            )


if __name__ == "__main__":
    main()
