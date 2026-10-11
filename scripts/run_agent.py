import argparse
import os

from langchain_openai import ChatOpenAI

from codepilot.agent.runner import run_task
from codepilot.benchmark import SWEBenchAdapter
from codepilot.sandbox import DockerSandbox
from codepilot.submission import SubmissionWriter

MODEL_NAME = "qwen/qwen3.8-flash:free"


def create_model(
    *,
    rate_limiter=None,
    model_name=MODEL_NAME,
    base_url="https://llm.onerouter.pro/v1",
    api_key_env="INFRON_API_KEY",
    temperature=0,
    top_p=0.95,
    max_completion_tokens=4096,
    thinking=None,
):
    extra = {}
    if thinking is not None:
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=os.environ[api_key_env] if api_key_env else "local",
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_completion_tokens,
        timeout=120,
        rate_limiter=rate_limiter,
        **extra,
    )


def run(instance_id: str, output_path: str):
    # 1. Load sanitized SWE-bench task
    adapter = SWEBenchAdapter()
    task = adapter.load_task(instance_id)

    print(f"Instance: {task.instance_id}")
    print(f"Repo: {task.repo}")

    model = create_model()
    final_state, patch = run_task(task, model, DockerSandbox)

    print("\n===== FINAL PATCH =====")

    if patch.is_empty:
        print("Agent produced an empty patch.")
    else:
        print(patch.text)

    print("\nChanged files:")
    for path in patch.changed_files:
        print(f"  - {path}")

    prediction = SubmissionWriter.from_patch(
        task,
        patch,
        model_name_or_path=MODEL_NAME,
    )
    predictions = {
        item.instance_id: item for item in SubmissionWriter.read(output_path)
    }
    predictions[task.instance_id] = prediction
    output = SubmissionWriter.write(output_path, predictions.values())
    print(f"\nPrediction written to: {output}")

    return final_state, patch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--instance-id",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions.jsonl",
    )

    args = parser.parse_args()

    run(args.instance_id, args.output)


if __name__ == "__main__":
    main()
