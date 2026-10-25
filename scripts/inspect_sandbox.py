from codepilot.benchmark import SWEBenchAdapter
from codepilot.sandbox import DockerSandbox


INSTANCE_ID = "sympy__sympy-20590"


def show(label: str, result) -> None:
    print(f"\n== {label} ==")
    print(f"exit_code: {result.exit_code}")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[stderr]")
        print(result.stderr)


def main() -> None:
    task = SWEBenchAdapter().load_task(INSTANCE_ID)

    sandbox = DockerSandbox(task)
    print(f"image: {sandbox.image_name}")

    with sandbox:
        show("pwd", sandbox.execute("pwd", check=True))
        show("HEAD", sandbox.execute("git rev-parse HEAD", check=True))
        show("recent commits", sandbox.execute("git log --oneline -3", check=True))
        show("status", sandbox.execute("git status --short", check=True))
        show("python", sandbox.execute("python --version", check=True))
        show("repo sample", sandbox.execute("ls | head -20", check=True))

        print()
        print(f"dataset base_commit: {task.base_commit}")
        print(f"image initial HEAD:  {sandbox.initial_head}")
        print(f"base is ancestor:    {sandbox.base_commit_is_ancestor()}")

        assert sandbox.base_commit_is_ancestor()
        assert sandbox.git_status() == ""

    print("\nSandbox smoke test passed and container was removed.")


if __name__ == "__main__":
    main()
