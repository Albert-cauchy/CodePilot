from codepilot.benchmark import SWEBenchAdapter


INSTANCE_ID = "sympy__sympy-20590"


def main() -> None:
    adapter = SWEBenchAdapter()
    task = adapter.load_task(INSTANCE_ID)

    print(f"dataset size: {len(adapter)}")
    print()
    print("instance_id:")
    print(task.instance_id)
    print()
    print("repo:")
    print(task.repo)
    print()
    print("base_commit:")
    print(task.base_commit)
    print()
    print("problem_statement:")
    print(task.problem_statement)


if __name__ == "__main__":
    main()
