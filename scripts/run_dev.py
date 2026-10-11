"""Run and evaluate only the pinned SWE-bench development task selection."""

from scripts.run_benchmark import main

if __name__ == "__main__":
    main(development=True)
