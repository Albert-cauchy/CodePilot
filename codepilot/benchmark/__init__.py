from .base import Task, TaskSource

__all__ = ["SWEBenchAdapter", "Task", "TaskSource"]


def __getattr__(name: str):
    if name == "SWEBenchAdapter":
        from .swebench import SWEBenchAdapter

        return SWEBenchAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
