from .base import (
    BaseSandbox,
    ExecResult,
    RepositorySandbox,
    SandboxCommandError,
    SandboxError,
)

__all__ = [
    "BaseSandbox",
    "ExecResult",
    "RepositorySandbox",
    "SandboxCommandError",
    "SandboxError",
    "DockerSandbox",
    "DockerSandboxConfig",
    "swebench_image_name",
]


def __getattr__(name: str):
    if name in {"DockerSandbox", "DockerSandboxConfig", "swebench_image_name"}:
        from .docker import DockerSandbox, DockerSandboxConfig, swebench_image_name

        return {
            "DockerSandbox": DockerSandbox,
            "DockerSandboxConfig": DockerSandboxConfig,
            "swebench_image_name": swebench_image_name,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
