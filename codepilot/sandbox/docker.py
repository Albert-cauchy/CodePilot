from __future__ import annotations

import io
import os
import platform
import re
import shlex
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import docker
from docker.errors import DockerException, ImageNotFound, NotFound

from codepilot.benchmark.base import Task

from .base import (
    ChildExecutionError,
    ExecResult,
    RepositorySandbox,
    SandboxCommandError,
    SandboxError,
)


def _default_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return "x86_64"


def _official_runtime_defaults() -> tuple[str | None, str]:
    try:
        from swebench.image_builder.constants import (
            CONTAINER_USER,
            CONTAINER_WORKDIR,
        )

        return CONTAINER_USER, CONTAINER_WORKDIR
    except Exception:
        return None, "/testbed"


def swebench_image_name(
    instance_id: str,
    *,
    namespace: str | None = "swebench",
    arch: str = "x86_64",
    tag: str = "latest",
) -> str:
    key = f"sweb.eval.{arch}.{instance_id.lower()}:{tag}"
    if namespace:
        return f"{namespace}/{key}".replace("__", "_1776_")
    return key


@dataclass(frozen=True, slots=True)
class DockerSandboxConfig:
    namespace: str | None = "swebench"
    arch: str = _default_arch()
    tag: str = "latest"
    auto_pull: bool = True
    network_disabled: bool = True
    cap_add_sys_admin: bool = False
    default_timeout: int = 120
    remove_on_stop: bool = True


class DockerSandbox(RepositorySandbox):
    """Disposable execution workspace backed by a SWE-bench instance image."""

    def __init__(
        self,
        task: Task,
        config: DockerSandboxConfig | None = None,
        *,
        client: docker.DockerClient | None = None,
    ) -> None:
        self.task = task
        self.config = config or DockerSandboxConfig()
        self.client = client or docker.from_env()
        self.container: Any = None

        # IMPORTANT:
        # A published SWE-bench instance image may contain a committed setup
        # change on top of dataset.base_commit. Therefore the image HEAD is the
        # real inference baseline; do not assume HEAD == task.base_commit.
        self._initial_head: str | None = None

        official_user, official_workdir = _official_runtime_defaults()
        self.container_user = official_user
        self.workdir = official_workdir

        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", task.instance_id.lower())
        self.container_name = f"codepilot.{safe_id}.{uuid.uuid4().hex[:8]}"

    @property
    def image_name(self) -> str:
        return swebench_image_name(
            self.task.instance_id,
            namespace=self.config.namespace,
            arch=self.config.arch,
            tag=self.config.tag,
        )

    @property
    def is_running(self) -> bool:
        if self.container is None:
            return False
        try:
            self.container.reload()
            return self.container.status == "running"
        except NotFound:
            return False

    @property
    def initial_head(self) -> str:
        if self._initial_head is None:
            raise SandboxError("Sandbox has not been started yet.")
        return self._initial_head

    def image_exists(self) -> bool:
        try:
            self.client.images.get(self.image_name)
            return True
        except ImageNotFound:
            return False

    def _ensure_image(self) -> None:
        if self.image_exists():
            return

        if not self.config.auto_pull:
            raise SandboxError(
                f"SWE-bench image is not available locally: {self.image_name}\n"
                f"Pull it with:\n  docker pull {self.image_name}"
            )

        try:
            self.client.images.pull(self.image_name)
        except DockerException as exc:
            raise SandboxError(f"Failed to pull SWE-bench image {self.image_name}: {exc}") from exc

    def start(self) -> None:
        if self.is_running:
            return

        try:
            self.client.ping()
        except DockerException as exc:
            raise SandboxError(
                "Docker daemon is not reachable. Verify `docker ps` works "
                "without sudo in the current shell."
            ) from exc

        self._ensure_image()

        create_kwargs: dict = {
            "image": self.image_name,
            "name": self.container_name,
            "detach": True,
            "command": "tail -f /dev/null",
            "network_disabled": self.config.network_disabled,
        }
        if self.container_user:
            create_kwargs["user"] = self.container_user
        if self.config.cap_add_sys_admin:
            create_kwargs["cap_add"] = ["SYS_ADMIN"]

        try:
            self.container = self.client.containers.create(**create_kwargs)
            self.container.start()
        except DockerException as exc:
            self.stop()
            raise SandboxError(
                f"Failed to create/start container from {self.image_name}: {exc}"
            ) from exc

        try:
            # The task base commit only needs to be an ancestor of the image
            # HEAD. The image HEAD itself becomes our reset baseline.
            self._initial_head = self.verify_task_revision()
            self.verify_clean_workspace()
        except Exception:
            self.stop()
            raise

    def _require_running(self):
        if not self.is_running:
            raise SandboxError(
                "Sandbox is not running. Call start() or use `with DockerSandbox(...)`."
            )
        return self.container

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
        check: bool = False,
        workdir: str | None = None,
    ) -> ExecResult:
        container = self._require_running()

        if timeout is None:
            timeout = self.config.default_timeout

        activation = (
            "source /opt/miniconda3/bin/activate >/dev/null 2>&1 || true; "
            "conda activate testbed >/dev/null 2>&1 || true; "
        )

        if timeout and timeout > 0:
            inner = shlex.quote(command)
            shell_script = (
                activation
                + f"timeout --signal=TERM --kill-after=5s {int(timeout)}s "
                + f"bash -c {inner}"
            )
        else:
            shell_script = activation + command

        started = time.monotonic()
        try:
            result = container.exec_run(
                ["/bin/bash", "-c", shell_script],
                workdir=workdir or self.workdir,
                user=self.container_user,
                demux=True,
            )
        except DockerException as exc:
            raise SandboxError(f"Docker exec failed for command {command!r}: {exc}") from exc

        duration = time.monotonic() - started
        stdout_bytes, stderr_bytes = result.output or (b"", b"")
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")

        exec_result = ExecResult(
            command=command,
            exit_code=int(result.exit_code),
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            timed_out=int(result.exit_code) in {124, 137},
        )

        if check and not exec_result.ok:
            raise SandboxCommandError(exec_result)

        return exec_result

    def head_commit(self) -> str:
        return self.execute(
            "git rev-parse HEAD",
            check=True,
        ).stdout.strip()

    def base_commit_is_ancestor(self) -> bool:
        """Check that this image really corresponds to the requested task.

        SWE-bench images may add their own committed setup change on top of
        task.base_commit, so checking HEAD equality is incorrect.
        """
        expected = shlex.quote(self.task.base_commit)
        result = self.execute(
            f"git merge-base --is-ancestor {expected} HEAD",
            timeout=30,
            check=False,
        )
        return result.exit_code == 0

    def verify_task_revision(self) -> str:
        actual = self.head_commit()

        if not self.base_commit_is_ancestor():
            raise SandboxError(
                "SWE-bench image/task mismatch: dataset base_commit is not "
                "an ancestor of the image HEAD.\n"
                f"  instance: {self.task.instance_id}\n"
                f"  base_commit: {self.task.base_commit}\n"
                f"  image HEAD:  {actual}\n"
                f"  image: {self.image_name}"
            )

        return actual

    def verify_clean_workspace(self) -> None:
        status = self.git_status()
        if status:
            raise SandboxError(
                "SWE-bench image started with a dirty worktree.\n"
                f"  instance: {self.task.instance_id}\n"
                f"  image HEAD: {self.head_commit()}\n"
                f"  status:\n{status}"
            )

    def git_status(self) -> str:
        return self.execute(
            "git status --short --untracked-files=all",
            check=True,
        ).stdout

    def prepare_patch_snapshot(self) -> None:
        """Make untracked, non-ignored files visible to git diff."""
        self.execute(
            "git add --intent-to-add -- .",
            check=True,
            timeout=60,
        )

    def git_diff(self, *, binary: bool = True) -> str:
        target = shlex.quote(self.initial_head)
        flag = " --binary" if binary else ""
        return self.execute(
            f"git -c core.fileMode=false diff{flag} {target} --",
            check=True,
        ).stdout

    def changed_files(self) -> list[str]:
        target = shlex.quote(self.initial_head)
        output = self.execute(
            f"git -c core.fileMode=false diff --name-only {target} --",
            check=True,
        ).stdout
        return [line for line in output.splitlines() if line.strip()]

    def reset_workspace(self) -> None:
        """Restore the exact clean state shipped by the instance image."""
        target = shlex.quote(self.initial_head)
        self.execute(
            f"git reset --hard {target} && git clean -fd",
            check=True,
            timeout=60,
        )

    def fork(self, baseline, allowed_files):
        baseline.validate_scope(list(allowed_files))
        return DockerChildSandbox(self, baseline, allowed_files)

    def stop(self) -> None:
        if self.container is None:
            return

        container = self.container
        self.container = None
        self._initial_head = None

        try:
            container.reload()
            if container.status == "running":
                container.stop(timeout=3)
        except NotFound:
            return
        except DockerException:
            pass

        if self.config.remove_on_stop:
            try:
                container.remove(force=True)
            except NotFound:
                pass
            except DockerException:
                pass


class DockerChildSandbox(DockerSandbox):
    """Same task image, private file contents, FUSE-enforced repository writes."""

    def __init__(self, parent, baseline, allowed_files):
        super().__init__(parent.task, parent.config, client=parent.client)
        self.parent = parent
        self.baseline = baseline
        self.allowed_files = tuple(allowed_files)
        self.workdir = parent.workdir
        self.storage = None
        self.policy = None

    @property
    def image_name(self):
        return self.parent.image_name

    def start(self):
        from .file_policy import RestrictedWorkspace

        self.storage = TemporaryDirectory(prefix="codepilot-child-")
        self.backing = Path(self.storage.name) / "private"
        try:
            self.baseline.materialize(self.backing)
            # Git history is readable, but never the controller's delta baseline.
            stream, _ = self.parent._require_running().get_archive(self.workdir + "/.git")
            with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as archive:
                for member in archive:
                    relative = Path(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise SandboxError("Unsafe Git archive member")
                    target = self.backing / relative
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = archive.extractfile(member)
                        if source is None:
                            raise SandboxError("Missing Git archive contents")
                        target.write_bytes(source.read())
                    else:
                        raise SandboxError("Git metadata must not contain links or devices")
            self.policy = RestrictedWorkspace(self.backing, self.allowed_files)
            mount = self.policy.__enter__()
            security = self.client.info().get("SecurityOptions", [])
            rootless = any("rootless" in option for option in security)
            # Match the FUSE mount owner's HOST identity; no allow_other needed.
            self.container_user = "0:0" if rootless else f"{os.getuid()}:{os.getgid()}"
            self.container = self.client.containers.create(
                image=self.image_name,
                name=self.container_name,
                detach=True,
                entrypoint=["/bin/sh", "-c"],
                command=["exec tail -f /dev/null"],
                user=self.container_user,
                read_only=True,
                network_disabled=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                volumes={str(mount): {"bind": self.workdir, "mode": "rw"}},
                tmpfs={"/tmp": "rw,nosuid,nodev", "/run": "rw,nosuid,nodev"},
                environment={
                    "HOME": "/tmp",
                    "TMPDIR": "/tmp",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": "/tmp/pycache",
                },
            )
            self.container.start()
            self._initial_head = self.parent.initial_head
            # Detect daemon mount-namespace/access problems before model calls.
            result = self.execute("test -d .git && test -r .git/HEAD", timeout=30)
            if not result.ok:
                raise SandboxError("Child container cannot access the enforced repository mount")
        except BaseException:
            self.stop()
            raise

    def changes_since(self, baseline):
        from ._repository_operations import capture
        from .repository import RepositorySnapshot

        # Stop background commands BEFORE reading the final private file state.
        execution_error = None
        try:
            if self.container is None:
                execution_error = "Child container is unavailable"
            else:
                self.container.reload()
                if self.container.status == "running":
                    self.container.stop(timeout=3)
                else:
                    execution_error = "Child container exited unexpectedly"
        except DockerException as exc:
            execution_error = f"Child container became unavailable: {exc}"
        changes = baseline.delta(RepositorySnapshot(capture(self.backing)))
        if execution_error:
            raise ChildExecutionError(execution_error, changes)
        return changes

    def stop(self):
        try:
            super().stop()
        finally:
            if self.policy is not None:
                self.policy.__exit__(None, None, None)
                self.policy = None
            if self.storage is not None:
                self.storage.cleanup()
                self.storage = None
