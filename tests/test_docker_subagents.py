import os

import pytest

from codepilot.benchmark import Task
from codepilot.sandbox import DockerSandbox
from codepilot.sandbox.base import ChildExecutionError


@pytest.mark.docker
def test_docker_child_mount_enforces_scope_and_parent_accepts_delta():
    image = os.environ.get("CODEPILOT_DOCKER_TEST_IMAGE")
    if not image:
        pytest.skip("Set CODEPILOT_DOCKER_TEST_IMAGE to a local image with Python and Git")

    class TestImageSandbox(DockerSandbox):
        @property
        def image_name(self):
            return image

        def start(self):
            self.workdir = "/testbed"
            self.container_user = "0:0"
            self.container = self.client.containers.run(
                self.image_name,
                entrypoint=["/bin/sh", "-c"],
                command=["exec tail -f /dev/null"],
                detach=True,
                network_disabled=True,
            )
            self.container.exec_run(["mkdir", "-p", self.workdir])
            self.execute(
                "git init -q; git config user.email test@example.com; "
                "git config user.name Test; echo before > allowed.txt; "
                "echo protected > other.txt; git add .; git commit -qm initial",
                check=True,
            )
            self._initial_head = self.head_commit()

    task = Task("codepilot-test", "local/test", "unused", "Test isolation")
    with TestImageSandbox(task) as parent:
        parent.execute("echo 'parent uncommitted' > untracked.txt", check=True)
        baseline = parent.snapshot()
        with parent.fork(baseline, ["allowed.txt", "new.txt", "renamed.txt"]) as child:
            assert "parent uncommitted" in child.execute("cat untracked.txt", check=True).stdout
            assert child.execute("echo attack > other.txt").exit_code != 0
            child.execute(
                "echo fixed > allowed.txt; echo new > new.txt; mv new.txt renamed.txt", check=True
            )
            assert "protected" in child.execute("cat other.txt", check=True).stdout
            changes = child.changes_since(baseline)
        parent.apply_changes(changes)
        assert "fixed" in parent.execute("cat allowed.txt", check=True).stdout
        assert "new" in parent.execute("cat renamed.txt", check=True).stdout
        assert "protected" in parent.execute("cat other.txt", check=True).stdout
        second_baseline = parent.snapshot()
        with parent.fork(second_baseline, ["allowed.txt"]) as child:
            child.execute("echo partial > allowed.txt", check=True)
            child.container.stop(timeout=1)
            with pytest.raises(ChildExecutionError) as failure:
                child.changes_since(second_baseline)
            assert [change["path"] for change in failure.value.changes] == ["allowed.txt"]
        assert "fixed" in parent.execute("cat allowed.txt", check=True).stdout
