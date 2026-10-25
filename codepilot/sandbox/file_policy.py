"""A controller-owned FUSE view enforcing exact repository write permissions.

Only this view is mounted into child containers. The private backing directory
and this process are never exposed to a child. No shell command parsing is used.
"""

import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TextIO

from .base import SandboxError


class RestrictedWorkspace:
    def __init__(self, backing: Path, allowed_files):
        self.backing = backing.resolve()
        self.allowed = list(allowed_files)
        self.process: subprocess.Popen | None = None
        self.temporary: TemporaryDirectory | None = None
        self.log_file: TextIO | None = None

    def __enter__(self):
        if not Path("/dev/fuse").exists():
            raise SandboxError("Exact file permissions require Linux /dev/fuse and fuse3")
        self.temporary = TemporaryDirectory(prefix="codepilot-fuse-")
        self.mount = Path(self.temporary.name) / "repository"
        self.mount.mkdir()
        self.log_file = open(Path(self.temporary.name) / "fuse.log", "w+")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                __name__,
                str(self.backing),
                str(self.mount),
                json.dumps(self.allowed),
            ],
            stdout=self.log_file,
            stderr=self.log_file,
            start_new_session=True,
        )
        deadline = time.monotonic() + 10
        while self.process.poll() is None and time.monotonic() < deadline:
            if os.path.ismount(self.mount):
                return self.mount
            time.sleep(0.02)
        self.log_file.seek(0)
        detail = self.log_file.read()
        self.__exit__(None, None, None)
        raise SandboxError("Cannot mount enforced child workspace: " + detail)

    def __exit__(self, *_):
        if self.process is not None:
            if os.path.ismount(self.mount):
                subprocess.run(
                    ["fusermount3", "-uz", str(self.mount)],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.log_file:
            self.log_file.close()
        if self.temporary:
            self.temporary.cleanup()


def serve(backing: Path, mount: Path, allowed_files):
    # Import only in the mount process: ordinary agent use does not require FUSE.
    from mfusepy import FUSE, FuseOSError, Operations

    allowed = frozenset(allowed_files)
    directories = {
        str(parent)
        for name in allowed
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }

    class FilePolicy(Operations):
        use_ns = True

        def target(self, path, *, write=False, directory=False):
            relative = path.lstrip("/")
            pieces = PurePosixPath(relative).parts
            if ".." in pieces or ".git" in pieces and write:
                raise FuseOSError(errno.EACCES)
            target = backing / relative
            # Existing symlink parents cannot redirect operations into the backing
            # tree or outside it. The kernel can resolve links through readlink.
            for parent in target.parents:
                if parent == backing:
                    break
                if parent.is_symlink():
                    raise FuseOSError(errno.EACCES)
            if write:
                permitted = relative in (directories if directory else allowed)
                if not permitted or target.is_symlink():
                    raise FuseOSError(errno.EACCES)
            return target

        def getattr(self, path, fh=None):
            info = os.fstat(fh) if fh is not None else self.target(path).lstat()
            return {
                key: getattr(info, key)
                for key in (
                    "st_atime",
                    "st_ctime",
                    "st_gid",
                    "st_mode",
                    "st_mtime",
                    "st_nlink",
                    "st_size",
                    "st_uid",
                    "st_ino",
                )
            }

        def readdir(self, path, fh):
            return [".", "..", *os.listdir(self.target(path))]

        def readlink(self, path):
            return os.readlink(self.target(path))

        def access(self, path, mode):
            self.target(path, write=bool(mode & os.W_OK))
            return 0

        def open(self, path, flags):
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND))
            return os.open(self.target(path, write=writing), flags | os.O_NOFOLLOW)

        def create(self, path, mode, fi=None):
            return os.open(
                self.target(path, write=True),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode & 0o777,
            )

        def read(self, path, size, offset, fh):
            return os.pread(fh, size, offset)

        def write(self, path, data, offset, fh):
            self.target(path, write=True)
            return os.pwrite(fh, data, offset)

        def truncate(self, path, length, fh=None):
            target = self.target(path, write=True)
            if fh is not None:
                os.ftruncate(fh, length)
            else:
                os.truncate(target, length)
            return 0

        def unlink(self, path):
            self.target(path, write=True).unlink()
            return 0

        def rename(self, old, new):
            source = self.target(old, write=True)
            destination = self.target(new, write=True)
            if source.is_dir() or destination.is_dir():
                raise FuseOSError(errno.EACCES)
            os.replace(source, destination)
            return 0

        def mkdir(self, path, mode):
            self.target(path, write=True, directory=True).mkdir(mode=mode & 0o777)
            return 0

        def chmod(self, path, mode):
            self.target(path, write=True).chmod(mode & 0o777)
            return 0

        def utimens(self, path, times=None):
            target = self.target(path, write=True)
            if times is None:
                os.utime(target, None)
            else:
                os.utime(target, ns=times)
            return 0

        def flush(self, path, fh):
            return 0

        def fsync(self, path, datasync, fh):
            os.fsync(fh)
            return 0

        def release(self, path, fh):
            os.close(fh)
            return 0

        def deny(self, *args, **kwargs):
            raise FuseOSError(errno.EACCES)

        # Do not inherit permissive no-op metadata methods from Operations.
        symlink = link = mknod = rmdir = chown = setxattr = removexattr = deny

    FUSE(
        FilePolicy(),
        str(mount),
        foreground=True,
        nothreads=True,
        fsname="codepilot",
        attr_timeout=0,
        entry_timeout=0,
        negative_timeout=0,
    )


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))
