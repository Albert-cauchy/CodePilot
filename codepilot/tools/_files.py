"""Transport file operations through the existing sandbox execution interface."""

import inspect
import json
import shlex

from codepilot.sandbox.base import BaseSandbox, SandboxError
from codepilot.tools import _file_operations

SCRIPT = (
    inspect.getsource(_file_operations)
    + """
import json
import sys

try:
    print(execute_operation(json.loads(sys.argv[1])))
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
"""
)


def execute_file_operation(sandbox: BaseSandbox, operation: str, **arguments) -> str:
    payload = json.dumps({"operation": operation, **arguments}, ensure_ascii=True)
    command = f"python -I -S -c {shlex.quote(SCRIPT)} {shlex.quote(payload)}"
    result = sandbox.execute(command)
    if result.timed_out:
        raise SandboxError(f"{operation} timed out for {arguments['path']}.")
    if not result.ok:
        raise SandboxError(result.output.strip() or f"{operation} failed for {arguments['path']}.")
    return result.stdout.rstrip("\n")
