import os
import platform
import subprocess

from .config import DOCKER_IMAGE


def _run_docker_cmd(args, timeout=3600, stream=False):
    kwargs = {
        "text": True,
        "timeout": timeout,
    }
    if not stream:
        kwargs["capture_output"] = True

    result = subprocess.run(args, **kwargs)
    if result.returncode != 0:
        cmd = " ".join(args)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {cmd}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    return result


def ensure_docker_image(image, docker_username=None):
    """Ensure a benchmark Docker image exists locally."""
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        return

    _run_docker_cmd(["docker", "pull", image], timeout=3600, stream=True)


def run_in_docker(script_argv, mounts, image=DOCKER_IMAGE, gpus=True,
                  workdir=None, timeout=600, env=None):
    """Run a command inside a Docker container.

    Mirrors cli.py:339-355 exactly.
    """
    ensure_docker_image(image)

    cmd = ["docker", "run", "--rm"]
    if gpus:
        cmd += ["--gpus", "all"]
    if platform.system() == "Linux":
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}",
                "-e", "HOME=/tmp",
                "-e", "USER=benchuser"]
    for host, container in mounts:
        cmd += ["-v", f"{host}:{container}"]
    if workdir:
        cmd += ["-w", workdir]
    if env:
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
    cmd += [image] + list(script_argv)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
