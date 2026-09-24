"""Test hooks that ride on the file name, so a stage can be driven into a path that a real model
would not take on demand. Every hook is off unless the caller enables it, and the services only
enable them with ENVIRONMENT=local."""

import re

MAX_SIMULATED_DELAY_SECONDS = 120.0
_DELAY = re.compile(r"delay(\d+)s", re.IGNORECASE)


def simulated_delay_seconds(filename: str | None, *, enabled: bool) -> float:
    """`delay20s-npwp.jpg` makes the stage wait 20 s before its work, which is how the tracker shows
    the orchestrator answering 202 (the pipeline outlives PIPELINE_WAIT_SECONDS) without a slow model."""
    if not enabled or not filename:
        return 0.0
    match = _DELAY.search(filename)
    if match is None:
        return 0.0
    return min(float(match.group(1)), MAX_SIMULATED_DELAY_SECONDS)
