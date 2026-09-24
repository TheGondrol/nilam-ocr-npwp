"""The `-test` twins of a service's pipeline endpoints (`TESTING_ENDPOINTS`).

A twin is the same handler, documentation and API-key check as the live route, registered at another path
with some of its dependencies swapped, e.g. `service=get_testing_job_service`. The handler code is not
copied, so a load test on a twin measures exactly what the live endpoint does.
"""

import functools
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.routing import APIRoute

from ocr_common.testing_endpoints import TESTING_TAG

TESTING_NOTE = (
    "**Testing twin, for load tests (`TESTING_ENDPOINTS`).** Runs exactly like the live endpoint described "
    "below, but on the `testing_*` tables, handing off to the next stage's `-test` endpoint, without any "
    "callback and without writing to the orchestrator's tables.\n\n"
)


def build_testing_router(
    live: APIRouter, paths: Mapping[str, str], *, note: str = "", **dependencies: Callable[..., Any]
) -> APIRouter:
    """A router with the twin of each route of `live` whose path is a key of `paths`, at the path it maps
    to. Every handler parameter named in `dependencies` gets `Depends(<the given one>)` as its source: a
    swapped dependency (`service=get_testing_job_service`), or a value made here instead of sent by the
    caller (guardrails' `request_id`). `note` is added to the twins' description, after `TESTING_NOTE`.
    Raises when a path is not a route of `live`, so a renamed endpoint cannot lose its twin."""
    router = APIRouter(tags=[TESTING_TAG], dependencies=list(live.dependencies))
    routes = {route.path: route for route in live.routes if isinstance(route, APIRoute)}
    missing = [path for path in paths if path not in routes]
    if missing:
        raise ValueError(f"no live route for {', '.join(missing)}")
    for live_path, testing_path in paths.items():
        route = routes[live_path]
        router.add_api_route(
            testing_path,
            _with_dependencies(route.endpoint, dependencies),
            methods=sorted(route.methods),
            status_code=route.status_code,
            response_model=route.response_model,
            responses=route.responses,
            summary=f"[Testing] {route.summary}" if route.summary else None,
            description=TESTING_NOTE + (f"{note}\n\n" if note else "") + (route.description or ""),
            operation_id=f"{route.operation_id}Test" if route.operation_id else None,
        )
    return router


def _with_dependencies(endpoint: Callable[..., Any], dependencies: Mapping[str, Callable[..., Any]]):
    """`endpoint` with the default `Depends(...)` of the named parameters replaced. Only the parameters
    `endpoint` actually has are replaced: one router can share a set of dependencies across handlers."""
    signature = inspect.signature(endpoint)
    parameters = [
        parameter.replace(default=Depends(dependencies[parameter.name]))
        if parameter.name in dependencies
        else parameter
        for parameter in signature.parameters.values()
    ]

    @functools.wraps(endpoint)
    async def twin(*args: Any, **kwargs: Any) -> Any:
        return await endpoint(*args, **kwargs)

    # FastAPI reads the parameters from __signature__; setattr because functools' wrapper type does not declare it.
    setattr(twin, "__signature__", signature.replace(parameters=parameters))  # noqa: B010
    return twin
