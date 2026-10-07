import json
import logging

from ocr_common.web.logging import JsonFormatter, ProbeAccessFilter, log_event
from ocr_common.web.request_id import bind_request_id, reset_request_id


def _record(name: str, msg: str, args: tuple = (), **attrs) -> logging.LogRecord:
    record = logging.LogRecord(name, logging.INFO, __file__, 1, msg, args, None)
    for key, value in attrs.items():
        setattr(record, key, value)
    return record


def _access(path: str, status: int) -> logging.LogRecord:
    return _record("uvicorn.access", '%s - "%s %s HTTP/%s" %d', ("10.0.0.1:1", "GET", path, "1.1", status))


def test_a_json_line_carries_the_ecs_fields_elasticsearch_reads():
    line = json.loads(JsonFormatter("scoring").format(_record("x", "hello", request_id="REQ_1")))

    assert line["@timestamp"] == line["time"]
    assert (line["log.level"], line["severity"]) == ("info", "INFO")
    assert (line["service"], line["request_id"], line["message"]) == ("scoring", "REQ_1", "hello")


def test_structured_fields_land_on_the_line_as_objects_without_overriding_the_fixed_ones(caplog):
    logger = logging.getLogger("test.events")
    token = bind_request_id("REQ_2")
    try:
        with caplog.at_level(logging.INFO, logger="test.events"):
            log_event(logger, logging.INFO, "sent", event={"dataset": "outbox"}, outbox={"attempt": 2}, message="x")
    finally:
        reset_request_id(token)
    [record] = caplog.records
    record.request_id = "REQ_2"

    line = json.loads(JsonFormatter(None).format(record))

    assert line["event"] == {"dataset": "outbox"}
    assert line["outbox"] == {"attempt": 2}
    assert line["message"] == "sent", "a field never replaces the fixed ones"


def test_a_successful_probe_leaves_no_access_line_and_everything_else_does():
    probe = ProbeAccessFilter()

    assert not probe.filter(_access("/ready", 200))
    assert not probe.filter(_access("/health?verbose=1", 200))
    assert not probe.filter(_access("/metrics", 200))
    assert probe.filter(_access("/ready", 503)), "a failing probe is worth a line"
    assert probe.filter(_access("/v1/extract-ocr", 200))
    assert probe.filter(_record("ocr_common.pipeline.outbox", "not an access line"))
