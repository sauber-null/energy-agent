import json

import pytest

from energy_agent.tools.registry import build_tools
from energy_agent.tools.switch_requests import get_switch_status, is_valid_malo_id, seed


@pytest.fixture
def tools():
    # Database and retriever are only touched when a tool actually runs a query.
    return {t.name: t for t in build_tools(conn=None, retriever=None)}


def test_schemas_are_function_specs_without_titles(tools):
    for tool in tools.values():
        spec = tool.schema()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == tool.name
        assert '"title"' not in json.dumps(spec)
    params = tools["calculate_deadline"].schema()["function"]["parameters"]
    assert params["required"] == ["process", "step"]
    assert "lieferbeginn" in params["properties"]["process"]["enum"]


def test_run_accepts_json_string(tools):
    result = tools["calculate_deadline"].run(
        '{"process": "lieferbeginn", "step": 5, "step1_sent_at": "2026-10-26T17:20"}'
    )
    assert result["deadline"] == "2026-10-27T11:00"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("calculate_deadline", {"process": "wechsel", "step": 1}),  # not in the enum
        ("calculate_deadline", {"process": "lieferbeginn", "step": 3}),  # no such rule
        ("calculate_deadline", "{not json"),
        ("get_switch_status", {}),  # neither request_id nor malo_id
        ("search_documents", {"query": "x", "k": 50}),  # k above the limit
    ],
)
def test_bad_calls_return_an_error_for_the_model(tools, name, arguments):
    assert "error" in tools[name].run(arguments)


def test_malo_id_check_digit():
    assert is_valid_malo_id("41373559241")  # BDEW example
    assert not is_valid_malo_id("41373559240")
    assert not is_valid_malo_id("4137355924")


@pytest.mark.integration
def test_switch_status_from_seeded_database(db_conn):
    seed(db_conn)
    result = get_switch_status(db_conn, request_id="lw-2026-0002")
    assert result["found"] == 1
    request = result["requests"][0]
    assert request["status"] == "abgelehnt"
    assert "nicht eindeutig" in request["note"]
    assert get_switch_status(db_conn, malo_id="51002003005")["requests"][0]["request_id"] == (
        "LW-2026-0003"
    )
    with pytest.raises(ValueError, match="MaLo-ID"):
        get_switch_status(db_conn, malo_id="51002003000")
