import json
from pathlib import Path
import socket

import pytest

from threadmark import cli
from threadmark.datasets import demo_dataset


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline command attempted network access")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(cli, "getpass", blocked)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # An injected tokenizer avoids any first-use tokenizer cache download.
    monkeypatch.setattr("threadmark.context.token_counter", lambda: lambda text: 1)


def write_fixture(path: Path):
    path.write_text(json.dumps(demo_dataset()), encoding="utf-8")
    return str(path)


def test_demo_is_explicitly_scripted_and_has_no_network(no_network, capsys):
    assert cli.main(["demo"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert "SCRIPTED OFFLINE" in output["demonstration"]
    assert output["network_calls"] == 0
    assert output["planned_queries_per_arm"] == 2
    assert len(output["answers"]) == 4
    assert "query_accuracy_over_planned" not in json.dumps(output)


def test_inspect_and_prepare_never_read_a_key_or_make_network_calls(
    no_network, tmp_path, capsys
):
    source = write_fixture(tmp_path / "workflow.json")
    assert cli.main(["inspect", "--input", source]) == 0
    assert json.loads(capsys.readouterr().out)["planned_queries_per_arm"] == 2
    output = tmp_path / "prepared"
    assert (
        cli.main(
            ["compare", "--input", source, "--output", str(output), "--prepare-only"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["network_calls"] == 0
    assert (output / "request-plan.json").is_file()
    assert not (output / "budget.json").exists()
    assert not (output / "report.json").exists()
    assert (
        cli.main(
            ["compare", "--input", source, "--output", str(output), "--prepare-only"]
        )
        == 2
    )


def test_paid_run_requires_explicit_budget_and_prices_before_prompt(
    no_network, tmp_path
):
    source = write_fixture(tmp_path / "workflow.json")
    output = tmp_path / "paid"
    with pytest.raises(SystemExit) as error:
        cli.main(["compare", "--input", source, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_generated_test_data_requires_locked_protocol_hash_offline(
    no_network, tmp_path, capsys
):
    output = tmp_path / "test.json"
    assert cli.main(["generate", "--split", "test", "--output", str(output)]) == 2
    assert "locked protocol SHA256" in capsys.readouterr().err
    assert not output.exists()


def test_cli_does_not_accept_api_key_as_command_line_argument():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["demo", "--api-key", "never-a-real-key"])


def test_setup_is_an_explicit_separate_command(monkeypatch, capsys):
    calls = []

    def setup():
        calls.append("explicit setup")
        return {"status": "ready", "encoding": "o200k_base"}

    monkeypatch.setattr("threadmark.context.setup_tokenizer", setup)
    assert cli.main(["setup-tokenizer"]) == 0
    assert calls == ["explicit setup"]
    assert json.loads(capsys.readouterr().out)["encoding"] == "o200k_base"
