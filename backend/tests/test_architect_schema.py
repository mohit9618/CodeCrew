"""The architect's output schema tolerates common model mistakes."""
from backend.app.agents.architect import ArchitectureOutput


def test_list_for_a_text_field_is_joined_into_one_string():
    # Real failure: the model returned a list for existing_architecture and
    # the whole run died with a pydantic ValidationError.
    output = ArchitectureOutput.model_validate({
        "architecture_summary": ["Flask library", "docs under docs/"],
        "existing_architecture": ["Flask core", "pre-commit hooks"],
    })

    assert output.existing_architecture == "Flask core\npre-commit hooks"
    assert output.architecture_summary == "Flask library\ndocs under docs/"


def test_null_text_becomes_an_empty_string():
    output = ArchitectureOutput.model_validate({"existing_architecture": None})

    assert output.existing_architecture == ""


def test_plain_strings_are_unchanged():
    output = ArchitectureOutput.model_validate({"architecture_summary": "ok"})

    assert output.architecture_summary == "ok"


def test_contract_fields_are_optional_and_listed():
    assert ArchitectureOutput.model_validate({}).api_contract == []

    output = ArchitectureOutput.model_validate({
        "api_contract": ["GET /items -> 200 [{id: int}]"],
        "data_model": ["items(id INTEGER PK)"],
    })

    assert output.api_contract == ["GET /items -> 200 [{id: int}]"]
    assert output.data_model == ["items(id INTEGER PK)"]
