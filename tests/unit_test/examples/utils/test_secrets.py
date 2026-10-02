import sys

import pytest

from utils import secrets as secrets_module

pytestmark = pytest.mark.unit


def test_env_wins_and_azure_is_never_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SECRET", "env-value")
    monkeypatch.delitem(sys.modules, "azure.keyvault.secrets", raising=False)

    assert secrets_module.get_secret("MY_SECRET", key_vault_name="vault") == "env-value"
    assert "azure.keyvault.secrets" not in sys.modules


def test_without_a_vault_a_missing_secret_says_where_to_set_it() -> None:
    with pytest.raises(
        LookupError, match="Export it or add it to .env, or set AZURE_KEY_VAULT_NAME"
    ):
        secrets_module.get_secret("MY_SECRET")
    assert secrets_module.get_secret("MY_SECRET", required=False) is None


def test_with_a_vault_the_kebab_case_twin_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def __init__(self) -> None:
            self.asked: list[str] = []

        def get_secret(self, name: str) -> object:
            self.asked.append(name)
            return type("Secret", (), {"value": "from-vault"})()

    client = _Client()
    monkeypatch.setattr(secrets_module, "_get_secret_client", lambda vault: client)

    assert (
        secrets_module.get_secret("MY_SECRET", key_vault_name="vault") == "from-vault"
    )
    assert client.asked == ["my-secret"]
