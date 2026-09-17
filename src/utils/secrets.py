"""Secret backend: env first, then Azure Key Vault. Imported only by config.settings."""

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from azure.keyvault.secrets import SecretClient


def _to_keyvault_name(name: str) -> str:
    """`AZURE_CLIENT_SECRET` -> `azure-client-secret`, the vault's naming rule."""
    return name.lower().replace("_", "-")


@lru_cache(maxsize=1)
def _get_secret_client(key_vault_name: str | None = None) -> "SecretClient":
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    key_vault_name = key_vault_name or os.getenv("AZURE_KEY_VAULT_NAME")
    if not key_vault_name:
        raise OSError("AZURE_KEY_VAULT_NAME environment variable is not set")
    return SecretClient(
        vault_url=f"https://{key_vault_name}.vault.azure.net",
        credential=DefaultAzureCredential(),
    )


@overload
def get_secret(
    secret_name: str,
    *,
    required: Literal[True] = True,
    key_vault_name: str | None = None,
) -> str: ...


@overload
def get_secret(
    secret_name: str, *, required: Literal[False], key_vault_name: str | None = None
) -> str | None: ...


def get_secret(
    secret_name: str, *, required: bool = True, key_vault_name: str | None = None
) -> str | None:
    """Return the env value under `secret_name`, else the vault's kebab-case twin.

    `required=False` turns a missing vault secret into `None`; any other vault
    failure is a `RuntimeError` naming both spellings of the secret.
    """
    if secret_value := os.getenv(secret_name):
        return secret_value

    from azure.core.exceptions import ResourceNotFoundError

    client = _get_secret_client(key_vault_name)
    kv_secret_name = _to_keyvault_name(secret_name)
    try:
        secret = client.get_secret(kv_secret_name)
    except ResourceNotFoundError as exc:
        if not required:
            return None
        raise RuntimeError(
            f"Secret '{secret_name}' (vault name '{kv_secret_name}') not found: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Error retrieving secret '{secret_name}' (vault name '{kv_secret_name}'): {exc}"
        ) from exc
    value: str | None = secret.value
    return value
