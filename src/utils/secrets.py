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
def _get_secret_client(key_vault_name: str) -> "SecretClient":
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

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

    Key Vault is consulted only when a vault is named; without one, a missing
    secret is a `LookupError` that says where to set it, and nothing from Azure is
    imported. `required=False` turns a missing secret into `None`; any other vault
    failure is a `RuntimeError` naming both spellings of the secret.
    """
    if secret_value := os.getenv(secret_name):
        return secret_value
    if key_vault_name is None:
        if not required:
            return None
        raise LookupError(
            f"Secret '{secret_name}' is not set. Export it or add it to .env, or set "
            "AZURE_KEY_VAULT_NAME to read it from Key Vault."
        )

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
