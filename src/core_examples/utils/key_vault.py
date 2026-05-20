import os
from functools import lru_cache

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


def _to_keyvault_name(name: str) -> str:
    """
    Convert an environment-style secret name (UPPER_SNAKE_CASE)
    to Azure Key Vault naming convention (kebab-case).

    Example:
        AZURE_CLIENT_SECRET -> azure-client-secret
    """
    return name.lower().replace("_", "-")


@lru_cache(maxsize=1)
def _get_secret_client(key_vault_name: str | None = None) -> SecretClient:
    key_vault_name = key_vault_name or os.getenv("AZURE_KEY_VAULT_NAME")
    if not key_vault_name:
        raise OSError("AZURE_KEY_VAULT_NAME environment variable is not set")

    key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    return SecretClient(vault_url=key_vault_uri, credential=DefaultAzureCredential())


def get_secret(
    secret_name: str,
    *,
    required: bool = True,
    key_vault_name: str | None = None,
) -> str | None:
    """
    Retrieve a secret value using a dual-source strategy:
    1. First attempts to read from environment variables.
    2. Falls back to Azure Key Vault if not found locally.

    This allows seamless usage across local development and cloud environments.

    Args:
        secret_name:
            Setting or secret name to resolve, using the same identifier expected
            in environment variables.
        required:
            When ``True`` a missing Key Vault secret raises an exception. When
            ``False`` a missing secret resolves to ``None``.
        key_vault_name:
            Optional Key Vault name override. When omitted, the helper falls back
            to ``AZURE_KEY_VAULT_NAME`` from the process environment.

    Returns:
        str | None:
            The secret value, or ``None`` when ``required=False`` and the
            secret is intentionally absent.

    Raises:
        EnvironmentError:
            If the required key_vault_name argument or environment variable
            'AZURE_KEY_VAULT_NAME' is not set.

        RuntimeError:
            If the secret cannot be retrieved from Azure Key Vault.

    Behavior:
        - Environment lookup uses the exact name provided.
        - Key Vault lookup converts the name to kebab-case
          (e.g., 'AZURE_CLIENT_SECRET' -> 'azure-client-secret').

    Example:
        >>> os.environ["AZURE_CLIENT_SECRET"] = "local-secret"
        >>> get_secret("AZURE_CLIENT_SECRET")
        'local-secret'

        If not present in environment:
        >>> get_secret("AZURE_CLIENT_SECRET")
        # Retrieves 'azure-client-secret' from Key Vault

    Notes:
        Environment lookup uses ``secret_name`` as-is. Key Vault lookup maps the
        same name to kebab-case, for example ``AZURE_CLIENT_SECRET`` becomes
        ``azure-client-secret``.
    """

    if secret_value := os.getenv(secret_name):
        return secret_value

    client = _get_secret_client(key_vault_name)
    kv_secret_name = _to_keyvault_name(secret_name)

    try:
        secret = client.get_secret(kv_secret_name)
        return secret.value
    except ResourceNotFoundError as exc:
        if not required:
            return None
        raise RuntimeError(
            f"Error retrieving secret '{secret_name}' (mapped to '{kv_secret_name}'): {str(exc)}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Error retrieving secret '{secret_name}' (mapped to '{kv_secret_name}'): {str(exc)}"
        ) from exc