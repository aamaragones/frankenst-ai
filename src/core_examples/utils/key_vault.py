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
def _get_secret_client() -> SecretClient:
    key_vault_name = os.getenv("AZURE_KEY_VAULT_NAME")
    if not key_vault_name:
        raise OSError("AZURE_KEY_VAULT_NAME environment variable is not set")

    key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    return SecretClient(vault_url=key_vault_uri, credential=DefaultAzureCredential())


def get_secret(secret_name: str, *, required: bool = True) -> str | None:
    """
    Retrieve a secret value using a dual-source strategy:
    1. First attempts to read from environment variables.
    2. Falls back to Azure Key Vault if not found locally.

    This allows seamless usage across local development and cloud environments.

    Args:
        secret_name (str):
            The name of the secret in environment variable format
            (e.g., 'AZURE_CLIENT_SECRET').
        required (bool):
            When ``True`` the lookup raises if the secret does not exist.
            When ``False`` a missing secret resolves to ``None`` so callers can
            fall back to managed identity or another default mechanism.

    Returns:
        str | None:
            The secret value, or ``None`` when ``required=False`` and the
            secret is intentionally absent.

    Raises:
        EnvironmentError:
            If the required Key Vault environment variable
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
        - Requires 'AZURE_KEY_VAULT_NAME' to be set in environment.
        - Uses DefaultAzureCredential for key vault client authentication.
        - Naming conventions differ between environment variables and
          Azure Key Vault secrets, so automatic mapping is applied.
    """
    # Attempt to retrieve from environment variables
    if secret_value := os.getenv(secret_name):
        return secret_value

    client = _get_secret_client()
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