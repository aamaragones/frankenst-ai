---
type: Concept
title: LLM Services
description: Provider-agnostic chat and embeddings runtimes (ollama, azure_ai, databricks) declared exclusively in config_llms.yaml, one runtime per launch key, with secrets resolved through settings.
tags: [llm-services, ollama, azure-ai, databricks, secrets, configuration, responses-api]
generated:
    by: reference_agent
    at: 2026-09-17T00:00:00Z
---

# LLM Services

`src/services/llm/llm_services.py` builds every chat model and embeddings client; the
rest of the code never instantiates a provider class. Providers: `ollama` (local),
`azure_ai` (Foundry) and `databricks` (serving endpoints / AI Gateway). The homologue in
`channel-planning` carries only `ollama` and `databricks`; a fix that applies to both
is made in both.

## The sanctioned configuration flow

**`config_llms.yaml` is the single configuration entry point.** One direction, two ends:

```
config_llms.yaml                  settings                          runtime
────────────────                  ────────                          ───────
kwargs + {secret: NAME}    ──►    get_settings().resolve_secret()  ──►  LLMServices.launch()
(values and VARIABLE NAMES)       env / .env, then the Key Vault        builds the LangChain
                                  named by AZURE_KEY_VAULT_NAME         client with resolved kwargs
```

Resolution is environment-first; the Key Vault lookup maps the name to kebab-case
(`AZURE_INFERENCE_MODEL_NAME` → `azure-inference-model-name`). `utils.secrets` is the
backend and only `config.settings` imports it: two readers of the same secret is how a
value ends up different in two places.

**Never construct `ChatOllama`, `AzureAIOpenAIApiChatModel` or `ChatDatabricks` in
application code**, and never call `utils.secrets.get_secret` directly.

## Contract

- One runtime per `launch` key. `launch.<kind>: <provider>` selects the provider and
  `<provider>.<kind>` holds the constructor kwargs. **Any kind name works**: adding
  `judge: databricks` under `launch` and a `databricks.judge` section is a new model,
  with no code change.
- The kind name picks the class: `embeddings` or `*_embeddings` builds the provider's
  embeddings class, anything else its chat class. The yaml section stays pure kwargs,
  so there is no marker to forget.
- `launch()` publishes one shared runtime per process (thread-safe); `force_reload=True`
  rebuilds it. `build_runtime(config)` is the pure, testable path.
- A launch key with no provider section raises
  `RuntimeError("launch.judge selects databricks but databricks.judge is not declared")`.
  It used to warn and skip, which surfaced later as an opaque `None` check at a call site.
- Unknown provider → `ValueError("Unsupported provider type: ...")`; empty or missing
  `launch` → `RuntimeError`.
- Consumers ask for what they need and get non-optional clients back:

```python
runtime = LLMServices.launch()
model, embeddings = runtime.require("model", "embeddings")   # RuntimeError names the missing kind
judge = runtime.judge                                        # attribute access, AttributeError if undeclared
LLMServices.get("model")                                     # RuntimeError before launch()
```

## Optional installs

Provider packages are imported inside each provider's importer (`_ollama_classes`,
`_azure_ai_classes`, `_databricks_classes`), which runs only when a launch key selects
that provider. An install may carry any subset of providers.

## Providers

| Provider | Chat / embeddings class | Local validation | Credential |
| --- | --- | --- | --- |
| `ollama` | `ChatOllama` / `OllamaEmbeddings` | `model` required; `host` becomes `base_url` through the WSL proxy helper | none |
| `azure_ai` | `AzureAIOpenAIApiChatModel` / `AzureAIOpenAIApiEmbeddingsModel` | exactly one of `endpoint`, `project_endpoint`; `model` required | `DefaultAzureCredential()` injected when the section declares none |
| `databricks` | `ChatDatabricks` / `DatabricksEmbeddings` | `model` names the endpoint; `endpoint` is the deprecated alias, and both together is an error | none: the SDK default auth chain |

Everything beyond that table is validated by the provider package itself.

## Example: the shipped file

```yaml
databricks:
  model:
    model: {secret: DATABRICKS_LLM_ENDPOINT}
    use_ai_gateway: true
    use_responses_api: true
    max_tokens: 1024
  embeddings:
    endpoint: {secret: DATABRICKS_EMBEDDINGS_ENDPOINT}

launch:
  model: ollama        # azure_ai | databricks
  embeddings: ollama
```

`ollama` ships `gemma4:e4b-it-qat` and `embeddinggemma`; `azure_ai` reads
`AZURE_FOUNDRY_PROJECT_ENDPOINT`, `AZURE_INFERENCE_MODEL_NAME`, `AZURE_EMBEDDINGS_ENDPOINT`
and `AZURE_EMBEDDINGS_MODEL_NAME` as secrets.

### `use_responses_api`, and why the yaml owns it

The flag is **data, declared per section**. The code used to default it to `False` for
`azure_ai`, which made a deployment fact (whether a Foundry region serves
`/v1/responses`) a Python constant; the yaml now says `false` for `azure_ai` and `true`
for `databricks`, each with its reason next to it.

Databricks needs it because the gateway model is a reasoning model and rejects
function tools on the chat-completions route:

> Function tools with reasoning_effort are not supported for … in
> `/v1/chat/completions`. To use function tools, use `/v1/responses`.

Only the **outbound** direction lives here: how this repo calls a provider. The
inbound one, a consumer reaching a served `ResponsesAgent`, belongs to
`channel-planning`.

Turning it on costs one adjustment in the examples. `StructuredGradeDocument` reads the
flag off the model with `getattr(model, "use_responses_api", False)` and takes a second
route: `with_structured_output` binds Chat Completions' `response_format`, which the
Responses route rejects with `TypeError: AsyncResponses.create() got an unexpected keyword
argument 'response_format'`; there the equivalent is `text.format` plus a
`PydanticOutputParser`. No example streams tokens, so the repeated-final-chunk parser
`channel-planning` needed is not ported.

## Authentication

| Provider | Mechanism |
| --- | --- |
| `azure_ai` | `DefaultAzureCredential`: `az login` locally, managed identity in Azure |
| `databricks` | SDK default chain: OAuth profile locally, ambient identity in-workspace, automatic through the AI Gateway |
| Key Vault | `DefaultAzureCredential` against the vault named by `AZURE_KEY_VAULT_NAME` |

No token is written in a yaml or a Python file.

## Settings integration

`config.settings` is the entry point for configuration **and** secrets. `AzureSettings`
fields flagged `key_vault_fallback` fill from Key Vault after env and `.env` said nothing
(`KeyVaultFallbackSettingsSource`), and `CoreSettings.resolve_secret(name)` is the door
for any ad hoc secret: `LLMServices`, the Functions orchestrators and the retriever tool
all go through it.

## Local validation

Unit tests stub the three provider importers and `settings.get_secret`, so no provider
install, network or vault is needed (`tests/unit_test/examples/services/test_llm_services.py`).
`tests/unit_test/examples/config/test_config_llms_yaml.py` builds the shipped file against every
provider, so a launch key without its section fails the PR, not the first run.
