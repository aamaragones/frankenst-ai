FROM mcr.microsoft.com/azure-functions/python:4-python3.12

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

# Define build arguments and set environment variables
ARG AZURE_KEY_VAULT_NAME

ENV AZURE_KEY_VAULT_NAME=${AZURE_KEY_VAULT_NAME}

# Install system dependencies and uv
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        unixodbc \
        curl \
        ca-certificates && \
    curl -fsSL https://astral.sh/uv/install.sh | sh && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:/home/.local/bin:$PATH"

# Install Python dependencies from the locked uv project
COPY pyproject.toml uv.lock /home/site/wwwroot/
RUN cd /home/site/wwwroot \
 && uv export --frozen --no-dev --no-emit-project --format requirements.txt > /tmp/uv-requirements.txt \
 && uv pip install --system --no-cache -r /tmp/uv-requirements.txt \
 && rm -f /tmp/uv-requirements.txt

# Prepare the Function App
COPY src/ /home/site/wwwroot/
RUN mv /home/site/wwwroot/services/functions/* /home/site/wwwroot/ && \
 rm -rf /home/site/wwwroot/services/functions