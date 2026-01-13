#!/bin/bash
# Setup environment variables for Databricks connection
# Source this file before running verification: source scripts/setup_env.sh

export DATABRICKS_HOST="dbc-83a50e08-9b04.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/e54f2a1ca9d9f622"

# Read token from file if it exists
TOKEN_FILE="../databricks_token.txt"
if [ -f "$TOKEN_FILE" ]; then
    export DATABRICKS_TOKEN=$(cat "$TOKEN_FILE")
    echo "✓ Environment variables set successfully"
    echo "  DATABRICKS_HOST: $DATABRICKS_HOST"
    echo "  DATABRICKS_HTTP_PATH: $DATABRICKS_HTTP_PATH"
    echo "  DATABRICKS_TOKEN: (loaded from $TOKEN_FILE)"
else
    echo "⚠ Token file not found at $TOKEN_FILE"
    echo "Please set manually: export DATABRICKS_TOKEN='your-token-here'"
fi

