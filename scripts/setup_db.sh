#!/bin/bash
# Setup PostgreSQL database for VecAdvisor++
# Prerequisites: PostgreSQL installed with pgvector extension available

set -e

DB_NAME="${1:-vecadvisor}"

echo "Creating database: $DB_NAME"
createdb "$DB_NAME" 2>/dev/null || echo "Database '$DB_NAME' already exists."

echo "Enabling pgvector extension..."
psql "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "Setup complete. Database '$DB_NAME' is ready with pgvector."
