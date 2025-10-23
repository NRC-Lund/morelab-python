#!/bin/bash
# new_database.sh — Create MySQL database and users. Run locally on database server.
# Usage: ./new_database.sh <database_name>

if [ -z "$1" ]; then
  echo "Usage: $0 <database_name>"
  exit 1
fi

DB_NAME="$1"
USER1="$DB_NAME"
USER2="med-pha"

# Prompt for password securely
echo -n "Create database password for user '${USER1}': "
read -s PASS1
echo

# Execute MySQL commands
sudo mysql -e "
CREATE DATABASE IF NOT EXISTS ${DB_NAME};
CREATE USER IF NOT EXISTS '${USER1}'@'%' IDENTIFIED BY '${PASS1}';
GRANT ALL ON ${DB_NAME}.* TO '${USER1}'@'%';
GRANT ALL ON ${DB_NAME}.* TO '${USER2}'@'%';
FLUSH PRIVILEGES;
"

# Create tables from SQL file
echo "Creating tables from create_tables.sql..."
sudo mysql ${DB_NAME} < create_tables.sql

echo "✅ Database '${DB_NAME}' and user '${USER1}' created (or already exists)."