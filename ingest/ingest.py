"""
Main script for ingesting QTM files based on project settings.
"""
import argparse
import os
import sys
import logging
from typing import Optional

try:
    import mysql.connector
except ImportError:
    print(
        "Missing dependency: mysql-connector-python. Install with: pip install mysql-connector-python",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from sshtunnel import SSHTunnelForwarder
except ImportError:
    SSHTunnelForwarder = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .database import Database
from .project_structure import ProjectStructure
from .settings_parser import parse_settings_file


def load_env():
    """Load environment variables from .env file."""
    if load_dotenv is None:
        return

    # Try project root .env first, then alongside this script
    root_dir = os.path.dirname(os.path.dirname(__file__))
    env_path_root = os.path.join(root_dir, ".env")
    env_path_local = os.path.join(os.path.dirname(__file__), ".env")

    if os.path.isfile(env_path_root):
        load_dotenv(env_path_root)
    elif os.path.isfile(env_path_local):
        load_dotenv(env_path_local)
    else:
        load_dotenv()


def get_connection(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    ssh_config: Optional[dict] = None
):
    """Get a database connection, optionally through SSH tunnel."""
    tunnel = None
    if ssh_config:
        if SSHTunnelForwarder is None:
            print(
                "Missing dependency: sshtunnel. Install with: pip install sshtunnel",
                file=sys.stderr,
            )
            sys.exit(1)

        tunnel = SSHTunnelForwarder(
            (ssh_config["host"], ssh_config["port"]),
            ssh_username=ssh_config["user"],
            ssh_pkey=ssh_config["key"],
            ssh_private_key_password=ssh_config.get("passphrase"),
            remote_bind_address=("127.0.0.1", 3306),
            local_bind_address=("127.0.0.1", 0),
        )
        tunnel.start()
        db_host = "127.0.0.1"
        db_port = tunnel.local_bind_port
    else:
        db_host = host
        db_port = port

    return mysql.connector.connect(
        host=db_host,
        port=db_port,
        user=user,
        password=password,
        database=database,
        autocommit=False,
    ), tunnel


def ingest_qtm_files(settings_file: str, base_dir: str, db: Database):
    """Main ingestion function."""
    settings = parse_settings_file(settings_file)
    project = ProjectStructure(settings, base_dir)
    qtm_files = project.scan_for_qtm_files()
    for file_info in qtm_files:
        participant_name = file_info.get("participant_name", file_info.get("participant_id"))
        session_name = file_info.get("session_name", file_info.get("session_type"))
        session_type = file_info.get("session_type", "Task session")
        logging.info("Processing file: %s", file_info.get("file_path"))

        participant_uuid = db.get_or_create_participant(
            name=participant_name
        )

        session_uuid = db.get_or_create_session(
            name=session_name,
            participant_uuid=participant_uuid,
            type_=session_type
        )

        # Add QTM record, using new schema fields
        if getattr(db, "dry_run", False):
            logging.info("Dry-run: would add qtm record for %s (session: %s)", file_info.get("file_path"), session_uuid)
        else:
            db.add_qtm_record(
                session_uuid=session_uuid,
                file_path=file_info["file_path"],
                trial=file_info.get("trial"),
                repetition=file_info.get("repetition"),
                type_=file_info.get("type"),  # measurement type
                start_time=file_info.get("start_time"),
                valid=file_info.get("valid", 1)
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest QTM files based on project settings"
    )
    
    # Database connection options
    parser.add_argument(
        "--host",
        default=os.environ.get("DB_HOST", "127.0.0.1"),
        help="Database host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DB_PORT", "3306")),
        help="Database port"
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("DB_USER", "root"),
        help="Database user"
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("DB_PASSWORD", ""),
        help="Database password"
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("DB_NAME", "escience_moves"),
        help="Database name"
    )
    # Dry-run option
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and log actions without writing to the database",
    )
    
    # SSH tunnel options
    parser.add_argument(
        "--ssh-host",
        default=os.environ.get("SSH_HOST"),
        help="SSH host for tunneling"
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=int(os.environ.get("SSH_PORT", "22")),
        help="SSH port"
    )
    parser.add_argument(
        "--ssh-user",
        default=os.environ.get("SSH_USER"),
        help="SSH username"
    )
    parser.add_argument(
        "--ssh-key",
        default=os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/id_rsa")),
        help="SSH private key path"
    )
    parser.add_argument(
        "--ssh-passphrase",
        default=os.environ.get("SSH_PASSPHRASE"),
        help="SSH key passphrase"
    )
    
    # Project options
    parser.add_argument(
        "settings_file",
        help="Path to the Settings.paf file"
    )
    parser.add_argument(
        "base_dir",
        help="Base directory containing QTM files"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Load environment variables
    load_env()
    
    # Parse command line arguments
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    # Set up SSH config if needed
    ssh_config = None
    if args.ssh_host:
        ssh_config = {
            "host": args.ssh_host,
            "port": args.ssh_port,
            "user": args.ssh_user,
            "key": args.ssh_key,
            "passphrase": args.ssh_passphrase,
        }
    
    # Connect to database
    connection, tunnel = get_connection(
        args.host,
        args.port,
        args.user,
        args.password,
        args.database,
        ssh_config
    )
    
    try:
        settings = parse_settings_file(args.settings_file)
        db = Database(connection, settings)
        # Attach dry_run flag to db for use in ingest_qtm_files
        setattr(db, "dry_run", getattr(args, "dry_run", False))
        ingest_qtm_files(args.settings_file, args.base_dir, db)
        print("Ingestion completed successfully.")
    finally:
        connection.close()
        if tunnel:
            tunnel.stop()


if __name__ == "__main__":
    main()