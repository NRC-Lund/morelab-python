"""
Database schema and operations for the enhanced QTM data ingestion.
"""
import os
import sys
import uuid
from typing import Dict, Optional, Tuple

import mysql.connector
from mysql.connector.cursor import MySQLCursor

try:
    from sshtunnel import SSHTunnelForwarder
except ImportError:
    SSHTunnelForwarder = None



class Database:
    def __init__(self, connection: mysql.connector.MySQLConnection, dry_run: bool = False):
        self.conn = connection
        self.dry_run = dry_run
        self.tunnel = None

    @classmethod
    def create_connection(
        cls,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        ssh_config: Optional[dict] = None
    ) -> Tuple['Database', Optional[object]]:
        """Create a database connection, optionally through SSH tunnel."""
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

        connection = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=user,
            password=password,
            database=database,
            autocommit=False,
        )
        
        return connection, tunnel

    @classmethod
    def create(
        cls,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        ssh_config: Optional[dict] = None,
        dry_run: bool = False
    ) -> 'Database':
        """Create a Database instance with connection setup."""
        connection, tunnel = cls.create_connection(
            host, port, user, password, database, ssh_config
        )
        db = cls(connection, dry_run)
        db.tunnel = tunnel
        return db

    def close(self):
        """Close database connection and tunnel."""
        if self.conn:
            self.conn.close()
        if self.tunnel:
            self.tunnel.stop()

    def get_or_create_participant(self, name: str, type_: str = None, sex: str = None, date_of_birth: str = None) -> tuple[str, bool]:
        """Get or create a participant record. Only inserts minimal info if not found.
        Returns: (uuid, was_created) tuple."""
        if self.dry_run:
            # In dry-run mode, simulate finding existing participant
            return str(uuid.uuid4()), True
            
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT uuid FROM participants WHERE name = %s",
                (name,)
            )
            row = cursor.fetchone()
            if row:
                return row[0], False

            participant_uuid = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO participants (uuid, name, type, sex, date_of_birth) VALUES (%s, %s, %s, %s, %s)",
                (participant_uuid, name, type_, sex, date_of_birth)
            )
            self.conn.commit()
            return participant_uuid, True
        finally:
            cursor.close()

    def get_or_create_session(
        self,
        name: str,
        participant_uuid: str,
        type_: str = None
    ) -> tuple[str, bool]:
        """Get or create a session record. Only inserts minimal info if not found.
        Returns: (uuid, was_created) tuple."""
        if self.dry_run:
            # In dry-run mode, simulate finding existing session
            return str(uuid.uuid4()), True
            
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT uuid FROM sessions WHERE name = %s AND participant_id = %s",
                (name, participant_uuid)
            )
            row = cursor.fetchone()
            if row:
                return row[0], False

            session_uuid = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO sessions (uuid, name, type, participant_id) VALUES (%s, %s, %s, %s)",
                (session_uuid, name, type_, participant_uuid)
            )
            self.conn.commit()
            return session_uuid, True
        finally:
            cursor.close()

    def add_qtm_record(
        self,
        session_uuid: str,
        file_path: str,
        trial: Optional[int] = None,
        repetition: Optional[int] = None,
        type_: Optional[str] = None,
        start_time: Optional[str] = None,
        valid: Optional[int] = 1
    ) -> tuple[str, bool]:
        """Add a QTM file record. Only inserts minimal info if not found.
        Returns: (uuid, was_created) tuple."""
        if self.dry_run:
            # In dry-run mode, simulate adding record
            return str(uuid.uuid4()), True
            
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT uuid FROM qtm_data WHERE session_id = %s AND file = %s",
                (session_uuid, file_path)
            )
            row = cursor.fetchone()
            if row:
                return row[0], False

            record_uuid = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO qtm_data (uuid, session_id, file, trial, repetition, type, start_time, valid) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (record_uuid, session_uuid, file_path, trial, repetition, type_, start_time, valid)
            )
            self.conn.commit()
            return record_uuid, True
        finally:
            cursor.close()