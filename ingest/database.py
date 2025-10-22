"""
Database schema and operations for the enhanced QTM data ingestion.
"""
import uuid
from typing import Dict, Optional

import mysql.connector
from mysql.connector.cursor import MySQLCursor

from .settings_parser import ProjectSettings


class Database:
    def __init__(self, connection: mysql.connector.MySQLConnection, settings: ProjectSettings):
        self.conn = connection
        self.settings = settings

    def get_or_create_participant(self, name: str, type_: str = None, sex: str = None, date_of_birth: str = None) -> tuple[str, bool]:
        """Get or create a participant record. Only inserts minimal info if not found.
        Returns: (uuid, was_created) tuple."""
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