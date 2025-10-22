"""
Project structure validator and handler.
"""
import os
import re
import logging
from typing import Dict, List, Optional, Union, Any

from .settings_parser import ProjectSettings


class ProjectStructure:
    def __init__(self, settings: ProjectSettings, base_dir: str):
        self.settings = settings
        self.base_dir = base_dir
        # Compile regex patterns for directory matching
        self.patterns = {
            name: self._compile_pattern(pattern)
            for name, pattern in settings.directory_patterns.items()
        }
        # Log available session types in a structured way
        session_types_str = "\n".join(
            f"- {stype}: measurements = [{', '.join(stype_obj.measurements)}]"
            for stype, stype_obj in self.settings.session_types.items()
        )
        logging.info("Available session types:\n%s", session_types_str)

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Convert a directory pattern into a regex pattern."""
        # Replace $ID$ with a capture group for any alphanumeric sequence
        pattern = pattern.replace("$ID$", "([A-Za-z0-9]+)")
        return re.compile(f"^{pattern}$")

    def validate_directory(self, dir_path: str) -> Dict[str, str]:
        """
        Validate a directory against the project structure rules.
        Returns a dict with extracted metadata (e.g., {'participant_id': 'P001'})
        """
        dir_name = os.path.basename(dir_path)
        parent_dir = os.path.dirname(dir_path)
        metadata = {}

        # First check if this is a session directory by checking the parent and dir name
        # Use the configurable participant pattern from settings instead of hardcoded pattern
        parent_dir_name = os.path.basename(parent_dir)
        is_participant_dir = False
        for entity_type, pattern in self.patterns.items():
            if entity_type == "Participant" and pattern.match(parent_dir_name):
                is_participant_dir = True
                break
        
        if is_participant_dir:
            # Handle session directories with optional number suffix
            dir_parts = dir_name.rsplit('_', 1)
            base_name = dir_parts[0]
            number = int(dir_parts[1]) if len(dir_parts) > 1 and dir_parts[1].isdigit() else None
            if base_name in self.settings.session_types:
                metadata["session_type"] = base_name
                metadata["session_name"] = dir_name  # full session directory name
                if number is not None:
                    metadata["session_number"] = number
                # Extract participant ID using the pattern match
                for entity_type, pattern in self.patterns.items():
                    if entity_type == "Participant":
                        match = pattern.match(parent_dir_name)
                        if match:
                            metadata["participant_id"] = match.group(1)  # Extract the ID from the pattern
                            break
                metadata["participant_name"] = parent_dir_name    # full dir name
                logging.info(f"✓ Valid session directory: {dir_name} (participant: {metadata['participant_name']})")
                return metadata

        # If not a session dir, try to match participant pattern
        for entity_type, pattern in self.patterns.items():
            match = pattern.match(dir_name)
            if match and entity_type == "Participant":
                metadata["participant_id"] = match.group(1)
                metadata["participant_name"] = dir_name  # full dir name
                break

        return metadata

    def scan_for_qtm_files(self) -> List[Dict[str, str]]:
        """
        Scan the project directory for QTM files and collect metadata.
        Returns a list of dicts with file info and metadata.
        """
        qtm_files = []
        logging.info("Scanning base directory: %s", os.path.abspath(self.base_dir))
        logging.info("Directory patterns to match: %s", {k: v.pattern for k, v in self.patterns.items()})

        # First level: participant directories
        for participant_dir in sorted(os.listdir(self.base_dir)):
            participant_path = os.path.join(self.base_dir, participant_dir)
            if not os.path.isdir(participant_path):
                continue
            participant_metadata = self.validate_directory(participant_path)
            if "participant_name" not in participant_metadata:
                continue
            logging.info("----------------------------------------")
            logging.info("Checking participant: %s", participant_dir)
            logging.info("✓ Valid participant directory: %s", participant_dir)

            # Second level: session directories inside participant
            for session_dir in sorted(os.listdir(participant_path)):
                session_path = os.path.join(participant_path, session_dir)
                if not os.path.isdir(session_path):
                    continue
                session_metadata = self.validate_directory(session_path)
                if "session_type" not in session_metadata:
                    continue
                logging.info("Checking session: %s", session_dir)
                logging.info("✓ Valid session directory: %s (type: %s)", session_dir, session_metadata["session_type"])

                # Find QTM files in session directory
                files = os.listdir(session_path)
                logging.info("Checking %d files: %s", len(files), files)
                for file in sorted(files):
                    if not file.lower().endswith(".qtm"):
                        continue
                    full_path = os.path.abspath(os.path.join(session_path, file))
                    logging.info("✓ Found QTM file: %s", full_path)
                    measurement_info = self._identify_measurement(file)
                    if not measurement_info:
                        logging.warning("❌ Unrecognized measurement format: %s", file)
                        continue
                    file_metadata = {
                        "file_path": os.path.join(participant_dir, session_dir, file),
                        "type": measurement_info["type"],  # measurement type
                        "repetition": measurement_info["repetition"],
                        "participant_name": participant_metadata.get("participant_name"),
                        "session_name": session_metadata.get("session_name"),
                        "session_type": session_metadata.get("session_type"),
                        "session_number": session_metadata.get("session_number"),
                        "participant_id": participant_metadata.get("participant_id"),
                    }
                    logging.debug("Found QTM file: %s", file_metadata)
                    qtm_files.append(file_metadata)

        logging.info("Found %d QTM files", len(qtm_files))
        return qtm_files

    def _identify_measurement(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Identify the measurement type and repetition from the filename.
        Returns dict with measurement info if found, None otherwise.
        """
        # Match "[Measurement] X.qtm" pattern
        match = re.match(r"^(.+?)\s+(\d+)\.qtm$", filename, re.IGNORECASE)
        if not match:
            return None

        measurement_name = match.group(1).strip()
        repetition = int(match.group(2))

        # Check if this is a known measurement type
        for meas_type, meas_info in self.settings.measurements.items():
            if measurement_name.lower() == meas_type.lower():
                return {
                    "type": meas_type,
                    "repetition": repetition
                }

        return None