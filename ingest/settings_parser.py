"""
Parse project settings files (Settings.paf) into a structured format.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class Field:
    name: str
    type: str
    inherit: Optional[str] = None
    unit: Optional[str] = None
    values: Optional[List[str]] = None  # For enum types
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    readonly: bool = False
    hidden: bool = False


@dataclass
class Measurement:
    name: str
    length: Optional[float] = None
    aim_models: Optional[List[str]] = None
    measurement_type: str = "Dynamic"
    min_count: int = 1
    max_count: int = 1
    count: int = 1
    fields: Optional[List[str]] = None


@dataclass
class SessionType:
    name: str
    fields: List[str]
    measurements: List[str]
    analyses: Optional[List[str]] = None


@dataclass
class ProjectSettings:
    project_id: str
    fields: Dict[str, Field]
    measurements: Dict[str, Measurement]
    session_types: Dict[str, SessionType]
    directory_patterns: Dict[str, str]


def parse_field(name: str, config: Dict[str, Any]) -> Field:
    """Parse a field definition from the settings file."""
    return Field(
        name=name,
        type=config["Type"],
        inherit=config.get("Inherit"),
        unit=config.get("Unit"),
        values=config.get("Values"),
        min_value=config.get("Min"),
        max_value=config.get("Max"),
        readonly=config.get("Readonly", False),
        hidden=config.get("Hidden", False)
    )


def parse_measurement(name: str, config: Dict[str, Any]) -> Measurement:
    """Parse a measurement definition from the settings file."""
    return Measurement(
        name=name,
        length=config.get("Measurement length"),
        aim_models=config.get("AIM models"),
        measurement_type=config.get("Measurement type", "Dynamic"),
        min_count=config.get("Minimum count", 1),
        max_count=config.get("Maximum count", 1),
        count=config.get("Count", 1),
        fields=config.get("Fields", [])
    )


def parse_session_type(name: str, config: Dict[str, Any]) -> SessionType:
    """Parse a session type definition from the settings file."""
    return SessionType(
        name=name,
        fields=config.get("Fields", []),
        measurements=config.get("Measurements", []),
        analyses=config.get("Analyses")
    )


def parse_settings_file(file_path: str) -> ProjectSettings:
    """Parse a Settings.paf file into a ProjectSettings object."""
    with open(file_path, 'r') as f:
        # The file isn't strictly YAML, but we can parse it as such
        data = yaml.safe_load(f)

    # Extract directory patterns
    directory_patterns = {}
    for type_name, type_config in data.get("Types", {}).items():
        for subtype_name, subtype_config in type_config.items():
            if "Directory pattern" in subtype_config:
                directory_patterns[subtype_name] = subtype_config["Directory pattern"]

    # Parse fields
    fields = {
        name: parse_field(name, config)
        for name, config in data.get("Fields", {}).items()
    }

    # Parse measurements
    measurements = {
        name: parse_measurement(name, config)
        for name, config in data.get("Measurements", {}).items()
        if name != "Fields"  # Skip the generic Fields section
    }

    # Parse session types
    session_types = {}
    for type_name, type_config in data.get("Types", {}).items():
        if type_name == "Session":
            for session_name, session_config in type_config.items():
                session_types[session_name] = parse_session_type(
                    session_name, session_config
                )

    return ProjectSettings(
        project_id=data.get("Project ID", ""),
        fields=fields,
        measurements=measurements,
        session_types=session_types,
        directory_patterns=directory_patterns
    )