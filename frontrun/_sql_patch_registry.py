"""Registry of SQL patch targets used by the SQL cursor modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PythonCursorTarget:
    """Pure-Python DBAPI cursor target patched by direct method replacement."""

    module_path: str
    class_name: str
    paramstyle_module: str


@dataclass(frozen=True, slots=True)
class PythonConnectionTarget:
    """Pure-Python DBAPI connection class needing transaction wrappers."""

    module_path: str
    class_name: str


@dataclass(frozen=True, slots=True)
class ConnectFactoryTarget:
    """DBAPI driver patched by wrapping its ``connect`` function."""

    module_name: str
    cursor_module_name: str
    cursor_attr_name: str
    paramstyle: str
    driver: str


PYTHON_CURSOR_TARGETS: tuple[PythonCursorTarget, ...] = (PythonCursorTarget("pymysql.cursors", "Cursor", "pymysql"),)

PYTHON_CONNECTION_TARGETS: tuple[PythonConnectionTarget, ...] = (
    PythonConnectionTarget("pymysql.connections", "Connection"),
)

CONNECT_FACTORY_TARGETS: tuple[ConnectFactoryTarget, ...] = (
    ConnectFactoryTarget("psycopg2", "psycopg2.extensions", "cursor", "pyformat", "psycopg2"),
    ConnectFactoryTarget("psycopg", "psycopg", "Cursor", "format", "psycopg"),
)
