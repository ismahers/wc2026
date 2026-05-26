"""
Data contracts for the WC 2026 betting pipeline.

The schema registry defines the normalized tables that feed feature
engineering, market models, odds comparison, and betting evaluation.
"""

from .schemas import (
    ColumnSpec,
    TableSchema,
    TABLE_SCHEMAS,
    create_empty_table,
    get_schema,
    table_columns,
    validate_columns,
)
from .sources import DataSource, SOURCE_CATALOG, get_source

__all__ = [
    "ColumnSpec",
    "DataSource",
    "TableSchema",
    "SOURCE_CATALOG",
    "TABLE_SCHEMAS",
    "create_empty_table",
    "get_schema",
    "get_source",
    "table_columns",
    "validate_columns",
]
