"""Convert all DateTime columns to TIMESTAMPTZ

Revision ID: b7f2a3c91e05
Revises: a3c9e1d82f01
Create Date: 2026-05-17
"""
from __future__ import annotations
from typing import Sequence, Union
from alembic import op

revision: str = 'b7f2a3c91e05'
down_revision: Union[str, Sequence[str], None] = 'a3c9e1d82f01'
branch_labels = None
depends_on = None

_COLUMNS = [
    ("assets",           "created_at"),
    ("market_data",      "date"),
    ("esg_scores",       "date"),
    ("training_jobs",    "started_at"),
    ("training_jobs",    "completed_at"),
    ("training_jobs",    "created_at"),
    ("model_checkpoints","saved_at"),
]


def upgrade() -> None:
    for table, col in _COLUMNS:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {col} "
            f"TYPE TIMESTAMPTZ USING {col} AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    for table, col in _COLUMNS:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {col} "
            f"TYPE TIMESTAMP WITHOUT TIME ZONE USING {col} AT TIME ZONE 'UTC'"
        )
