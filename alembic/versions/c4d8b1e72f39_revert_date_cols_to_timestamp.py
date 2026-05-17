"""Revert market_data.date and esg_scores.date back to TIMESTAMP (no tz)

These columns hold trading dates from XLSX — always timezone-naive pandas Timestamps.

Revision ID: c4d8b1e72f39
Revises: b7f2a3c91e05
Create Date: 2026-05-17
"""
from __future__ import annotations
from typing import Sequence, Union
from alembic import op

revision: str = 'c4d8b1e72f39'
down_revision: Union[str, Sequence[str], None] = 'b7f2a3c91e05'
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("market_data", "esg_scores"):
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN date "
            f"TYPE TIMESTAMP WITHOUT TIME ZONE USING date AT TIME ZONE 'UTC'"
        )


def downgrade() -> None:
    for table in ("market_data", "esg_scores"):
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN date "
            f"TYPE TIMESTAMPTZ USING date AT TIME ZONE 'UTC'"
        )
