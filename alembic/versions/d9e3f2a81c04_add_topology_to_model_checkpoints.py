"""Add topology column to model_checkpoints

Revision ID: d9e3f2a81c04
Revises: c4d8b1e72f39
Create Date: 2026-05-17
"""
from __future__ import annotations
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'd9e3f2a81c04'
down_revision: Union[str, Sequence[str], None] = 'c4d8b1e72f39'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_checkpoints",
        sa.Column("topology", sa.String(32), nullable=False, server_default="cooperative"),
    )
    op.create_index("ix_model_checkpoints_topology", "model_checkpoints", ["topology"])


def downgrade() -> None:
    op.drop_index("ix_model_checkpoints_topology", table_name="model_checkpoints")
    op.drop_column("model_checkpoints", "topology")
