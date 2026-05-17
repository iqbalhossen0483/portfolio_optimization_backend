"""drop portfolios table

Revision ID: a3c9e1d82f01
Revises: f1e493029045
Create Date: 2026-05-17

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'a3c9e1d82f01'
down_revision: Union[str, Sequence[str], None] = 'f1e493029045'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS portfolios CASCADE")


def downgrade() -> None:
    op.create_table(
        'portfolios',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('query_id', sa.String(36), nullable=False),
        sa.Column('job_id', sa.Integer(), sa.ForeignKey('training_jobs.id'), nullable=True),
        sa.Column('topology', sa.String(32), nullable=False),
        sa.Column('portfolio_model', sa.String(8), nullable=False),
        sa.Column('allocation_json', sa.JSON(), nullable=False),
        sa.Column('metrics_json', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_portfolios_query_id', 'portfolios', ['query_id'])
