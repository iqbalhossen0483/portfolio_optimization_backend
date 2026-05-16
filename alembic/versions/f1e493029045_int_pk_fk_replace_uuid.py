"""int pk fk replace uuid

Revision ID: f1e493029045
Revises: 31d734722d27
Create Date: 2026-05-16 14:37:37.777270

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1e493029045'
down_revision: Union[str, Sequence[str], None] = '31d734722d27'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop all tables in reverse-dependency order (children first)
    op.execute("DROP TABLE IF EXISTS training_normalizer_params CASCADE")
    op.execute("DROP TABLE IF EXISTS model_checkpoints CASCADE")
    op.execute("DROP TABLE IF EXISTS portfolios CASCADE")
    op.execute("DROP TABLE IF EXISTS esg_scores CASCADE")
    op.execute("DROP TABLE IF EXISTS market_data CASCADE")
    op.execute("DROP TABLE IF EXISTS training_jobs CASCADE")
    op.execute("DROP TABLE IF EXISTS assets CASCADE")

    # Recreate assets
    op.create_table(
        'assets',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('isin', sa.String(12), nullable=False, unique=True),
        sa.Column('name', sa.String(256), nullable=False),
        sa.Column('sector', sa.String(128), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_assets_isin', 'assets', ['isin'])

    # Recreate training_jobs
    op.create_table(
        'training_jobs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('status', sa.String(32), nullable=False),
        sa.Column('portfolio_model', sa.String(8), nullable=False),
        sa.Column('topology', sa.String(32), nullable=False),
        sa.Column('config_json', sa.JSON(), nullable=False),
        sa.Column('current_step', sa.Integer(), nullable=True, default=0),
        sa.Column('best_sharpe', sa.Float(), nullable=True),
        sa.Column('best_mu_esg', sa.Float(), nullable=True),
        sa.Column('error_message', sa.String(1024), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Recreate market_data
    op.create_table(
        'market_data',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('asset_id', sa.Integer(), sa.ForeignKey('assets.id'), nullable=False),
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('open', sa.Float(), nullable=True),
        sa.Column('high', sa.Float(), nullable=True),
        sa.Column('low', sa.Float(), nullable=True),
        sa.Column('close', sa.Float(), nullable=True),
        sa.Column('volume', sa.Float(), nullable=True),
        sa.Column('return_pct', sa.Float(), nullable=True),
        sa.Column('rsi', sa.Float(), nullable=True),
        sa.Column('macd_hist', sa.Float(), nullable=True),
        sa.UniqueConstraint('asset_id', 'date'),
    )
    op.create_index('ix_market_data_asset_id', 'market_data', ['asset_id'])
    op.create_index('ix_market_data_date', 'market_data', ['date'])

    # Recreate esg_scores
    op.create_table(
        'esg_scores',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('asset_id', sa.Integer(), sa.ForeignKey('assets.id'), nullable=False),
        sa.Column('date', sa.DateTime(), nullable=False),
        sa.Column('bloomberg_score', sa.Float(), nullable=True),
        sa.Column('lesg_score', sa.Float(), nullable=True),
        sa.Column('esg_b_norm', sa.Float(), nullable=True),
        sa.Column('esg_l_norm', sa.Float(), nullable=True),
        sa.Column('delta_esg', sa.Float(), nullable=True),
        sa.Column('mu_esg', sa.Float(), nullable=True),
        sa.UniqueConstraint('asset_id', 'date'),
    )
    op.create_index('ix_esg_scores_asset_id', 'esg_scores', ['asset_id'])
    op.create_index('ix_esg_scores_date', 'esg_scores', ['date'])

    # Recreate model_checkpoints
    op.create_table(
        'model_checkpoints',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('job_id', sa.Integer(), sa.ForeignKey('training_jobs.id'), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('path', sa.String(512), nullable=False),
        sa.Column('sharpe', sa.Float(), nullable=True),
        sa.Column('mu_esg', sa.Float(), nullable=True),
        sa.Column('entropy', sa.Float(), nullable=True),
        sa.Column('saved_at', sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index('ix_model_checkpoints_job_id', 'model_checkpoints', ['job_id'])

    # Recreate portfolios
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

    # Recreate training_normalizer_params
    op.create_table(
        'training_normalizer_params',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('job_id', sa.Integer(), sa.ForeignKey('training_jobs.id'), nullable=False),
        sa.Column('isin', sa.String(12), nullable=False),
        sa.Column('feature_name', sa.String(32), nullable=False),
        sa.Column('min_val', sa.Float(), nullable=False),
        sa.Column('max_val', sa.Float(), nullable=False),
        sa.UniqueConstraint('job_id', 'isin', 'feature_name'),
    )
    op.create_index('ix_training_normalizer_params_job_id', 'training_normalizer_params', ['job_id'])
    op.create_index('ix_training_normalizer_params_isin', 'training_normalizer_params', ['isin'])


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS training_normalizer_params CASCADE")
    op.execute("DROP TABLE IF EXISTS model_checkpoints CASCADE")
    op.execute("DROP TABLE IF EXISTS portfolios CASCADE")
    op.execute("DROP TABLE IF EXISTS esg_scores CASCADE")
    op.execute("DROP TABLE IF EXISTS market_data CASCADE")
    op.execute("DROP TABLE IF EXISTS training_jobs CASCADE")
    op.execute("DROP TABLE IF EXISTS assets CASCADE")
