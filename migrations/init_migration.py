"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # 这里会自动生成创建表的SQL语句
    pass

def downgrade():
    # 这里会自动生成删除表的SQL语句
    pass
