from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
from sqlalchemy.dialects.mysql import LONGTEXT

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """用户表"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment='用户ID')
    username = db.Column(db.String(80), unique=True, nullable=False, comment='用户名')
    email = db.Column(db.String(255), unique=True, nullable=False, comment='邮箱')
    password_hash = db.Column(db.String(255), nullable=False, comment='密码哈希')
    
    # 用户状态
    is_active = db.Column(db.Boolean, default=True, comment='是否激活')
    is_admin = db.Column(db.Boolean, default=False, comment='是否管理员')
    show_visualization = db.Column(db.Boolean, default=True, nullable=False, comment='是否展示可视化界面')
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow, comment='创建时间')
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment='更新时间')
    last_login = db.Column(db.DateTime, comment='最后登录时间')
    
    # 用户统计信息
    total_conversations = db.Column(db.Integer, default=0, comment='总对话数')
    total_tokens_used = db.Column(db.BigInteger, default=0, comment='总使用Token数')
    
    # 关联到对话记录
    conversations = db.relationship('Conversation', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """设置密码"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)
    
    def update_login_time(self):
        """更新最后登录时间"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'show_visualization': self.show_visualization,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'total_conversations': self.total_conversations,
            'total_tokens_used': self.total_tokens_used
        }

class Conversation(db.Model):
    """对话记录表"""
    __tablename__ = 'conversations'
    
    # 基本信息
    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment='对话ID')
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, comment='用户ID')
    
    # 对话内容
    question = db.Column(db.Text, nullable=False, comment='用户问题')
    answer = db.Column(LONGTEXT, nullable=True, comment='AI回答')
    raw_output = db.Column(LONGTEXT, nullable=True, comment='原始输出')
    
    # API和模型信息
    api_provider = db.Column(db.String(50), nullable=True, comment='API提供商')
    model = db.Column(db.String(100), nullable=True, comment='使用的模型')
    reasoning_method = db.Column(db.String(100), nullable=True, comment='推理方法')
    
    # 配置参数
    max_tokens = db.Column(db.Integer, nullable=True, comment='最大Token数')
    chars_per_line = db.Column(db.Integer, nullable=True, comment='每行字符数')
    max_lines = db.Column(db.Integer, nullable=True, comment='最大行数')
    
    # 实际使用的Token数
    input_tokens = db.Column(db.Integer, nullable=True, comment='输入Token数')
    output_tokens = db.Column(db.Integer, nullable=True, comment='输出Token数')
    total_tokens = db.Column(db.Integer, nullable=True, comment='总Token数')
    
    # 可视化数据
    visualization_data = db.Column(LONGTEXT, nullable=True, comment='可视化数据(JSON)')
    mermaid_code = db.Column(LONGTEXT, nullable=True, comment='Mermaid图表代码')
    
    # 状态和错误信息
    status = db.Column(db.String(20), default='pending', comment='状态: pending/completed/failed')
    error_message = db.Column(db.Text, nullable=True, comment='错误信息')
    
    # 性能指标
    processing_time = db.Column(db.Float, nullable=True, comment='处理时间(秒)')
    response_quality_score = db.Column(db.Float, nullable=True, comment='回答质量评分')
    
    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow, comment='创建时间')
    started_at = db.Column(db.DateTime, nullable=True, comment='开始处理时间')
    completed_at = db.Column(db.DateTime, nullable=True, comment='完成时间')
    
    # 额外元数据
    client_ip = db.Column(db.String(45), nullable=True, comment='客户端IP')
    user_agent = db.Column(db.String(500), nullable=True, comment='用户代理')
    session_id = db.Column(db.String(100), nullable=True, comment='会话ID')
    
    def set_visualization_data(self, data):
        """设置可视化数据"""
        if data:
            self.visualization_data = json.dumps(data, ensure_ascii=False, indent=2)
    
    def get_visualization_data(self):
        """获取可视化数据"""
        if self.visualization_data:
            try:
                return json.loads(self.visualization_data)
            except json.JSONDecodeError:
                return None
        return None
    
    def calculate_processing_time(self):
        """计算处理时间"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.processing_time = delta.total_seconds()
            return self.processing_time
        return None
    
    def mark_as_started(self):
        """标记为开始处理"""
        self.started_at = datetime.utcnow()
        self.status = 'processing'
    
    def mark_as_completed(self, answer, raw_output=None, visualization_data=None, token_usage=None):
        """标记为完成"""
        self.answer = answer
        self.raw_output = raw_output or answer
        self.completed_at = datetime.utcnow()
        self.status = 'completed'
        
        if visualization_data:
            self.set_visualization_data(visualization_data)
        
        if token_usage:
            self.input_tokens = token_usage.get('input_tokens')
            self.output_tokens = token_usage.get('output_tokens')
            self.total_tokens = token_usage.get('total_tokens')
        
        self.calculate_processing_time()
    
    def mark_as_failed(self, error_message):
        """标记为失败"""
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        self.status = 'failed'
        self.calculate_processing_time()
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username if self.user else None,
            'question': self.question,
            'answer': self.answer,
            'raw_output': self.raw_output,
            'api_provider': self.api_provider,
            'model': self.model,
            'reasoning_method': self.reasoning_method,
            'max_tokens': self.max_tokens,
            'chars_per_line': self.chars_per_line,
            'max_lines': self.max_lines,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'visualization_data': self.get_visualization_data(),
            'mermaid_code': self.mermaid_code,
            'status': self.status,
            'error_message': self.error_message,
            'processing_time': self.processing_time,
            'response_quality_score': self.response_quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'client_ip': self.client_ip,
            'user_agent': self.user_agent,
            'session_id': self.session_id
        }

class ConversationTag(db.Model):
    """对话标签表"""
    __tablename__ = 'conversation_tags'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    tag_name = db.Column(db.String(50), nullable=False, comment='标签名称')
    tag_value = db.Column(db.String(200), nullable=True, comment='标签值')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    conversation = db.relationship('Conversation', backref='tags')

class UserSession(db.Model):
    """用户会话表"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    
    user = db.relationship('User', backref='sessions')

# 创建索引
def create_indexes():
    """创建数据库索引以优化查询性能"""

    engine = db.engine
    dialect_name = engine.dialect.name.lower()

    def _index_exists(conn, table_name: str, index_name: str) -> bool:
        """检查索引是否已存在，避免在不支持 IF NOT EXISTS 的 MySQL 上报错"""
        if dialect_name != 'mysql':
            return False

        query = db.text(
            """
            SELECT COUNT(1)
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = :table_name
              AND index_name = :index_name
            """
        )
        result = conn.execute(query, {"table_name": table_name, "index_name": index_name}).scalar()
        return bool(result)

    def _ensure_index(conn, table_name: str, index_name: str, column_expr: str) -> None:
        if dialect_name == 'mysql':
            if _index_exists(conn, table_name, index_name):
                return
            conn.execute(db.text(f"CREATE INDEX {index_name} ON {table_name} ({column_expr})"))
        else:
            conn.execute(db.text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_expr})"))

    with engine.connect() as conn:
        # 对话表索引
        _ensure_index(conn, "conversations", "idx_conversations_user_id", "user_id")
        _ensure_index(conn, "conversations", "idx_conversations_created_at", "created_at")
        _ensure_index(conn, "conversations", "idx_conversations_status", "status")
        _ensure_index(conn, "conversations", "idx_conversations_api_provider", "api_provider")
        _ensure_index(conn, "conversations", "idx_conversations_model", "model")
        _ensure_index(conn, "conversations", "idx_conversations_reasoning_method", "reasoning_method")

        # 用户表索引
        _ensure_index(conn, "users", "idx_users_username", "username")
        _ensure_index(conn, "users", "idx_users_email", "email")
        _ensure_index(conn, "users", "idx_users_created_at", "created_at")

        # 标签表索引
        _ensure_index(conn, "conversation_tags", "idx_tags_conversation_id", "conversation_id")
        _ensure_index(conn, "conversation_tags", "idx_tags_name", "tag_name")

        # 会话表索引
        _ensure_index(conn, "user_sessions", "idx_sessions_user_id", "user_id")
        _ensure_index(conn, "user_sessions", "idx_sessions_token", "session_token")
        _ensure_index(conn, "user_sessions", "idx_sessions_expires", "expires_at")
