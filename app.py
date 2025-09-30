import logging
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from datetime import datetime
from dotenv import load_dotenv
import json

from database_config import DatabaseConfig
from models import db, User, Conversation, ConversationTag, UserSession, create_indexes
import config

# 加载环境变量
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# 使用数据库配置
app.config.update(DatabaseConfig.get_config_dict())
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录后再使用推理功能'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility functions
def get_client_info():
    """获取客户端信息"""
    return {
        'ip': request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR')),
        'user_agent': request.headers.get('User-Agent', '')
    }

def update_user_stats(user, tokens_used=0):
    """更新用户统计信息"""
    user.total_conversations += 1
    user.total_tokens_used += tokens_used
    db.session.commit()

# Routes
@app.route('/')
def index():
    """Render the main page"""
    if current_user.is_authenticated:
        return render_template('index.html', user=current_user)
    return redirect(url_for('login'))

@app.route('/index.html')
def index_direct():
    """Directly render the main page when accessed via index.html"""
    if current_user.is_authenticated:
        return render_template('index.html', user=current_user)
    return redirect(url_for('login'))

@app.route('/index_cn.html')
def index_cn():
    """Render the Chinese version of the main page"""
    if current_user.is_authenticated:
        return render_template('index_cn.html', user=current_user)
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """用户登录"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'})
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            user.update_login_time()
            return jsonify({'success': True, 'message': '登录成功'})
        else:
            return jsonify({'success': False, 'message': '用户名或密码错误'})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """用户注册"""
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'success': False, 'message': '所有字段都是必填的'})
        
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': '用户名已存在'})
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': '邮箱已被注册'})
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            return jsonify({'success': True, 'message': '注册成功'})
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {str(e)}")
            return jsonify({'success': False, 'message': '注册失败，请重试'})
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """用户登出"""
    logout_user()
    return redirect(url_for('login'))

@app.route('/config')
@login_required
def get_config():
    """Get initial configuration"""
    return jsonify(config.get_initial_values())

@app.route('/method-config/<method_id>')
@login_required
def get_method_config(method_id):
    """Get configuration for specific method"""
    method_config = config.get_method_config(method_id)
    if method_config:
        return jsonify(method_config)
    return jsonify({"error": "Method not found"}), 404

@app.route('/process', methods=['POST'])
@login_required
def process():
    """Process the reasoning request"""
    conversation = None
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Extract parameters
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'API key is required'
            }), 400

        question = data.get('question')
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400

        # 获取客户端信息
        client_info = get_client_info()

        # 创建对话记录
        conversation = Conversation(
            user_id=current_user.id,
            question=question,
            api_provider=data.get('provider', ''),
            model=data.get('model', ''),
            reasoning_method=data.get('reasoning_method', ''),
            max_tokens=data.get('max_tokens', 0),
            chars_per_line=data.get('chars_per_line', 50),
            max_lines=data.get('max_lines', 10),
            client_ip=client_info['ip'],
            user_agent=client_info['user_agent'],
            status='pending'
        )
        
        db.session.add(conversation)
        db.session.commit()

        # 标记开始处理
        conversation.mark_as_started()
        db.session.commit()

        # 执行推理处理
        try:
            # 这里是模拟推理过程，实际使用时需要替换为真实的推理逻辑
            import time
            time.sleep(1)  # 模拟处理时间
            
            raw_output = f"这是对问题 '{question}' 的推理回答。使用的模型是 {data.get('model')}，推理方法是 {data.get('reasoning_method')}。"
            
            # 模拟可视化数据
            visualization_data = {
                "type": "mermaid",
                "code": "flowchart TD\n    A[问题] --> B[分析]\n    B --> C[推理]\n    C --> D[结论]",
                "chars_per_line": data.get('chars_per_line', 50),
                "max_lines": data.get('max_lines', 10)
            }
            
            # 模拟Token使用情况
            token_usage = {
                'input_tokens': len(question) // 4,  # 估算输入token
                'output_tokens': len(raw_output) // 4,  # 估算输出token
                'total_tokens': (len(question) + len(raw_output)) // 4
            }
            
            # 标记完成
            conversation.mark_as_completed(
                answer=raw_output,
                raw_output=raw_output,
                visualization_data=visualization_data,
                token_usage=token_usage
            )
            
            # 更新用户统计
            update_user_stats(current_user, token_usage['total_tokens'])
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'conversation_id': conversation.id,
                'raw_output': raw_output,
                'visualization': visualization_data,
                'token_usage': token_usage,
                'processing_time': conversation.processing_time
            })

        except Exception as processing_error:
            logger.error(f"Error during reasoning processing: {str(processing_error)}")
            conversation.mark_as_failed(str(processing_error))
            db.session.commit()
            return jsonify({
                'success': False,
                'error': f'处理推理请求时发生错误: {str(processing_error)}'
            }), 500

    except Exception as e:
        logger.error(f"Error in process endpoint: {str(e)}")
        if conversation:
            conversation.mark_as_failed(str(e))
            db.session.commit()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/select-method', methods=['POST'])
@login_required
def select_method():
    """Select reasoning method"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({
                'success': False,
                'error': 'Question is required for method selection'
            }), 400
        
        # 智能方法选择逻辑
        selected_method = {
            'method_id': 'cot',
            'name': 'Chain of Thought',
            'description': '逐步推理方法'
        }
        
        return jsonify({
            'success': True,
            'selected_method': selected_method
        })
        
    except Exception as e:
        logger.error(f"Error in select-method: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/long-reasoning', methods=['POST'])
@login_required
def long_reasoning():
    """Handle long reasoning requests"""
    conversation = None
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        question = data.get('question')
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400

        # 获取客户端信息
        client_info = get_client_info()

        # 创建对话记录
        conversation = Conversation(
            user_id=current_user.id,
            question=question,
            api_provider=data.get('provider', ''),
            model=data.get('model', ''),
            reasoning_method='long_reasoning',
            max_tokens=data.get('max_tokens', 0),
            chars_per_line=data.get('chars_per_line', 50),
            max_lines=data.get('max_lines', 10),
            client_ip=client_info['ip'],
            user_agent=client_info['user_agent'],
            status='pending'
        )
        
        db.session.add(conversation)
        db.session.commit()

        # 标记开始处理
        conversation.mark_as_started()
        db.session.commit()

        # 执行长推理处理
        try:
            import time
            time.sleep(2)  # 模拟更长的处理时间
            
            raw_output = f"这是对问题 '{question}' 的长推理回答。这个回答包含了更详细的分析过程和多步骤推理。"
            
            # 模拟Token使用情况
            token_usage = {
                'input_tokens': len(question) // 4,
                'output_tokens': len(raw_output) // 4,
                'total_tokens': (len(question) + len(raw_output)) // 4
            }
            
            # 标记完成
            conversation.mark_as_completed(
                answer=raw_output,
                raw_output=raw_output,
                token_usage=token_usage
            )
            
            # 更新用户统计
            update_user_stats(current_user, token_usage['total_tokens'])
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'conversation_id': conversation.id,
                'raw_output': raw_output,
                'token_usage': token_usage,
                'processing_time': conversation.processing_time
            })

        except Exception as processing_error:
            logger.error(f"Error during long reasoning: {str(processing_error)}")
            conversation.mark_as_failed(str(processing_error))
            db.session.commit()
            return jsonify({
                'success': False,
                'error': f'长推理处理失败: {str(processing_error)}'
            }), 500

    except Exception as e:
        logger.error(f"Error in long-reasoning endpoint: {str(e)}")
        if conversation:
            conversation.mark_as_failed(str(e))
            db.session.commit()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/history')
@login_required
def get_history():
    """获取用户的历史对话记录"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        conversations = Conversation.query.filter_by(user_id=current_user.id)\
                                       .order_by(Conversation.created_at.desc())\
                                       .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'conversations': [conv.to_dict() for conv in conversations.items],
            'total': conversations.total,
            'pages': conversations.pages,
            'current_page': page
        })
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/history/<int:conversation_id>')
@login_required
def get_conversation(conversation_id):
    """获取特定对话的详细信息"""
    try:
        conversation = Conversation.query.filter_by(id=conversation_id, user_conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first())
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        return jsonify(conversation.to_dict())
    except Exception as e:
        logger.error(f"Error getting conversation {conversation_id}: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/profile')
@login_required
def profile():
    """用户个人资料页面"""
    # 获取用户统计信息
    user_stats = {
        'total_conversations': current_user.total_conversations,
        'total_tokens_used': current_user.total_tokens_used,
        'member_since': current_user.created_at.strftime('%Y年%m月%d日') if current_user.created_at else None,
        'last_login': current_user.last_login.strftime('%Y年%m月%d日 %H:%M') if current_user.last_login else None
    }
    
    # 获取最近的对话
    recent_conversations = Conversation.query.filter_by(user_id=current_user.id)\
                                           .order_by(Conversation.created_at.desc())\
                                           .limit(5).all()
    
    return render_template('profile.html', user=current_user, stats=user_stats, recent_conversations=recent_conversations)

@app.route('/stats')
@login_required
def get_user_stats():
    """获取用户统计信息"""
    try:
        # 基本统计
        total_conversations = Conversation.query.filter_by(user_id=current_user.id).count()
        completed_conversations = Conversation.query.filter_by(user_id=current_user.id, status='completed').count()
        failed_conversations = Conversation.query.filter_by(user_id=current_user.id, status='failed').count()
        
        # 使用的模型统计
        model_stats = db.session.query(
            Conversation.model,
            db.func.count(Conversation.id).label('count')
        ).filter_by(user_id=current_user.id).group_by(Conversation.model).all()
        
        # API提供商统计
        provider_stats = db.session.query(
            Conversation.api_provider,
            db.func.count(Conversation.id).label('count')
        ).filter_by(user_id=current_user.id).group_by(Conversation.api_provider).all()
        
        # Token使用统计
        token_stats = db.session.query(
            db.func.sum(Conversation.total_tokens).label('total_tokens'),
            db.func.avg(Conversation.total_tokens).label('avg_tokens'),
            db.func.max(Conversation.total_tokens).label('max_tokens')
        ).filter_by(user_id=current_user.id).first()
        
        # 处理时间统计
        time_stats = db.session.query(
            db.func.avg(Conversation.processing_time).label('avg_time'),
            db.func.max(Conversation.processing_time).label('max_time'),
            db.func.min(Conversation.processing_time).label('min_time')
        ).filter_by(user_id=current_user.id, status='completed').first()
        
        return jsonify({
            'basic_stats': {
                'total_conversations': total_conversations,
                'completed_conversations': completed_conversations,
                'failed_conversations': failed_conversations,
                'success_rate': round((completed_conversations / total_conversations * 100), 2) if total_conversations > 0 else 0
            },
            'model_stats': [{'model': stat.model, 'count': stat.count} for stat in model_stats],
            'provider_stats': [{'provider': stat.api_provider, 'count': stat.count} for stat in provider_stats],
            'token_stats': {
                'total_tokens': int(token_stats.total_tokens) if token_stats.total_tokens else 0,
                'avg_tokens': round(float(token_stats.avg_tokens), 2) if token_stats.avg_tokens else 0,
                'max_tokens': int(token_stats.max_tokens) if token_stats.max_tokens else 0
            },
            'time_stats': {
                'avg_time': round(float(time_stats.avg_time), 2) if time_stats.avg_time else 0,
                'max_time': round(float(time_stats.max_time), 2) if time_stats.max_time else 0,
                'min_time': round(float(time_stats.min_time), 2) if time_stats.min_time else 0
            }
        })
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/stats')
@login_required
def admin_stats():
    """管理员统计信息（仅管理员可访问）"""
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # 总体统计
        total_users = User.query.count()
        active_users = User.query.filter_by(is_active=True).count()
        total_conversations = Conversation.query.count()
        completed_conversations = Conversation.query.filter_by(status='completed').count()
        
        # 今日统计
        today = datetime.utcnow().date()
        today_conversations = Conversation.query.filter(
            db.func.date(Conversation.created_at) == today
        ).count()
        
        today_users = User.query.filter(
            db.func.date(User.created_at) == today
        ).count()
        
        # 最受欢迎的模型
        popular_models = db.session.query(
            Conversation.model,
            db.func.count(Conversation.id).label('count')
        ).group_by(Conversation.model).order_by(db.desc('count')).limit(10).all()
        
        # 最受欢迎的API提供商
        popular_providers = db.session.query(
            Conversation.api_provider,
            db.func.count(Conversation.id).label('count')
        ).group_by(Conversation.api_provider).order_by(db.desc('count')).limit(10).all()
        
        return jsonify({
            'overview': {
                'total_users': total_users,
                'active_users': active_users,
                'total_conversations': total_conversations,
                'completed_conversations': completed_conversations,
                'success_rate': round((completed_conversations / total_conversations * 100), 2) if total_conversations > 0 else 0
            },
            'today': {
                'new_users': today_users,
                'new_conversations': today_conversations
            },
            'popular_models': [{'model': stat.model, 'count': stat.count} for stat in popular_models],
            'popular_providers': [{'provider': stat.api_provider, 'count': stat.count} for stat in popular_providers]
        })
    except Exception as e:
        logger.error(f"Error getting admin stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete-conversation/<int:conversation_id>', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    """删除对话记录"""
    try:
        conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first()
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        db.session.delete(conversation)
        db.session.commit()
        
        return jsonify({'success': True, 'message': '对话已删除'})
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    if current_user.is_authenticated:
        return render_template('404.html'), 404
    else:
        return redirect(url_for('login'))

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    if current_user.is_authenticated:
        return render_template('500.html'), 500
    else:
        return redirect(url_for('login'))

@app.errorhandler(403)
def forbidden_error(error):
    return jsonify({'error': 'Access forbidden'}), 403

# Database initialization and management
def init_database():
    """初始化数据库"""
    try:
        # 创建所有表
        db.create_all()
        logger.info("Database tables created successfully")
        
        # 创建索引
        create_indexes()
        logger.info("Database indexes created successfully")
        
        # 创建默认管理员用户
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                email='admin@reasongraph.com',
                is_admin=True
            )
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            db.session.commit()
            logger.info("Default admin user created: admin/admin123")
            
        # 创建测试用户（仅在开发环境）
        if os.getenv('FLASK_ENV') == 'development':
            test_user = User.query.filter_by(username='testuser').first()
            if not test_user:
                test_user = User(
                    username='testuser',
                    email='test@reasongraph.com'
                )
                test_user.set_password('test123')
                db.session.add(test_user)
                db.session.commit()
                logger.info("Test user created: testuser/test123")
                
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

# Database management commands
@app.cli.command()
def init_db():
    """Initialize the database."""
    init_database()
    print("Database initialized successfully!")

@app.cli.command()
def reset_db():
    """Reset the database (WARNING: This will delete all data!)."""
    if input("Are you sure you want to reset the database? This will delete all data! (yes/no): ") == 'yes':
        db.drop_all()
        init_database()
        print("Database reset successfully!")
    else:
        print("Database reset cancelled.")

@app.cli.command()
def create_admin():
    """Create an admin user."""
    username = input("Enter admin username: ")
    email = input("Enter admin email: ")
    password = input("Enter admin password: ")
    
    if User.query.filter_by(username=username).first():
        print("Username already exists!")
        return
    
    if User.query.filter_by(email=email).first():
        print("Email already exists!")
        return
    
    admin_user = User(
        username=username,
        email=email,
        is_admin=True
    )
    admin_user.set_password(password)
    
    try:
        db.session.add(admin_user)
        db.session.commit()
        print(f"Admin user '{username}' created successfully!")
    except Exception as e:
        db.session.rollback()
        print(f"Error creating admin user: {str(e)}")

# Health check endpoint
@app.route('/health')
def health_check():
    """健康检查接口"""
    try:
        # 检查数据库连接
        db.session.execute(db.text('SELECT 1'))
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected',
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }), 500

# Main execution
if __name__ == '__main__':
    try:
        with app.app_context():
            init_database()
        
        logger.info("Starting ReasonGraph application with MySQL...")
        app.run(
            debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
            host='0.0.0.0',
            port=int(os.getenv('FLASK_PORT', 5000))
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")