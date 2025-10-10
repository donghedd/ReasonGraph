import logging
import os
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_migrate import Migrate
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
from sqlalchemy import or_

from database_config import DatabaseConfig
from models import db, User, Conversation, ConversationTag, UserSession, create_indexes
import config
from api_base import APIError
from reasoning_service import ReasoningService

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

# Reasoning service and config shortcuts
reasoning_service = ReasoningService()
reasoning_config = config.config
general_config = reasoning_config.general

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录后再使用推理功能'

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None

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


def extract_answer_from_result(result):
    """推断模型回答内容，优先选择结构化结果中的最终答案。"""
    parsed = getattr(result, 'parsed', None)

    if parsed is None:
        return result.raw_output

    # Self-refine prefers revised answer when可用
    if hasattr(parsed, 'revised_answer') and parsed.revised_answer:
        return parsed.revised_answer

    for attr in ('final_answer', 'answer'):
        if hasattr(parsed, attr):
            value = getattr(parsed, attr)
            if value:
                return value

    # Self-consistency stores答案在 final_answer, already handled.
    # For parsed字符串，直接返回
    if isinstance(parsed, str) and parsed.strip():
        return parsed

    # Attempt to derive from attributes commonly used in路径集合
    if hasattr(parsed, 'paths') and getattr(parsed, 'paths'):
        first_path = parsed.paths[0]
        if hasattr(first_path, 'answer') and first_path.answer:
            return first_path.answer

    return result.raw_output


def admin_required(func):
    """装饰器：仅允许管理员访问"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return login_manager.unauthorized()

        if not getattr(current_user, 'is_admin', False):
            if request.accept_mimetypes.best == 'application/json' or request.is_json:
                return jsonify({'success': False, 'error': '仅限管理员访问'}), 403
            abort(403)

        return func(*args, **kwargs)

    return wrapper


METHOD_DESCRIPTIONS = {
    'cot': '逐步拆解问题，适合大多数一般推理任务',
    'tot': '树状探索多条路径，适用于开放式与复杂分析题',
    'l2m': 'Least-to-Most 策略，先解易后解难，适合流程规划',
    'bs': 'Beam Search 多方案筛选，适合对比多种备选思路',
    'srf': 'Self-Refine 先答后校，适合需要自检和修订的任务',
    'scr': 'Self-Consistency 多路径投票，增强结论稳定性',
    'plain': '直接输出模型回答，无额外结构化处理',
}

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
        remember_me = data.get('remember_me', False)
        
        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'})
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember_me)  # 添加记住我功能
            user.update_login_time()
            return jsonify({'success': True, 'message': '登录成功'})
        else:
            return jsonify({'success': False, 'message': '用户名或密码错误'})
    
    return render_template('login.html')

# 添加获取用户历史记录的路由
@app.route('/history')
@login_required
def get_history():
    """获取用户的历史对话记录"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = (request.args.get('q') or '').strip()

    page = max(page, 1)
    per_page = max(1, min(per_page, 50))

    query = Conversation.query.filter_by(user_id=current_user.id)

    if search:
        like_pattern = f"%{search}%"
        query = query.filter(
            or_(
                Conversation.question.ilike(like_pattern),
                Conversation.answer.ilike(like_pattern),
                Conversation.raw_output.ilike(like_pattern),
                Conversation.error_message.ilike(like_pattern),
                Conversation.model.ilike(like_pattern)
            )
        )

    pagination = query.order_by(Conversation.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    conversations = pagination.items

    return jsonify({
        'conversations': [conv.to_dict() for conv in conversations],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })

@app.route('/history/<int:conversation_id>')
@login_required
def get_conversation(conversation_id):
    """获取特定对话的详细信息"""
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': '对话不存在或无权访问'}), 404
    
    return jsonify(conversation.to_dict())

@app.route('/history/<int:conversation_id>/delete', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    """删除特定对话"""
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': '对话不存在或无权访问'}), 404
    
    try:
        db.session.delete(conversation)
        db.session.commit()
        return jsonify({'success': True, 'message': '对话已删除'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"删除对话错误: {str(e)}")
        return jsonify({'success': False, 'message': '删除对话失败'})

@app.route('/register', methods=['GET', 'POST'])
@login_required
@admin_required
def register():
    """仅管理员可创建新用户"""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip()
        password = (data.get('password') or '').strip()

        if not all([username, email, password]):
            return jsonify({'success': False, 'message': '用户名、邮箱和密码均为必填项'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': '用户名已存在'}), 409

        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': '邮箱已被使用'}), 409

        user = User(username=username, email=email)
        user.set_password(password)

        try:
            db.session.add(user)
            db.session.commit()
            logger.info("管理员 %s 创建用户 %s", current_user.username, username)
            return jsonify({'success': True, 'message': '用户创建成功'}), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {str(e)}")
            return jsonify({'success': False, 'message': '创建失败，请稍后重试'}), 500

    return render_template('register.html', admin=current_user)

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    """管理员控制台"""
    provider_options = [
        {
            'id': provider_id,
            'name': general_config.provider_display_names.get(provider_id, provider_id)
        }
        for provider_id in general_config.providers
    ]

    method_options = [
        {
            'id': method_id,
            'name': reasoning_config.methods[method_id].name
        }
        for method_id in reasoning_config.methods
    ]

    return render_template(
        'admin_dashboard.html',
        admin=current_user,
        provider_options=provider_options,
        method_options=method_options
    )


@app.route('/admin/users', methods=['GET'])
@login_required
@admin_required
def admin_users():
    """管理员查看用户列表"""
    search = (request.args.get('q') or '').strip()

    query = User.query.order_by(User.created_at.desc())
    if search:
        like_pattern = f"%{search}%"
        query = query.filter(or_(User.username.ilike(like_pattern), User.email.ilike(like_pattern)))

    users = [user.to_dict() for user in query.all()]
    return jsonify({'success': True, 'users': users})


@app.route('/admin/history', methods=['GET'])
@login_required
@admin_required
def admin_history():
    """管理员全量搜索历史记录"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = max(1, min(per_page, 50))
    page = max(page, 1)

    keyword = (request.args.get('q') or '').strip()
    username = (request.args.get('username') or '').strip()
    status = (request.args.get('status') or '').strip()
    provider = (request.args.get('provider') or '').strip()
    model = (request.args.get('model') or '').strip()
    method = (request.args.get('method') or '').strip()
    start_date = (request.args.get('start') or '').strip()
    end_date = (request.args.get('end') or '').strip()

    query = Conversation.query.join(User)

    if keyword:
        like_pattern = f"%{keyword}%"
        query = query.filter(
            or_(
                Conversation.question.ilike(like_pattern),
                Conversation.answer.ilike(like_pattern),
                Conversation.raw_output.ilike(like_pattern),
                Conversation.error_message.ilike(like_pattern)
            )
        )

    if username:
        query = query.filter(User.username.ilike(f"%{username}%"))

    if status:
        query = query.filter(Conversation.status == status)

    if provider:
        query = query.filter(Conversation.api_provider == provider)

    if model:
        query = query.filter(Conversation.model == model)

    if method:
        query = query.filter(Conversation.reasoning_method == method)

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
            query = query.filter(Conversation.created_at >= start_dt)
        except ValueError:
            logger.warning("无效开始日期: %s", start_date)

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
            # 包含当天需加一天
            query = query.filter(Conversation.created_at < end_dt + timedelta(days=1))
        except ValueError:
            logger.warning("无效结束日期: %s", end_date)

    pagination = query.order_by(Conversation.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    conversations = []
    for conv in pagination.items:
        data = conv.to_dict()
        if conv.user:
            data['username'] = conv.user.username
            data['email'] = conv.user.email
        conversations.append(data)

    return jsonify({
        'success': True,
        'conversations': conversations,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page,
        'per_page': pagination.per_page
    })


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
        # 获取请求数据
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': '未提供数据'
            }), 400

        # 提取参数
        provider = (data.get('provider') or '').strip().lower()
        if not provider:
            return jsonify({'success': False, 'error': '需要指定API提供商'}), 400

        api_key = (data.get('api_key') or '').strip()
        if not api_key:
            api_key = general_config.get_default_api_key(provider) or ''
        if not api_key:
            return jsonify({'success': False, 'error': '请提供有效的API密钥'}), 400

        question = (data.get('question') or '').strip()
        if not question:
            return jsonify({'success': False, 'error': '需要提问内容'}), 400

        provider_models = general_config.provider_model_map.get(provider, [])
        model = (data.get('model') or '').strip() or (provider_models[0] if provider_models else None)
        if not model:
            return jsonify({'success': False, 'error': '未找到可用模型，请检查提供商设置'}), 400

        reasoning_method = (data.get('reasoning_method') or 'cot').strip().lower()
        method_config = config.get_method_config(reasoning_method) or {}
        prompt_template = method_config.get('prompt_format')

        try:
            max_tokens = int(data.get('max_tokens', general_config.max_tokens))
        except (TypeError, ValueError):
            max_tokens = general_config.max_tokens
        max_tokens = max(1, max_tokens)

        try:
            chars_per_line = int(data.get('chars_per_line', general_config.chars_per_line))
        except (TypeError, ValueError):
            chars_per_line = general_config.chars_per_line
        chars_per_line = max(10, min(200, chars_per_line))

        try:
            max_lines = int(data.get('max_lines', general_config.max_lines))
        except (TypeError, ValueError):
            max_lines = general_config.max_lines
        max_lines = max(3, min(30, max_lines))

        # 获取客户端信息
        client_info = get_client_info()

        # 创建对话记录
        conversation = Conversation(
            user_id=current_user.id,
            question=question,
            api_provider=provider,
            model=model,
            reasoning_method=reasoning_method,
            max_tokens=max_tokens,
            chars_per_line=chars_per_line,
            max_lines=max_lines,
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
            result = reasoning_service.run(
                provider=provider,
                api_key=api_key,
                model=model,
                question=question,
                reasoning_method=reasoning_method,
                prompt_template=prompt_template,
                max_tokens=max_tokens,
                chars_per_line=chars_per_line,
                max_lines=max_lines,
            )

            visualization_data = result.visualization
            if visualization_data:
                conversation.mermaid_code = visualization_data.get('code')

            answer_text = extract_answer_from_result(result)

            token_usage = {
                'input_tokens': len(question) // 4,
                'output_tokens': len(result.raw_output) // 4,
                'total_tokens': (len(question) + len(result.raw_output)) // 4
            }

            conversation.mark_as_completed(
                answer=answer_text,
                raw_output=result.raw_output,
                visualization_data=visualization_data,
                token_usage=token_usage
            )

            update_user_stats(current_user, token_usage['total_tokens'])
            db.session.commit()

            return jsonify({
                'success': True,
                'conversation_id': conversation.id,
                'raw_output': result.raw_output,
                'answer': answer_text,
                'visualization': visualization_data,
                'token_usage': token_usage,
                'processing_time': conversation.processing_time
            })

        except APIError as api_err:
            logger.error("第三方API错误: %s", api_err)
            conversation.mark_as_failed(str(api_err))
            db.session.commit()
            return jsonify({'success': False, 'error': str(api_err)}), 502
        except Exception as processing_error:
            logger.error(f"推理处理错误: {str(processing_error)}")
            conversation.mark_as_failed(str(processing_error))
            db.session.commit()
            return jsonify({
                'success': False,
                'error': f'处理推理请求时发生错误: {str(processing_error)}'
            }), 500

    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
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
        question = (data.get('question') or '').strip()

        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required for method selection'
            }), 400

        lowered = question.lower()
        length = len(question)

        method_id = 'cot'

        if any(keyword in lowered for keyword in ['revise', 'refine', 'improve', '校正', '优化', '审查']):
            method_id = 'srf'
        elif any(keyword in lowered for keyword in ['multiple answers', 'possibilities', 'uncertain', '多种答案', '多种可能']):
            method_id = 'scr'
        elif any(keyword in lowered for keyword in ['step', 'steps', 'procedure', '流程', '计划', '步骤', 'how to', '如何', '怎么做']):
            method_id = 'l2m'
        elif any(keyword in lowered for keyword in ['compare', '对比', 'pros', 'cons', '方案', '最优']):
            method_id = 'bs'
        elif length > 280 or any(keyword in lowered for keyword in ['why', '分析', '深入', '根因', 'comprehensive', 'detailed']):
            method_id = 'tot'
        elif any(keyword in lowered for keyword in ['code', '编程', '实现代码', '函数', 'snippet']):
            method_id = 'plain'

        method_cfg = config.get_method_config(method_id) or {}
        selected_method = {
            'method_id': method_id,
            'name': method_cfg.get('name', method_id),
            'description': METHOD_DESCRIPTIONS.get(method_id, method_cfg.get('name', ''))
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

        question = (data.get('question') or '').strip()
        if not question:
            return jsonify({
                'success': False,
                'error': 'Question is required'
            }), 400

        provider = (data.get('provider') or '').strip().lower()
        if not provider:
            return jsonify({'success': False, 'error': '需要指定API提供商'}), 400

        api_key = (data.get('api_key') or '').strip()
        if not api_key:
            api_key = general_config.get_default_api_key(provider) or ''
        if not api_key:
            return jsonify({'success': False, 'error': '请提供有效的API密钥'}), 400

        provider_models = general_config.provider_model_map.get(provider, [])
        model = (data.get('model') or '').strip() or (provider_models[0] if provider_models else None)
        if not model:
            return jsonify({'success': False, 'error': '未找到可用模型，请检查提供商设置'}), 400

        try:
            max_tokens = int(data.get('max_tokens', general_config.max_tokens * 2))
        except (TypeError, ValueError):
            max_tokens = general_config.max_tokens * 2
        max_tokens = max(1, max_tokens)

        try:
            chars_per_line = int(data.get('chars_per_line', general_config.chars_per_line))
        except (TypeError, ValueError):
            chars_per_line = general_config.chars_per_line
        chars_per_line = max(10, min(200, chars_per_line))

        try:
            max_lines = int(data.get('max_lines', general_config.max_lines))
        except (TypeError, ValueError):
            max_lines = general_config.max_lines
        max_lines = max(3, min(40, max_lines))

        prompt_template = data.get('prompt_format') or None

        # 获取客户端信息
        client_info = get_client_info()

        # 创建对话记录
        conversation = Conversation(
            user_id=current_user.id,
            question=question,
            api_provider=provider,
            model=model,
            reasoning_method='long_reasoning',
            max_tokens=max_tokens,
            chars_per_line=chars_per_line,
            max_lines=max_lines,
            client_ip=client_info['ip'],
            user_agent=client_info['user_agent'],
            status='pending'
        )

        db.session.add(conversation)
        db.session.commit()

        # 标记开始处理
        conversation.mark_as_started()
        db.session.commit()

        try:
            result = reasoning_service.run(
                provider=provider,
                api_key=api_key,
                model=model,
                question=question,
                reasoning_method='long_reasoning',
                prompt_template=prompt_template,
                max_tokens=max_tokens,
                chars_per_line=chars_per_line,
                max_lines=max_lines,
            )

            answer_text = extract_answer_from_result(result)
            token_usage = {
                'input_tokens': len(question) // 4,
                'output_tokens': len(result.raw_output) // 4,
                'total_tokens': (len(question) + len(result.raw_output)) // 4
            }

            conversation.mark_as_completed(
                answer=answer_text,
                raw_output=result.raw_output,
                visualization_data=None,
                token_usage=token_usage
            )

            update_user_stats(current_user, token_usage['total_tokens'])
            db.session.commit()

            return jsonify({
                'success': True,
                'conversation_id': conversation.id,
                'raw_output': result.raw_output,
                'answer': answer_text,
                'token_usage': token_usage,
                'processing_time': conversation.processing_time
            })

        except APIError as api_err:
            logger.error("第三方API错误: %s", api_err)
            conversation.mark_as_failed(str(api_err))
            db.session.commit()
            return jsonify({'success': False, 'error': str(api_err)}), 502
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
def delete_conversation_api(conversation_id):
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
            if os.getenv('AUTO_INIT_DB', 'false').lower() == 'true':
                logger.info("AUTO_INIT_DB=true，执行数据库初始化流程")
                init_database()
            else:
                logger.info("跳过自动数据库初始化，假设数据库和表已存在")
        
        logger.info("Starting ReasonGraph application with MySQL...")
        app.run(
            debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
            host='0.0.0.0',
            port=int(os.getenv('FLASK_PORT', 5000))
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
