-- 创建数据库和用户
CREATE DATABASE IF NOT EXISTS reasongraph_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建用户（如果不存在）
CREATE USER IF NOT EXISTS 'reasongraph_user'@'localhost' IDENTIFIED BY 'your_secure_password_here';
CREATE USER IF NOT EXISTS 'reasongraph_user'@'%' IDENTIFIED BY 'your_secure_password_here';

-- 授予权限
GRANT ALL PRIVILEGES ON reasongraph_db.* TO 'reasongraph_user'@'localhost';
GRANT ALL PRIVILEGES ON reasongraph_db.* TO 'reasongraph_user'@'%';

-- 刷新权限
FLUSH PRIVILEGES;

-- 使用数据库
USE reasongraph_db;

-- 设置字符集
SET NAMES utf8mb4;

-- 删除旧表以避免重复创建冲突（注意：将清空表内数据）
DROP TABLE IF EXISTS conversation_tags;
DROP TABLE IF EXISTS user_sessions;
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS users;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID',
    username VARCHAR(80) NOT NULL UNIQUE COMMENT '用户名',
    email VARCHAR(255) NOT NULL UNIQUE COMMENT '邮箱',
    password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希',
    is_active TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否激活',
    is_admin TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否管理员',
    show_visualization TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否展示可视化界面',
    api_keys LONGTEXT NULL COMMENT 'API密钥配置(JSON)',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    last_login DATETIME NULL COMMENT '最后登录时间',
    total_conversations INT NOT NULL DEFAULT 0 COMMENT '总对话数',
    total_tokens_used BIGINT NOT NULL DEFAULT 0 COMMENT '总使用Token数'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 对话表
CREATE TABLE IF NOT EXISTS conversations (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '对话ID',
    user_id INT NOT NULL COMMENT '用户ID',
    question TEXT NOT NULL COMMENT '用户问题',
    answer LONGTEXT NULL COMMENT 'AI回答',
    raw_output LONGTEXT NULL COMMENT '原始输出',
    api_provider VARCHAR(50) NULL COMMENT 'API提供商',
    model VARCHAR(100) NULL COMMENT '使用的模型',
    reasoning_method VARCHAR(100) NULL COMMENT '推理方法',
    max_tokens INT NULL COMMENT '最大Token数',
    chars_per_line INT NULL COMMENT '每行字符数',
    max_lines INT NULL COMMENT '最大行数',
    input_tokens INT NULL COMMENT '输入Token数',
    output_tokens INT NULL COMMENT '输出Token数',
    total_tokens INT NULL COMMENT '总Token数',
    visualization_data LONGTEXT NULL COMMENT '可视化数据(JSON)',
    mermaid_code LONGTEXT NULL COMMENT 'Mermaid图表代码',
    status VARCHAR(20) NOT NULL DEFAULT 'pending' COMMENT '状态',
    error_message TEXT NULL COMMENT '错误信息',
    processing_time DOUBLE NULL COMMENT '处理时间(秒)',
    response_quality_score DOUBLE NULL COMMENT '回答质量评分',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    started_at DATETIME NULL COMMENT '开始处理时间',
    completed_at DATETIME NULL COMMENT '完成时间',
    client_ip VARCHAR(45) NULL COMMENT '客户端IP',
    user_agent VARCHAR(500) NULL COMMENT '用户代理',
    session_id VARCHAR(100) NULL COMMENT '会话ID',
    CONSTRAINT fk_conversations_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_conversations_user_id (user_id),
    INDEX idx_conversations_created_at (created_at),
    INDEX idx_conversations_status (status),
    INDEX idx_conversations_api_provider (api_provider),
    INDEX idx_conversations_model (model),
    INDEX idx_conversations_reasoning_method (reasoning_method)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 对话标签表
CREATE TABLE IF NOT EXISTS conversation_tags (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '标签ID',
    conversation_id INT NOT NULL COMMENT '对话ID',
    tag_name VARCHAR(50) NOT NULL COMMENT '标签名称',
    tag_value VARCHAR(200) NULL COMMENT '标签值',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    CONSTRAINT fk_tags_conversation_id FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    INDEX idx_tags_conversation_id (conversation_id),
    INDEX idx_tags_name (tag_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 用户会话表
CREATE TABLE IF NOT EXISTS user_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '记录ID',
    user_id INT NOT NULL COMMENT '用户ID',
    session_token VARCHAR(255) NOT NULL UNIQUE COMMENT '会话令牌',
    ip_address VARCHAR(45) NULL COMMENT 'IP地址',
    user_agent VARCHAR(500) NULL COMMENT '用户代理',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    expires_at DATETIME NOT NULL COMMENT '过期时间',
    is_active TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否有效',
    CONSTRAINT fk_sessions_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_sessions_user_id (user_id),
    INDEX idx_sessions_token (session_token),
    INDEX idx_sessions_expires (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
