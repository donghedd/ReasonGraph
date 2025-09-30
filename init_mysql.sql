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
