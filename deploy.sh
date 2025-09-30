#!/bin/bash

# ReasonGraph MySQL 部署脚本

echo "=== ReasonGraph MySQL 部署脚本 ==="

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 检查 MySQL
if ! command -v mysql &> /dev/null; then
    echo "错误: MySQL 未安装"
    exit 1
fi

# 安装 Python 依赖
echo "1. 安装 Python 依赖..."
pip3 install -r requirements.txt

# 设置环境变量
echo "2. 设置环境变量..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "请编辑 .env 文件设置数据库配置"
fi

# 初始化数据库
echo "3. 初始化 MySQL 数据库..."
read -p "请输入 MySQL root 密码: " -s mysql_root_password
echo

mysql -u root -p$mysql_root_password < init_mysql.sql

# 初始化 Flask 数据库
echo "4. 初始化 Flask 应用数据库..."
export FLASK_APP=app.py
flask init-db

# 创建管理员用户
echo "5. 创建管理员用户..."
flask create-admin

echo "=== 部署完成 ==="
echo "使用以下命令启动应用:"
echo "python3 app.py"
