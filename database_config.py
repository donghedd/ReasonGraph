import os
from urllib.parse import quote_plus

class DatabaseConfig:
    """数据库配置类"""
    
    # MySQL数据库配置
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = os.getenv('MYSQL_PORT', 3306)
    MYSQL_USER = os.getenv('MYSQL_USER', 'reasongraph_user')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'your_password_here')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'reasongraph_db')
    
    @classmethod
    def get_database_uri(cls):
        """获取数据库连接URI"""
        # URL编码密码以处理特殊字符
        password_encoded = quote_plus(cls.MYSQL_PASSWORD)
        
        return f"mysql+pymysql://{cls.MYSQL_USER}:{password_encoded}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}?charset=utf8mb4"
    
    @classmethod
    def get_config_dict(cls):
        """获取Flask配置字典"""
        return {
            'SQLALCHEMY_DATABASE_URI': cls.get_database_uri(),
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'pool_size': 10,
                'pool_recycle': 120,
                'pool_pre_ping': True,
                'connect_args': {
                    'charset': 'utf8mb4'
                }
            }
        }
