import mysql.connector
from datetime import datetime, timedelta
import json

class DatabaseMonitor:
    def __init__(self, config):
        self.config = config
    
    def get_connection(self):
        return mysql.connector.connect(
            host=self.config['host'],
            port=self.config['port'],
            user=self.config['user'],
            password=self.config['password'],
            database=self.config['database']
        )
    
    def get_table_stats(self):
        """获取表统计信息"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                table_name,
                table_rows,
                data_length,
                index_length,
                (data_length + index_length) as total_size
            FROM information_schema.tables 
            WHERE table_schema = %s
        """, (self.config['database'],))
        
        stats = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return stats
    
    def get_slow_queries(self, limit=10):
        """获取慢查询信息"""
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(f"""
            SELECT 
                sql_text,
                exec_count,
                avg_timer_wait/1000000000 as avg_time_seconds,
                max_timer_wait/1000000000 as max_time_seconds
            FROM performance_schema.events_statements_summary_by_digest 
            WHERE schema_name = %s
            ORDER BY avg_timer_wait DESC 
            LIMIT {limit}
        """, (self.config['database'],))
        
        queries = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return queries
    
    def generate_report(self):
        """生成监控报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'table_stats': self.get_table_stats(),
            'slow_queries': self.get_slow_queries()
        }
        
        return report

if __name__ == '__main__':
    from database_config import DatabaseConfig
    
    config = {
        'host': DatabaseConfig.MYSQL_HOST,
        'port': DatabaseConfig.MYSQL_PORT,
        'user': DatabaseConfig.MYSQL_USER,
        'password': DatabaseConfig.MYSQL_PASSWORD,
        'database': DatabaseConfig.MYSQL_DATABASE
    }
    
    monitor = DatabaseMonitor(config)
    report = monitor.generate_report()
    
    print(json.dumps(report, indent=2, ensure_ascii=False))
