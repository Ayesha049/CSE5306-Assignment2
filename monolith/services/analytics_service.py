"""
N6: Analytics Service (Monolithic)
──────────────────────────────────
• Stores metrics in MySQL
• Called directly by:
    - BufferService
    - LearnerService
    - EnvironmentService
    - exp_service (for status queries)
• No gRPC
"""

import time
import mysql.connector
from mysql.connector import pooling

import analytics_pb2


# ── MySQL connection pool ─────────────────────────────────────────────
DB_CONFIG = {
    "host": "rl-mysql-mono",
    "user": "root",
    "password": "password",
    "database": "rl_analytics",
    "autocommit": True
}

POOL_NAME = "analytics_pool"
POOL_SIZE = 10

connection_pool = pooling.MySQLConnectionPool(
    pool_name=POOL_NAME,
    pool_size=POOL_SIZE,
    **DB_CONFIG
)


def get_conn():
    retries = 10
    for i in range(retries):
        try:
            return connection_pool.get_connection()
        except mysql.connector.Error as e:
            print(f"[Analytics-Mono] Pool error ({i+1}/{retries}): {e}")
            time.sleep(2)

    raise Exception("Failed to get MySQL connection")


# ════════════════════════════════════════════════════════════════════════════════
#  Analytics Service
# ════════════════════════════════════════════════════════════════════════════════

class AnalyticsService:

    def __init__(self):
        print("[Analytics-Mono] Initializing...")
        self.init_db()
        print("[Analytics-Mono] Ready.")

    # ────────────────────────────────────────────────────────────────
    # Database initialization
    # ────────────────────────────────────────────────────────────────

    def init_db(self):
        conn = get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INT AUTO_INCREMENT PRIMARY KEY,
                node_name VARCHAR(50) UNIQUE,
                node_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_types (
                metric_type_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(50) UNIQUE,
                description VARCHAR(200)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INT AUTO_INCREMENT PRIMARY KEY,
                node_id INT,
                metric_type_id INT,
                value FLOAT,
                experiment_id VARCHAR(100),
                step BIGINT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id),
                FOREIGN KEY (metric_type_id) REFERENCES metric_types(metric_type_id)
            )
        """)

        cursor.close()
        conn.close()

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    def _get_or_create_node(self, node_name, node_type, cursor):
        cursor.execute("SELECT node_id FROM nodes WHERE node_name=%s", (node_name,))
        row = cursor.fetchone()

        if row:
            return row[0]

        cursor.execute(
            "INSERT INTO nodes (node_name, node_type) VALUES (%s, %s)",
            (node_name, node_type)
        )
        return cursor.lastrowid

    def _get_or_create_metric_type(self, metric_name, cursor):
        cursor.execute("SELECT metric_type_id FROM metric_types WHERE name=%s", (metric_name,))
        row = cursor.fetchone()

        if row:
            return row[0]

        cursor.execute(
            "INSERT INTO metric_types (name) VALUES (%s)",
            (metric_name,)
        )
        return cursor.lastrowid

    # ────────────────────────────────────────────────────────────────
    # Public API (same names as gRPC)
    # ────────────────────────────────────────────────────────────────

    def ReportMetrics(self, request):

        try:
            conn = get_conn()
            cursor = conn.cursor()

            node_id = self._get_or_create_node(
                request.source_node,
                "unknown",
                cursor
            )

            metric_type_id = self._get_or_create_metric_type(
                request.metric_name,
                cursor
            )

            cursor.execute("""
                INSERT INTO metrics
                (node_id, metric_type_id, value, experiment_id, step)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                node_id,
                metric_type_id,
                request.value,
                request.experiment_id,
                request.step
            ))

            cursor.close()
            conn.close()

            return analytics_pb2.Status(ok=True)

        except mysql.connector.Error as e:
            print(f"[Analytics-Mono ERROR] {e}")
            return analytics_pb2.Status(ok=False)

    def QueryMetrics(self, request):

        try:
            conn = get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT m.value, m.timestamp, n.node_name, mt.name
                FROM metrics m
                JOIN nodes n ON m.node_id = n.node_id
                JOIN metric_types mt ON m.metric_type_id = mt.metric_type_id
                WHERE mt.name=%s
            """, (request.metric_name,))

            rows = cursor.fetchall()

            cursor.close()
            conn.close()

            response = analytics_pb2.MetricList()

            for value, timestamp, node_name, metric_name in rows:
                metric = response.metrics.add()
                metric.metric_name = metric_name
                metric.value = value
                metric.source_node = node_name

            return response

        except mysql.connector.Error as e:
            print(f"[Analytics-Mono ERROR] {e}")
            return analytics_pb2.MetricList()