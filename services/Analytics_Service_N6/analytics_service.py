import sys
sys.path.append("/app/stubs")

import grpc
from concurrent import futures
import time
import mysql.connector
from mysql.connector import pooling

import analytics_pb2
import analytics_pb2_grpc

# ─────────────────────────────────────────────────────────────
# MySQL Configuration
# ─────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host": "rl-mysql",
    "user": "root",
    "password": "password",
    "database": "rl_analytics",
    "autocommit": True
}

POOL_NAME = "analytics_pool"
POOL_SIZE = 10

connection_pool = None


# ─────────────────────────────────────────────────────────────
# MySQL Initialization with Retry (Fixes Startup Race)
# ─────────────────────────────────────────────────────────────

def init_connection_pool():
    global connection_pool

    retries = 20
    for i in range(retries):
        try:
            connection_pool = pooling.MySQLConnectionPool(
                pool_name=POOL_NAME,
                pool_size=POOL_SIZE,
                **DB_CONFIG
            )
            print("[Analytics] MySQL connection pool established.")
            return
        except mysql.connector.Error as e:
            print(f"[Analytics] Waiting for MySQL ({i+1}/{retries})... {e}")
            time.sleep(3)

    raise Exception("Failed to connect to MySQL after retries.")


def get_conn():
    return connection_pool.get_connection()


# ─────────────────────────────────────────────────────────────
# Analytics Service Implementation
# ─────────────────────────────────────────────────────────────

class AnalyticsService(analytics_pb2_grpc.AnalyticsServiceServicer):

    def __init__(self):
        print("[Analytics] Initializing service...")
        self.init_db()
        print("[Analytics] Service initialized.")

    def init_db(self):
        conn = get_conn()
        cursor = conn.cursor()

        # Nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INT AUTO_INCREMENT PRIMARY KEY,
                node_name VARCHAR(50) UNIQUE,
                node_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Metric types table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_types (
                metric_type_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(50) UNIQUE,
                description VARCHAR(200)
            )
        """)

        # Metrics table
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
        print("[Analytics] Tables ensured.")

    # ─────────────────────────────────────────────────────────

    def get_or_create_node(self, node_name, node_type, cursor):
        cursor.execute("SELECT node_id FROM nodes WHERE node_name=%s", (node_name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            "INSERT INTO nodes (node_name, node_type) VALUES (%s, %s)",
            (node_name, node_type)
        )
        return cursor.lastrowid

    def get_or_create_metric_type(self, metric_name, cursor):
        cursor.execute("SELECT metric_type_id FROM metric_types WHERE name=%s", (metric_name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            "INSERT INTO metric_types (name) VALUES (%s)",
            (metric_name,)
        )
        return cursor.lastrowid

    # ─────────────────────────────────────────────────────────

    def ReportMetrics(self, request, context):
        try:
            conn = get_conn()
            cursor = conn.cursor()

            node_id = self.get_or_create_node(request.source_node, "unknown", cursor)
            metric_type_id = self.get_or_create_metric_type(request.metric_name, cursor)

            cursor.execute("""
                INSERT INTO metrics (node_id, metric_type_id, value, experiment_id, step)
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
            print(f"[Analytics ERROR] MySQL Error: {e}")
            return analytics_pb2.Status(ok=False)

    # ─────────────────────────────────────────────────────────

    def QueryMetrics(self, request, context):
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
            print(f"[Analytics ERROR] MySQL Query Error: {e}")
            return analytics_pb2.MetricList()


# ─────────────────────────────────────────────────────────────
# Server Entry Point
# ─────────────────────────────────────────────────────────────

def serve():
    init_connection_pool()   # ✅ Wait until MySQL is ready

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    analytics_pb2_grpc.add_AnalyticsServiceServicer_to_server(
        AnalyticsService(), server
    )

    server.add_insecure_port('[::]:50052')
    server.start()
    print("[Analytics] N6 Analytics Service running on port 50052")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()