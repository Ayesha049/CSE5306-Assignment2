import sys
sys.path.append("../../stubs")

import grpc
from concurrent import futures
import analytics_pb2
import analytics_pb2_grpc
import mysql.connector
import time

# MySQL connection config
DB_CONFIG = {
    "host": "localhost",   # change to mysql container name in Docker
    "user": "root",
    "password": "password",
    "database": "rl_analytics"
}



class AnalyticsService(analytics_pb2_grpc.AnalyticsServiceServicer):
    def __init__(self):
        self.conn = mysql.connector.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor()
        print("[Analytics] Connected to MySQL")
        self.init_db()

    def get_or_create_node(self, node_name, node_type):
        self.cursor.execute(
            "SELECT node_id FROM nodes WHERE node_name=%s",
            (node_name,)
        )
        result = self.cursor.fetchone()

        if result:
            return result[0]

        self.cursor.execute(
            "INSERT INTO nodes (node_name, node_type) VALUES (%s, %s)",
            (node_name, node_type)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def get_or_create_metric_type(self, metric_name):
        self.cursor.execute(
            "SELECT metric_type_id FROM metric_types WHERE name=%s",
            (metric_name,)
        )
        result = self.cursor.fetchone()

        if result:
            return result[0]

        self.cursor.execute(
            "INSERT INTO metric_types (name) VALUES (%s)",
            (metric_name,)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def init_db(self):
        # Ensure nodes table exists
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id INT AUTO_INCREMENT PRIMARY KEY,
                node_name VARCHAR(50) UNIQUE,
                node_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure metric_types table exists
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metric_types (
                metric_type_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(50) UNIQUE,
                description VARCHAR(200)
            )
        """)

        # Ensure metrics table exists with experiment_id and step columns
        self.cursor.execute("""
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

        self.conn.commit()
        print("[Analytics] Tables ensured.")

    def ReportMetrics(self, request, context):
        try:
            node_id = self.get_or_create_node(
                request.source_node,   # FIXED
                "unknown"              # since proto has no node_type
            )

            metric_type_id = self.get_or_create_metric_type(
                request.metric_name
            )

            self.cursor.execute(
                """
                INSERT INTO metrics (node_id, metric_type_id, value, experiment_id, step)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    node_id,
                    metric_type_id,
                    request.value,
                    request.experiment_id,
                    request.step
                )
            )
            self.conn.commit()

            print(f"[Analytics] {request.metric_name}={request.value} from {request.source_node}")

            return analytics_pb2.Status(ok=True)

        except Exception as e:
            print(f"[Analytics ERROR] {e}")
            return analytics_pb2.Status(ok=False)
        
    def QueryMetrics(self, request, context):
        try:
            self.cursor.execute("""
                SELECT m.value, m.timestamp, n.node_name, mt.name
                FROM metrics m
                JOIN nodes n ON m.node_id = n.node_id
                JOIN metric_types mt ON m.metric_type_id = mt.metric_type_id
                WHERE mt.name=%s
            """, (request.metric_name,))

            rows = self.cursor.fetchall()

            response = analytics_pb2.MetricList()

            for row in rows:
                metric = response.metrics.add()
                metric.metric_name = row[3]
                metric.value = row[0]
                metric.source_node = row[2]

            return response

        except Exception as e:
            print(f"[Query ERROR] {e}")
            return analytics_pb2.MetricList()
            


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    analytics_pb2_grpc.add_AnalyticsServiceServicer_to_server(
        AnalyticsService(), server
    )
    server.add_insecure_port('[::]:50052')
    server.start()
    print("N6 Analytics Service running on port 50052")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()