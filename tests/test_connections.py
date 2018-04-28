"""Test all API/DB connections."""
import unittest
import psycopg2
import os
import json

class TestConnections(unittest.TestCase):
    """Test all API/DB connections."""

    def test_db_connection(self):
        """Test connection to DB."""
        conn = psycopg2.connect(database="evictions"
                                , user=os.environ['DB_USER']
                                , password=os.environ['DB_PASSWORD']
                                , host=os.environ['DB_HOST']
                                , port=os.environ['DB_PORT']
                                , options=f'-c search_path=evictions')
        cur = conn.cursor()

if __name__ == "__main__":
    unittest.main()
