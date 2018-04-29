"""Test all API/DB connections."""
import unittest
import psycopg2
import os
import json
from src.db_init import db_connect

class TestConnections(unittest.TestCase):
    """Test all API/DB connections."""

    def test_db_connection(self):
        """Test connection to DB."""
        cur = db_connect()
        cur.execute("select distinct state from evictions.blockgroup;")
        l = cur.fetchall()
        self.assertIsNotNone(l)

        print(l)

if __name__ == "__main__":
    unittest.main()
