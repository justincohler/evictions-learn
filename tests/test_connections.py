"""Test all API/DB connections."""
import unittest
import psycopg2
import os
import json
from src.db_client import DBClient

class TestConnections(unittest.TestCase):
    """Test all API/DB connections."""

    def test_db_connection(self):
        """Test connection to DB."""
        client = DBClient()
        l = client.read("select distinct state from evictions.blockgroup;")
        self.assertIsNotNone(l)

        print(l)

if __name__ == "__main__":
    unittest.main()
