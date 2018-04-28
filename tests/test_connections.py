"""Test all API/DB connections."""
import unittest
import psycopg2
import os
import json

class TestConnections(unittest.TestCase):
    """Test all API/DB connections."""

    def test_db_connection(self):
        """Test connection to DB."""
        here = os.path.dirname(__file__)
        secrets_path = os.path.join(here, '../resources/secrets.json')
        env = json.load(open(secrets_path))

        conn = psycopg2.connect(database="evictions", user=env['db_user'], password=env['db_password'], host=env['db_host'], port=env['db_port'], options=f'-c search_path=evictions')
        cur = conn.cursor()

if __name__ == "__main__":
    unittest.main()
