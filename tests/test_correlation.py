"""Locally test Correlation Plot."""
import matplotlib.pyplot as plt
import unittest
from src.analysis.correlation import plot_corr
from src.db_client import DBClient

class TestCorrelation(unittest.TestCase):
    """Locally test Correlation Plot."""

    def setUp(self):
        """Get DB connection."""
        self.client = DBClient()

if __name__=="__main__":
    unittest.main()
