"""Locally test Correlation Plot."""
import matplotlib.pyplot as plt
import unittest
from src.analysis.correlation import plot_corr
from src.db_init import db_connect

class TestCorrelation(unittest.TestCase):
    """Locally test Correlation Plot."""

    def setUp(self):
        """Get DB connection."""
        self.cur,self.conn = db_connect()

    @unittest.SkipTest
    def test_correlation_matrix(self):
        """Test correlation plot."""
        plt.ioff()
        plot_corr(self.cur)

if __name__=="__main__":
    unittest.main()
