import matplotlib.pyplot as plt
import unittest
from src.analysis.correlation import plot_corr
from src.db_init import db_connect

class TestCorrelation(unittest.TestCase):

    def setUp(self):
        self.cur,self.conn = db_connect()

    def test_correlation_matrix(self):
        plot_corr(self.cur)
        
if __name__=="__main__":
    unittest.main()
