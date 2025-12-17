import unittest
import pandas as pd
from ui import get_xaxis_layout

class TestXAxis(unittest.TestCase):
    def test_range_less_equal_1_day(self):
        start = pd.Timestamp("2023-01-01 09:00")
        end = pd.Timestamp("2023-01-01 16:00")
        config = get_xaxis_layout(start, end)
        self.assertEqual(config['tickformat'], "%H:%M")

    def test_range_1_month(self):
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-01-15")
        config = get_xaxis_layout(start, end)
        self.assertEqual(config['tickformat'], "%b %d")

    def test_range_1_year(self):
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-06-01")
        config = get_xaxis_layout(start, end)
        self.assertEqual(config['tickformat'], "%b")

    def test_range_multi_year(self):
        start = pd.Timestamp("2022-01-01")
        end = pd.Timestamp("2024-01-01")
        config = get_xaxis_layout(start, end)
        self.assertEqual(config['tickformat'], "%Y")

if __name__ == '__main__':
    unittest.main()
