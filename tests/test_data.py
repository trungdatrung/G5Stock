import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date
from data import load_data

class TestData(unittest.TestCase):
    
    @patch('data.yf.download')
    def test_load_data_success(self, mock_download):
        """Test successful data fetching."""
        # Mocking the return value
        mock_df = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        mock_df.index.name = 'Date'
        mock_download.return_value = mock_df
        
        result = load_data('AAPL', date(2023, 1, 1), date(2023, 1, 3))
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
        self.assertIn('Date', result.columns) # reset_index adds Date
        self.assertIn('Close', result.columns)

    @patch('data.yf.download')
    def test_load_data_empty(self, mock_download):
        """Test handling of empty data."""
        mock_download.return_value = pd.DataFrame()
        
        result = load_data('INVALID', date(2023, 1, 1), date(2023, 1, 3))
        
        self.assertIsNone(result)

    @patch('data.yf.download')
    def test_load_data_exception(self, mock_download):
        """Test exception handling."""
        mock_download.side_effect = Exception("API Error")
        
        result = load_data('ERROR', date(2023, 1, 1), date(2023, 1, 3))
        
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
