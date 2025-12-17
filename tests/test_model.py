import unittest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from model import train_model

class TestModel(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        dates = [date.today() - timedelta(days=x) for x in range(100)]
        dates.reverse()
        # Linear trend: y = 2x + 10
        prices = [10 + 2*i + np.random.normal(0, 0.1) for i in range(100)]
        
        self.df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Close': prices
        })

    def test_train_model_output_structure(self):
        """Test that the function returns the correct structure."""
        future_days = 5
        model, df_pred, future_df, r2, mse = train_model(self.df, future_days)
        
        self.assertIsNotNone(model)
        self.assertEqual(len(df_pred), 100)
        self.assertEqual(len(future_df), future_days)
        self.assertIn('Prediction', df_pred.columns)
        self.assertIsInstance(r2, float)
        self.assertIsInstance(mse, float)

    def test_prediction_logic(self):
        """Test that predictions are reasonable for a perfect line."""
        # Create perfect linear data
        dates = [date(2023, 1, 1) + timedelta(days=x) for x in range(10)]
        # Price increases by 10 every day: 10, 20, 30...
        prices = [10 * (i+1) for i in range(10)]
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'Close': prices
        })
        
        model, _, future_df, r2, mse = train_model(df, future_days=1)
        
        # Next day (11th day) should be 110
        next_pred = future_df['Prediction'].iloc[0]
        self.assertAlmostEqual(next_pred, 110.0, places=5)
        self.assertAlmostEqual(r2, 1.0, places=5)
        self.assertAlmostEqual(mse, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
