import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

def train_model(df: pd.DataFrame, future_days: int):
    """
    Trains a Linear Regression model on the provided dataframe.
    
    Args:
        df (pd.DataFrame): Historical stock data.
        future_days (int): Number of days to predict into the future.
        
    Returns:
        tuple: (model, df_with_predictions, future_df, r2, mse)
    """
    logger.info(f"Training model with {len(df)} records, predicting {future_days} days out.")
    
    try:
        data = df.copy()
        
        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
             data['Date'] = pd.to_datetime(data['Date'])
        
        # Create an ordinal date column for regression
        data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
        
        X = data[['Date_Ordinal']]
        y = data['Close']
        
        # Train Model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict on training data (regression line)
        data['Prediction'] = model.predict(X)
        
        # Calculate Metrics
        r2 = model.score(X, y)
        mse = np.mean((data['Prediction'] - y) ** 2)
        
        logger.info(f"Model trained. R2: {r2:.4f}, MSE: {mse:.4f}")
        
        # Future Predictions
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, future_days + 1)]
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        
        # Use DataFrame for prediction to match training feature names and silence warnings
        future_X = pd.DataFrame(future_ordinals, columns=['Date_Ordinal'])
        future_preds = model.predict(future_X)
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Prediction': future_preds
        })
        
        return model, data, future_df, r2, mse
        
    except Exception as e:
        logger.error(f"Error checking model training: {e}", exc_info=True)
        raise e
