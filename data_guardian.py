import pandas as pd
import numpy as np

def validate_and_clean_feedback(feedback_file_path):
    """
    Automated cleaning firewall to filter out malicious spam or typos
    from public user feedback inputs.
    """
    try:
        # Load feedback data columns matching api.py layout
        # Columns: User, Town, State, Predicted_Naira, Rating, Comments
        df = pd.read_csv(feedback_file_path, names=["User", "Town", "State", "Predicted_Naira", "Rating", "Comments"])
        if len(df) == 0:
            return pd.DataFrame()
            
        # 1. Strip structural formatting errors
        df['State'] = df['State'].astype(str).str.strip().str.title()
        df['Town'] = df['Town'].astype(str).str.strip().str.title()
        
        # 2. Extract the numeric price from the Comments column
        df['Actual_Sold_Price'] = pd.to_numeric(df['Comments'].astype(str).str.replace(r'[^0-9]', '', regex=True), errors='coerce')
        
        # Drop rows where no number could be extracted from comments
        df = df.dropna(subset=['Actual_Sold_Price'])
        
        # 3. Filter out extreme pricing anomalies (Keep between 2 Million and 1.5 Billion Naira)
        df = df[(df['Actual_Sold_Price'] >= 2_000_000) & (df['Actual_Sold_Price'] <= 1_500_000_000)]
        
        # 4. ANTI-DATA POISONING SHIELD
        if len(df) > 10:
            q1 = df['Actual_Sold_Price'].quantile(0.25)
            q3 = df['Actual_Sold_Price'].quantile(0.75)
            iqr = q3 - q1
            df = df[(df['Actual_Sold_Price'] >= q1 - 1.5 * iqr) & (df['Actual_Sold_Price'] <= q3 + 1.5 * iqr)]
            
        return df
    except Exception as e:
        print(f"Firewall error: {e}")
        return pd.DataFrame()
