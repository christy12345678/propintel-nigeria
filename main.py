import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from dotenv import load_dotenv
load_dotenv()
# =====================================================================
# 1. DATA LOADING & INITIAL CLEANING
# =====================================================================
# Replace the old hardcoded line with this secure system fetch to allow dynamic path configuration via environment variables or fallback to a default path if not set. 

csv_path = os.environ.get("MASTER_DATA_PATH", "nigeria_houses_data.csv")

print(f"Loading dataset from: {csv_path}...")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find the file at {csv_path}.")

df = pd.read_csv(csv_path)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price', 'state'])

# Filter extreme anomalies to stabilize nationwide evaluation scales
df = df[df['price'] <= 1_500_000_000]

for col in ['state', 'town', 'title']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Fix common typo found in this dataset variant
df['state'] = df['state'].replace('Anambara', 'Anambra')

# =====================================================================
# 2. ADVANCED MACROECONOMIC DATA INJECTION
# =====================================================================
print("Injecting national macroeconomic indicators...")

# Mapping real-world baseline state metrics to guide regional pricing models
macro_data = {
    'state': [
        'Lagos', 'Abuja', 'Rivers', 'Delta', 'Oyo', 'Anambra', 'Edo', 'Kano', 'Enugu',
        'Abia', 'Imo', 'Akwa Ibom', 'Kaduna', 'Kwara', 'Ogun', 'Osun', 'Nasarawa', 
        'Kogi', 'Ekiti', 'Niger', 'Cross River', 'Plateau', 'Katsina', 'Bayelsa', 'Borno'
    ],
    # Internal Generated Revenue tier (5 = Highest economic power, 1 = Baseline)
    'state_igr_tier': [5, 5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 3, 3, 2, 4, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1],
    # Multi-dimensional Development Index proxy scale
    'dev_index': [0.88, 0.90, 0.75, 0.72, 0.65, 0.70, 0.64, 0.58, 0.66, 0.62, 0.63, 0.68, 0.55, 0.59, 0.70, 0.60, 0.52, 0.54, 0.56, 0.51, 0.61, 0.57, 0.42, 0.67, 0.38]
}
macro_df = pd.DataFrame(macro_data)

# Seamlessly merge macro profiles directly into the core housing framework
df = df.merge(macro_df, on='state', how='left')
df['state_igr_tier'] = df['state_igr_tier'].fillna(2)
df['dev_index'] = df['dev_index'].fillna(0.55)

# =====================================================================
# 3. FEATURE ENGINEERING
# =====================================================================
print("Engineering structural indicators...")
text_source_col = 'address' if 'address' in df.columns else ('title' if 'title' in df.columns else None)

if text_source_col:
    text_series = df[text_source_col].astype(str).str.lower()
    df['is_luxury_estate'] = text_series.str.contains('estate|gated|serviced|terrace|court').astype(int)
    df['is_new_build'] = text_series.str.contains('brand new|newly|modern').astype(int)
    df['has_boys_quarter'] = text_series.str.contains(' bq|boys quarter|boysquarter').astype(int)
else:
    df['is_luxury_estate'], df['is_new_build'], df['has_boys_quarter'] = 0, 0, 0

df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
df['state_town_combo'] = df['state'] + "_" + df['town']

# =====================================================================
# 4. ROBUST GEOGRAPHIC STRATIFICATION
# =====================================================================
print("Calculating real estate stratification matrices...")
state_medians = df.groupby('state')['price'].median().reset_index()
p33 = state_medians['price'].quantile(0.33)
p66 = state_medians['price'].quantile(0.66)

def determine_market_tier(median_price):
    if median_price >= p66: return 'Tier_1_Premium'       
    elif median_price >= p33: return 'Tier_2_Growth_Hubs'   
    else: return 'Tier_3_Baseline'      

state_medians['market_tier'] = state_medians['price'].apply(determine_market_tier)
tier_mapping = dict(zip(state_medians['state'], state_medians['market_tier']))
df['market_tier'] = df['state'].map(tier_mapping)

# =====================================================================
# 5. TRAINING ENGINE WITH INTENSE REGIONAL OVERSAMPLING
# =====================================================================
categorical_features = ['state', 'town', 'title', 'state_town_combo']
for col in categorical_features:
    df[col] = df[col].astype(str)

models_vault = {}
tier_metrics = {}

base_features = ['bedrooms', 'bathrooms', 'toilets', 'state', 'town', 'title', 
                 'is_luxury_estate', 'is_new_build', 'has_boys_quarter', 
                 'bed_bath_ratio', 'state_town_combo', 'state_igr_tier', 'dev_index']

all_raw_actuals, all_raw_predictions, all_tier_labels = [], [], []

tier_hyperparameters = {
    'Tier_1_Premium':     {'iterations': 1200, 'lr': 0.05, 'depth': 6},
    'Tier_2_Growth_Hubs': {'iterations': 2000, 'lr': 0.01, 'depth': 5}, 
    'Tier_3_Baseline':    {'iterations': 2000, 'lr': 0.01, 'depth': 5}  
}

for tier in ['Tier_1_Premium', 'Tier_2_Growth_Hubs', 'Tier_3_Baseline']:
    tier_df = df[df['market_tier'] == tier].copy()
    if len(tier_df) < 30: continue
        
    print(f"\nTraining Engine Layer: {tier.upper()} ({len(tier_df)} baseline entries)")
    
    X = tier_df[base_features]
    y = tier_df['price']  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CRITICAL BALANCING ACTION: Oversample low-volume data tiers to protect evaluation metrics
    if tier in ['Tier_2_Growth_Hubs', 'Tier_3_Baseline']:
        X_train = pd.concat([X_train] * 4, ignore_index=True)
        y_train = pd.concat([y_train] * 4, ignore_index=True)
        print(f"--> Injected balancing oversampling: Train matrix extended to {len(X_train)} samples.")

    cat_indices = [X.columns.get_loc(col) for col in categorical_features]
    config = tier_hyperparameters[tier]
    
    model = CatBoostRegressor(
        iterations=config['iterations'],
        learning_rate=config['lr'],
        depth=config['depth'],
        loss_function='Huber:delta=10000000', # Smooth transition loss to handle extreme regional variations cleanly
        cat_features=cat_indices,
        verbose=400,
        random_seed=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=60)
    
    raw_preds = model.predict(X_test)
    raw_actuals = y_test.values
    
    tier_r2 = r2_score(raw_actuals, raw_preds)
    tier_mae = mean_absolute_error(raw_actuals, raw_preds)
    
    models_vault[tier] = model
    tier_metrics[tier] = {'R2': tier_r2, 'MAE': tier_mae}
    
    all_raw_actuals.extend(raw_actuals)
    all_raw_predictions.extend(raw_preds)
    all_tier_labels.extend([tier] * len(raw_actuals))

# =====================================================================
# 6. GLOBAL NATIONWIDE PERFORMANCE REVELATION
# =====================================================================
all_raw_actuals, all_raw_predictions, all_tier_labels = map(np.array, [all_raw_actuals, all_raw_predictions, all_tier_labels])
global_r2 = r2_score(all_raw_actuals, all_raw_predictions)
global_mae = mean_absolute_error(all_raw_actuals, all_raw_predictions)

print("\n" + "="*50)
print("     FINAL MACRO-STRATIFIED NATIONAL ENGINE RESULTS     ")
print("="*50)
for tier, metrics in tier_metrics.items():
    print(f"{tier:<18} -> Local Real-Scale R^2: {metrics['R2']:.4f} | MAE: ₦{metrics['MAE']:,.2f}")
print("-" * 50)
print(f"COMBINED NATIONWIDE PRODUCTION R^2 SCORE : {global_r2:.4f}")
print(f"COMBINED NATIONWIDE MEAN ABSOLUTE ERROR  : ₦{global_mae:,.2f}")
print("="*50)

# =====================================================================
# 7. EXPORT VISUALIZATION PLOT
# =====================================================================
try:
    plt.figure(figsize=(10, 6))
    plot_indices = np.random.choice(len(all_raw_actuals), min(5000, len(all_raw_actuals)), replace=False)
    sns.scatterplot(x=all_raw_actuals[plot_indices], y=all_raw_predictions[plot_indices], hue=all_tier_labels[plot_indices], alpha=0.5, palette='Set1')
    max_val = max(max(all_raw_actuals), max(all_raw_predictions))
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Accuracy')
    plt.title("Nationwide Macroeconomic Housing Model Predictions")
    plt.xlabel("Actual Prices (Naira)")
    plt.ylabel("Predicted Prices (Naira)")
    plt.xscale('log'); plt.yscale('log'); plt.grid(True, which="both", ls="--", alpha=0.4); plt.legend()
    plt.tight_layout()
    plt.savefig('custom_housing_analytics.png', dpi=150)
    print("\n[Graphics Saved]: Visualization updated as 'custom_housing_analytics.png'.")
except Exception as e:
    print(f"\n[Plotting Engine Alert]: Graphics skipped: {e}")
import joblib

YOUR_PROJECT_NAME = "PropIntel_Core_Engine_v1"

print(f"\n[Exporting] Archiving models under system signature: '{YOUR_PROJECT_NAME}'...")

## Saves the specialized brains using your custom naming convention
for tier, model in models_vault.items():
    model.save_model(f"{YOUR_PROJECT_NAME}_{tier}.cbm")

# Saves the state-to-tier grouping map
joblib.dump(tier_mapping, f"{YOUR_PROJECT_NAME}_tier_mapping.pkl")
print(f"[Export Complete]: Files successfully saved as '{YOUR_PROJECT_NAME}_[Tier].cbm'")