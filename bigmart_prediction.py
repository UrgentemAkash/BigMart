import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    """Load and preprocess the data with advanced techniques"""
    print("Loading data...")
    # Load the datasets
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Create a copy of the datasets
    train_df = train.copy()
    test_df = test.copy()
    
    # Add a dataset indicator column
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    
    # Add target column to test data with NaN values
    test_df['Item_Outlet_Sales'] = np.nan
    
    # Combine datasets for preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print("Advanced preprocessing...")
    # Standardize Item_Fat_Content values
    combined_df['Item_Fat_Content'] = combined_df['Item_Fat_Content'].replace(
        {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
    )
    
    # Handling missing values
    
    # For Item_Weight: Use median for each Item_Type to be more robust
    item_type_weight_median = combined_df.groupby(['Item_Type'])['Item_Weight'].median()
    for idx in combined_df[combined_df['Item_Weight'].isna()].index:
        item_type = combined_df.at[idx, 'Item_Type']
        combined_df.at[idx, 'Item_Weight'] = item_type_weight_median[item_type]
    
    # For Outlet_Size: Use a more sophisticated approach combining multiple outlet attributes
    outlet_attributes = combined_df.groupby(['Outlet_Type', 'Outlet_Location_Type'])['Outlet_Size'].apply(
        lambda x: x.mode()[0] if not x.mode().empty else 'Medium'
    )
    
    for idx in combined_df[combined_df['Outlet_Size'].isna()].index:
        outlet_type = combined_df.at[idx, 'Outlet_Type']
        outlet_location = combined_df.at[idx, 'Outlet_Location_Type']
        try:
            combined_df.at[idx, 'Outlet_Size'] = outlet_attributes[(outlet_type, outlet_location)]
        except:
            # If combination doesn't exist in training data, use outlet type only
            combined_df.at[idx, 'Outlet_Size'] = combined_df[combined_df['Outlet_Type'] == outlet_type]['Outlet_Size'].mode()[0]
    
    # Enhanced Feature Engineering
    
    # 1. Create a feature for Item_Visibility
    # Replace 0 visibility with mean visibility of that product + a small random jitter to avoid identical values
    zero_visibility_indices = combined_df['Item_Visibility'] == 0
    item_visibility_mean = combined_df.groupby(['Item_Identifier', 'Item_Type'])['Item_Visibility'].transform('mean')
    
    for idx in combined_df[zero_visibility_indices].index:
        mean_value = item_visibility_mean.iloc[idx]
        # Add a small random jitter (1% of the mean)
        combined_df.at[idx, 'Item_Visibility'] = mean_value * (1 + np.random.normal(0, 0.01))
    
    # 2. Create Item_Identifier categories with more granularity
    combined_df['Item_Category'] = combined_df['Item_Identifier'].apply(lambda x: x[:2])
    
    # Map FDA, FDW, etc. to actual categories 
    # Fix: Use label encoding instead of direct mapping to avoid string values in numeric columns
    category_mapping = {
        'FD': 0,  # Food
        'DR': 1,  # Drinks
        'NC': 2   # Non-Consumable
    }
    combined_df['Item_Category_Encoded'] = combined_df['Item_Category'].map(category_mapping)
    
    # Create subcategories based on the third character
    # Fix: Convert subcategory to numeric using label encoding
    label_encoder = LabelEncoder()
    combined_df['Item_Subcategory_Encoded'] = label_encoder.fit_transform(combined_df['Item_Identifier'].apply(lambda x: x[2]))
    
    # 3. Add more features related to outlet
    # Outlet age (years of operation)
    combined_df['Outlet_Years'] = 2013 - combined_df['Outlet_Establishment_Year']
    
    # Group outlets into age bins and convert to numeric - fixing categorical issue by converting to int
    age_bins = [0, 10, 20, 30]
    age_labels = [0, 1, 2]
    combined_df['Outlet_Age_Binned'] = pd.cut(
        combined_df['Outlet_Years'], 
        bins=age_bins,
        labels=age_labels
    )
    # Convert to integer explicitly to avoid category type
    combined_df['Outlet_Age_Binned'] = combined_df['Outlet_Age_Binned'].astype(int)
    
    # 4. Advanced item features
    # Normalized Item_Visibility within Item_Type
    combined_df['Item_Visibility_Normalized'] = combined_df.groupby('Item_Type')['Item_Visibility'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )
    
    # Price tier based on MRP - convert to numeric and fix category issue
    price_bins = pd.qcut(combined_df['Item_MRP'], q=4, labels=False)  # Use False for numeric labels
    combined_df['Price_Tier'] = price_bins.astype(int)  # Explicitly convert to int
    
    # Item MRP to outlet type ratio - capture price sensitivity by outlet type
    combined_df['MRP_Outlet_Type_Ratio'] = combined_df.groupby('Outlet_Type')['Item_MRP'].transform(
        lambda x: x / x.mean()
    )
    
    # 5. Implement additional suggested features
    
    # Visibility_MeanRatio = Item_Visibility / mean visibility per Item_Identifier
    item_vis_mean = combined_df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    combined_df['Visibility_MeanRatio'] = combined_df['Item_Visibility'] / item_vis_mean
    # Handle division by zero
    combined_df['Visibility_MeanRatio'].replace([np.inf, -np.inf], 1.0, inplace=True)
    combined_df['Visibility_MeanRatio'].fillna(1.0, inplace=True)
    
    # Create MRP bins (already implemented as Price_Tier)
    # Add more granular MRP bins for finer segmentation
    combined_df['MRP_Bins_10'] = pd.qcut(combined_df['Item_MRP'], q=10, labels=False).astype(int)
    
    # Log transform of target variable to address skewness (if needed)
    if 'Item_Outlet_Sales' in combined_df.columns and combined_df['source'].eq('train').any():
        # Only apply to training data where target is not NaN
        train_mask = combined_df['source'] == 'train'
        combined_df.loc[train_mask, 'Item_Outlet_Sales_Log'] = np.log1p(combined_df.loc[train_mask, 'Item_Outlet_Sales'])
    
    # 6. Additional interaction features
    # Calculate log of Item_MRP for better distribution
    combined_df['Item_MRP_Log'] = np.log1p(combined_df['Item_MRP'])
    
    # Create interaction between Item Category and Outlet Type as a numeric feature
    combined_df['Cat_Outlet_Interaction'] = combined_df['Item_Category_Encoded'] * combined_df['Outlet_Years']
    
    # Create interaction between Item fat content and Item type
    fat_content_map = {'Low Fat': 0, 'Regular': 1}
    combined_df['Item_Fat_Encoded'] = combined_df['Item_Fat_Content'].map(fat_content_map)
    combined_df['Fat_Category_Interaction'] = combined_df['Item_Fat_Encoded'] * combined_df['Item_Category_Encoded']
    
    # Create ratio of Item_Weight to Item_Visibility
    combined_df['Weight_Visibility_Ratio'] = combined_df['Item_Weight'] / (combined_df['Item_Visibility'] + 0.001)
    
    # Encode categorical variables
    
    # Label encoding for ordinal features
    ordinal_features = ['Outlet_Size', 'Outlet_Location_Type']
    
    for feature in ordinal_features:
        combined_df[feature + '_Encoded'] = label_encoder.fit_transform(combined_df[feature])
    
    # One-hot encoding for nominal categorical features
    nominal_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type']
    combined_df = pd.get_dummies(combined_df, columns=nominal_features, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numeric_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 
                         'Outlet_Years', 'Item_Visibility_Normalized', 'MRP_Outlet_Type_Ratio', 
                         'Item_MRP_Log', 'Visibility_MeanRatio', 'Weight_Visibility_Ratio']
    
    combined_df[numeric_features] = scaler.fit_transform(combined_df[numeric_features])
    
    # Prepare the final datasets for modeling
    train_final = combined_df[combined_df['source'] == 'train'].drop('source', axis=1)
    test_final = combined_df[combined_df['source'] == 'test'].drop(['source', 'Item_Outlet_Sales'], axis=1)
    
    # Remove Item_Outlet_Sales_Log from test_final if it exists
    if 'Item_Outlet_Sales_Log' in test_final.columns:
        test_final = test_final.drop(['Item_Outlet_Sales_Log'], axis=1)
    
    # Select features for modeling
    drop_columns = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year',
                   'Outlet_Size', 'Outlet_Location_Type', 'Item_Category', 'Item_Fat_Encoded']
    
    if 'Item_Outlet_Sales_Log' in train_final.columns:
        # Use the log-transformed target if it was created
        X_train = train_final.drop(drop_columns + ['Item_Outlet_Sales', 'Item_Outlet_Sales_Log'], axis=1)
        y_train = train_final['Item_Outlet_Sales_Log']  # Use log-transformed target
        use_log_target = True
    else:
        X_train = train_final.drop(drop_columns + ['Item_Outlet_Sales'], axis=1)
        y_train = train_final['Item_Outlet_Sales']
        use_log_target = False
    
    X_test = test_final.drop(drop_columns, axis=1)
    
    # Print feature info for debugging
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Using log-transformed target: {use_log_target}")
    
    return X_train, y_train, X_test, test_final['Item_Identifier'], test_final['Outlet_Identifier'], use_log_target

def create_ensemble_model(X_train, y_train):
    """Build an advanced ensemble model combining multiple regressors"""
    print("Building ensemble model...")
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Base models with carefully tuned hyperparameters - sticking with stable models first
    base_models = [
        ('ridge', Ridge(alpha=0.5, random_state=42)),
        ('lasso', Lasso(alpha=0.001, random_state=42)),
        ('en', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42)),
        ('gbr', GradientBoostingRegressor(learning_rate=0.05, n_estimators=200, max_depth=4, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=200, max_features='sqrt', max_depth=15, random_state=42))
    ]
    
    # Try to add XGBoost and LightGBM if evaluation succeeds
    try:
        xgb_model = XGBRegressor(learning_rate=0.05, n_estimators=300, max_depth=5, colsample_bytree=0.7, random_state=42)
        xgb_model.fit(X_train_split, y_train_split)
        xgb_pred = xgb_model.predict(X_val)
        xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
        print(f"XGB test: RMSE = {xgb_rmse:.4f}")
        base_models.append(('xgb', xgb_model))
    except Exception as e:
        print(f"XGBoost model failed: {e}")
    
    try:
        lgb_model = LGBMRegressor(learning_rate=0.05, n_estimators=300, num_leaves=31, random_state=42)
        lgb_model.fit(X_train_split, y_train_split)
        lgb_pred = lgb_model.predict(X_val)
        lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
        print(f"LGBM test: RMSE = {lgb_rmse:.4f}")
        base_models.append(('lgbm', lgb_model))
    except Exception as e:
        print(f"LightGBM model failed: {e}")
    
    # Evaluate individual models
    print("Evaluating individual models:")
    model_scores = {}
    for name, model in base_models:
        # Fit the model
        model.fit(X_train_split, y_train_split)
        # Make predictions
        val_pred = model.predict(X_val)
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        model_scores[name] = rmse
        print(f"{name.upper()}: RMSE = {rmse:.4f}")
    
    # Create a voting ensemble with the best models
    best_models = sorted(model_scores.items(), key=lambda x: x[1])[:4]
    print(f"Best models: {best_models}")
    
    # Extract the best models and assign weights (higher weight for better models)
    voting_estimators = []
    voting_weights = []
    for i, (name, score) in enumerate(best_models):
        weight = 1 + (len(best_models) - i) * 0.5  # Higher weight for better models
        for model_name, model in base_models:
            if model_name == name:
                voting_estimators.append((name, model))
                voting_weights.append(weight)
                break
    
    voting_regressor = VotingRegressor(
        estimators=voting_estimators,
        weights=voting_weights
    )
    
    # Create a stacking ensemble with the same models
    stacking_regressor = StackingRegressor(
        estimators=voting_estimators,
        final_estimator=Ridge(alpha=0.5, random_state=42),
        cv=5
    )
    
    # Compare voting and stacking
    print("\nEvaluating ensemble methods:")
    
    # Train and evaluate voting regressor
    voting_regressor.fit(X_train_split, y_train_split)
    voting_pred = voting_regressor.predict(X_val)
    voting_rmse = np.sqrt(mean_squared_error(y_val, voting_pred))
    print(f"Voting Ensemble RMSE: {voting_rmse:.4f}")
    
    # Train and evaluate stacking regressor
    stacking_regressor.fit(X_train_split, y_train_split)
    stacking_pred = stacking_regressor.predict(X_val)
    stacking_rmse = np.sqrt(mean_squared_error(y_val, stacking_pred))
    print(f"Stacking Ensemble RMSE: {stacking_rmse:.4f}")
    
    # Choose the best ensemble method
    if stacking_rmse < voting_rmse:
        print("Stacking ensemble is better. Training on full dataset...")
        best_model = stacking_regressor
        best_model.fit(X_train, y_train)
        print("Stacking ensemble trained successfully!")
    else:
        print("Voting ensemble is better. Training on full dataset...")
        best_model = voting_regressor
        best_model.fit(X_train, y_train)
        print("Voting ensemble trained successfully!")
    
    return best_model

def make_prediction(model, X_test, item_ids, outlet_ids, use_log_target=False):
    """Make predictions and create submission file"""
    print("Making predictions...")
    
    # Ensure Item_Outlet_Sales_Log isn't in X_test
    if 'Item_Outlet_Sales_Log' in X_test.columns:
        X_test = X_test.drop(['Item_Outlet_Sales_Log'], axis=1)
    
    # Predict on test data
    test_predictions = model.predict(X_test)
    
    # If we used log transform on the target, transform predictions back
    if use_log_target:
        test_predictions = np.expm1(test_predictions)
    
    # Ensure predictions are positive (sales can't be negative)
    test_predictions = np.maximum(0, test_predictions)
    
    # Create submission file
    submission = pd.DataFrame({
        'Item_Identifier': item_ids,
        'Outlet_Identifier': outlet_ids,
        'Item_Outlet_Sales': test_predictions
    })
    
    # Save submission file
    submission.to_csv('ensemble_submission_enhanced.csv', index=False)
    print("Submission file created: ensemble_submission_enhanced.csv")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    with open('models/ensemble_model_enhanced.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved: models/ensemble_model_enhanced.pkl")
    
    return submission

def main():
    """Main function to run the entire process"""
    print("BigMart Sales Prediction - Enhanced Ensemble Solution")
    print("---------------------------------------------------")
    
    # Prepare data with advanced feature engineering
    X_train, y_train, X_test, item_ids, outlet_ids, use_log_target = prepare_data()
    
    # Create and train ensemble model
    best_model = create_ensemble_model(X_train, y_train)
    
    # Make predictions and create submission
    submission = make_prediction(best_model, X_test, item_ids, outlet_ids, use_log_target)
    
    print("Done!")

if __name__ == "__main__":
    main() 