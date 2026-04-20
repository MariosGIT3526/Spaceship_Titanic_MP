# src/preprocessor.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Defined at module level so joblib can pickle it
def cast_to_str(x):
    return x.astype(str)

class SpaceshipPreprocessor:
    """
    Reusable preprocessing pipeline for binary classification projects.
    Fits on training data only and transforms all splits consistently.
    """
    
    def __init__(self, num_cols, cat_cols):
        """Store column configuration and initialize pipeline as None."""
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.preprocessor = None      # will be set after fit
        self.feature_names_ = None    # will be set after fit
    
    def _build_pipeline(self):
        """Build and return the ColumnTransformer pipeline."""

        numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('to_str', FunctionTransformer(cast_to_str, feature_names_out='one-to-one')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(transformers=[
        ('num', numeric_pipeline, self.num_cols),
        ('cat', categorical_pipeline, self.cat_cols)
        ], remainder='drop')
    
    def fit_transform(self, X_train, X_val, X_test):
        """
        Fit pipeline on X_train only.
        Transform X_train, X_val, and X_test.
        Returns tuple: (X_train_p, X_val_p, X_test_p)
        """
        # Build and fit the pipeline
        self.preprocessor = self._build_pipeline()
        
        # fit_transform on train, transform only on val and test
        X_train_p = self.preprocessor.fit_transform(X_train)
        X_val_p = self.preprocessor.transform(X_val) 
        X_test_p = self.preprocessor.transform(X_test)
        
        # Build feature names
        self.feature_names_ = self._get_feature_names()
        
        return X_train_p, X_val_p, X_test_p
    
    def _get_feature_names(self):
        """Build readable feature names after fitting."""
        ohe = self.preprocessor.named_transformers_['cat']['encoder']
        cat_feature_names = [f"{col}_{val}" for col, cats in 
                             zip(self.cat_cols, ohe.categories_) 
                             for val in cats[1:]]
        return np.array(self.num_cols + cat_feature_names)
    
    def summary(self, X_train_p, X_val_p, X_test_p):
        """Print shapes, NaN counts, and feature names."""
        print(f"X_train shape: {X_train_p.shape}")
        print(f"X_val shape:   {X_val_p.shape}")
        print(f"X_test shape:  {X_test_p.shape}")
        print(f"\nNaN in X_train: {np.isnan(X_train_p).sum()}")
        print(f"NaN in X_val:   {np.isnan(X_val_p).sum()}")
        print(f"\nTotal features: {len(self.feature_names_)}")
        print(f"Feature names:\n{self.feature_names_}")