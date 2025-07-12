"""
Data Loader Module for Iris Dataset
===================================

This module provides functions to load and preprocess the Iris dataset.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_data(file_path='../data/Iris.csv'):
    """
    Load Iris dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the Iris.csv file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded Iris dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {len(df)} samples and {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Loading from sklearn.datasets instead...")
        return load_iris_sklearn()


def load_iris_sklearn():
    """
    Load Iris dataset from sklearn.datasets.
    
    Returns:
    --------
    pandas.DataFrame
        Loaded Iris dataset
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target_names[iris.target]
    print(f"Dataset loaded from sklearn with {len(df)} samples and {len(df.columns)} features")
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the Iris dataset for machine learning.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    test_size : float
        Proportion of dataset to include in the test split
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data preprocessed: Train set {X_train.shape[0]} samples, Test set {X_test.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_dataset_info(df):
    """
    Get basic information about the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
        
    Returns:
    --------
    dict
        Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'species_distribution': df['Species'].value_counts().to_dict() if 'Species' in df.columns else None
    }
    return info 