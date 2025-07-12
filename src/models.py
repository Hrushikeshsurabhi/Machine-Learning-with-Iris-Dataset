"""
Machine Learning Models Module for Iris Dataset
===============================================

This module provides various machine learning models and evaluation functions
for the Iris classification problem.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import os


class IrisClassifier:
    """
    A comprehensive classifier for the Iris dataset with multiple algorithms.
    """
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=3),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        
    def train_all_models(self, X_train, y_train, cv=5):
        """
        Train all models and find the best performing one.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Dictionary with model names and their cross-validation scores
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            scores = cross_val_score(model, X_train, y_train, cv=cv)
            mean_score = scores.mean()
            std_score = scores.std()
            
            results[name] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            }
            
            print(f"{name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
            
            # Update best model
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with score: {self.best_score:.4f}")
        return results
    
    def train_best_model(self, X_train, y_train):
        """
        Train the best performing model on the full training set.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run train_all_models first.")
        
        print(f"Training best model ({self.best_model_name}) on full dataset...")
        self.best_model.fit(X_train, y_train)
        print("Training completed!")
    
    def evaluate_model(self, X_test, y_test, model=None):
        """
        Evaluate a model on the test set.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        model : sklearn estimator, optional
            Model to evaluate. If None, uses the best model.
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if model is None:
            model = self.best_model
            
        if model is None:
            raise ValueError("No model available for evaluation.")
        
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return results
    
    def save_model(self, filepath='../models/best_iris_model.pkl'):
        """
        Save the best model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path where to save the model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='../models/best_iris_model.pkl'):
        """
        Load a saved model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        if os.path.exists(filepath):
            self.best_model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")


def create_model_comparison_report(results):
    """
    Create a comparison report of all models.
    
    Parameters:
    -----------
    results : dict
        Results from train_all_models
        
    Returns:
    --------
    pandas.DataFrame
        Comparison table
    """
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Mean CV Score': f"{result['mean_score']:.4f}",
            'Std CV Score': f"{result['std_score']:.4f}",
            'Score Range': f"{result['mean_score'] - result['std_score']:.4f} - {result['mean_score'] + result['std_score']:.4f}"
        })
    
    return pd.DataFrame(comparison_data) 