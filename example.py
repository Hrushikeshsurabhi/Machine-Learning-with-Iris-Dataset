#!/usr/bin/env python3
"""
Simple Example Script for Iris Dataset Analysis
==============================================

This script demonstrates basic usage of the modular components
for quick analysis and experimentation.

Author: Hrushikeshsurabhi (Forked from venky14)
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_iris_data, preprocess_data
from src.models import IrisClassifier
from src.visualization import plot_data_distribution, plot_correlation_matrix


def simple_example():
    """
    Simple example demonstrating basic functionality.
    """
    print("🌸 Simple Iris Dataset Analysis Example")
    print("=" * 50)
    
    # 1. Load data
    print("\n1. Loading Iris dataset...")
    df = load_iris_data()
    print(f"   Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # 2. Quick visualization
    print("\n2. Creating basic visualizations...")
    plot_data_distribution(df)
    plot_correlation_matrix(df)
    
    # 3. Train a simple model
    print("\n3. Training models...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    classifier = IrisClassifier()
    results = classifier.train_all_models(X_train, y_train, cv=3)
    
    # 4. Show results
    print("\n4. Results Summary:")
    print(f"   Best model: {classifier.best_model_name}")
    print(f"   Best CV score: {classifier.best_score:.4f}")
    
    # 5. Evaluate on test set
    classifier.train_best_model(X_train, y_train)
    evaluation = classifier.evaluate_model(X_test, y_test)
    print(f"   Test accuracy: {evaluation['accuracy']:.4f}")
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    simple_example() 