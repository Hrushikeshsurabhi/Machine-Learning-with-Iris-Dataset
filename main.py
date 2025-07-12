#!/usr/bin/env python3
"""
Main Script for Iris Dataset Machine Learning Project
====================================================

This script demonstrates a complete machine learning pipeline for the Iris dataset,
including data loading, preprocessing, model training, evaluation, and visualization.

Author: Hrushikeshsurabhi (Forked from venky14)
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_iris_data, preprocess_data, get_dataset_info
from src.models import IrisClassifier, create_model_comparison_report
from src.visualization import create_summary_plots, plot_confusion_matrix
import pandas as pd


def main():
    """
    Main function to run the complete machine learning pipeline.
    """
    print("=" * 60)
    print("IRIS DATASET MACHINE LEARNING PROJECT")
    print("=" * 60)
    print("Forked from venky14/Machine-Learning-with-Iris-Dataset")
    print("Enhanced with modular structure and comprehensive analysis")
    print("=" * 60)
    
    # Step 1: Load Data
    print("\n1. LOADING DATA")
    print("-" * 30)
    df = load_iris_data()
    
    # Display dataset information
    info = get_dataset_info(df)
    print(f"\nDataset Shape: {info['shape']}")
    print(f"Features: {info['columns']}")
    print(f"Species Distribution:")
    for species, count in info['species_distribution'].items():
        print(f"  {species}: {count} samples")
    
    # Step 2: Data Preprocessing
    print("\n2. DATA PREPROCESSING")
    print("-" * 30)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Step 3: Model Training
    print("\n3. MODEL TRAINING")
    print("-" * 30)
    classifier = IrisClassifier()
    results = classifier.train_all_models(X_train, y_train, cv=5)
    
    # Display model comparison
    print("\nModel Comparison Summary:")
    comparison_df = create_model_comparison_report(results)
    print(comparison_df.to_string(index=False))
    
    # Step 4: Train Best Model
    print("\n4. TRAINING BEST MODEL")
    print("-" * 30)
    classifier.train_best_model(X_train, y_train)
    
    # Step 5: Model Evaluation
    print("\n5. MODEL EVALUATION")
    print("-" * 30)
    evaluation_results = classifier.evaluate_model(X_test, y_test)
    
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])
    
    # Step 6: Visualization
    print("\n6. CREATING VISUALIZATIONS")
    print("-" * 30)
    
    # Get predictions for visualization
    y_pred = classifier.best_model.predict(X_test)
    
    # Create comprehensive visualizations
    create_summary_plots(
        df=df,
        results=results,
        y_true=y_test,
        y_pred=y_pred,
        model=classifier.best_model,
        feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    )
    
    # Step 7: Save Model
    print("\n7. SAVING MODEL")
    print("-" * 30)
    classifier.save_model()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best Model: {classifier.best_model_name}")
    print(f"Best CV Score: {classifier.best_score:.4f}")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main() 