"""
Visualization Module for Iris Dataset
=====================================

This module provides comprehensive visualization functions for data exploration
and model results analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_data_distribution(df, figsize=(15, 10)):
    """
    Plot distribution of features by species.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Iris dataset
    figsize : tuple
        Figure size
    """
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Feature Distributions by Species', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2
        
        for species in df['Species'].unique():
            species_data = df[df['Species'] == species][feature]
            axes[row, col].hist(species_data, alpha=0.7, label=species, bins=15)
        
        axes[row, col].set_title(f'{feature} Distribution')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(10, 8)):
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Iris dataset
    figsize : tuple
        Figure size
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_pairplot(df, hue='Species', figsize=(12, 10)):
    """
    Create a pairplot for feature relationships.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Iris dataset
    hue : str
        Column to use for color coding
    figsize : tuple
        Figure size
    """
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    plt.figure(figsize=figsize)
    sns.pairplot(df[features + [hue]], hue=hue, diag_kind='hist', 
                 plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.7})
    plt.suptitle('Feature Relationships by Species', y=1.02, fontsize=16, fontweight='bold')
    plt.show()


def plot_boxplots(df, figsize=(15, 8)):
    """
    Create boxplots for feature distributions by species.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Iris dataset
    figsize : tuple
        Figure size
    """
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle('Feature Distributions by Species (Boxplots)', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(features):
        sns.boxplot(data=df, x='Species', y=feature, ax=axes[i])
        axes[i].set_title(f'{feature} by Species')
        axes[i].set_xlabel('Species')
        axes[i].set_ylabel(feature)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of the classes
    figsize : tuple
        Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results, figsize=(12, 6)):
    """
    Plot model comparison results.
    
    Parameters:
    -----------
    results : dict
        Results from model training
    figsize : tuple
        Figure size
    """
    model_names = list(results.keys())
    mean_scores = [results[name]['mean_score'] for name in model_names]
    std_scores = [results[name]['std_score'] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot of mean scores
    bars = ax1.bar(model_names, mean_scores, yerr=std_scores, 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_title('Model Performance Comparison', fontweight='bold')
    ax1.set_ylabel('Cross-Validation Score')
    ax1.set_xlabel('Models')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, mean_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Score distribution
    for i, name in enumerate(model_names):
        scores = results[name]['scores']
        ax2.hist(scores, alpha=0.7, label=name, bins=10)
    
    ax2.set_title('Score Distributions', fontweight='bold')
    ax2.set_xlabel('Cross-Validation Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, figsize=(10, 6)):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of the features
    figsize : tuple
        Figure size
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=figsize)
        plt.title('Feature Importance', fontsize=16, fontweight='bold')
        plt.bar(range(len(importances)), importances[indices], 
               color='lightcoral', alpha=0.7)
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("This model doesn't have feature importance attribute.")


def create_summary_plots(df, results=None, y_true=None, y_pred=None, 
                        model=None, feature_names=None):
    """
    Create a comprehensive set of summary plots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Iris dataset
    results : dict, optional
        Model training results
    y_true : array-like, optional
        True labels for confusion matrix
    y_pred : array-like, optional
        Predicted labels for confusion matrix
    model : sklearn estimator, optional
        Trained model for feature importance
    feature_names : list, optional
        Feature names for importance plot
    """
    print("Creating comprehensive visualization summary...")
    
    # Data exploration plots
    plot_data_distribution(df)
    plot_correlation_matrix(df)
    plot_pairplot(df)
    plot_boxplots(df)
    
    # Model evaluation plots
    if results is not None:
        plot_model_comparison(results)
    
    if y_true is not None and y_pred is not None:
        plot_confusion_matrix(y_true, y_pred)
    
    if model is not None and feature_names is not None:
        plot_feature_importance(model, feature_names)
    
    print("Visualization summary completed!") 