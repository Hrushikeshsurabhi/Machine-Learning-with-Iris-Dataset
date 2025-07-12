# Project Documentation

## Overview
This is an enhanced version of the original Iris dataset machine learning project, featuring a modular structure and comprehensive analysis capabilities.

## Project Structure

```
Machine-Learning-with-Iris-Dataset/
├── data/                           # Data files
│   └── Iris.csv                   # Original Iris dataset
├── notebooks/                      # Jupyter notebooks
│   ├── Machine Learning with Iris Dataset.ipynb
│   └── Iris Species Dataset Visualization.ipynb
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── models.py                 # Machine learning models
│   └── visualization.py          # Visualization functions
├── models/                        # Saved models (created during execution)
├── docs/                         # Documentation
│   └── README.md                 # This file
├── main.py                       # Main execution script
├── requirements.txt              # Python dependencies
├── README.md                     # Project README
├── .gitignore                    # Git ignore file
└── .gitattributes               # Git attributes
```

## Module Descriptions

### src/data_loader.py
- **Purpose**: Data loading and preprocessing functions
- **Key Functions**:
  - `load_iris_data()`: Load dataset from CSV or sklearn
  - `preprocess_data()`: Split data and apply scaling
  - `get_dataset_info()`: Get dataset statistics

### src/models.py
- **Purpose**: Machine learning models and evaluation
- **Key Classes**:
  - `IrisClassifier`: Comprehensive classifier with multiple algorithms
- **Supported Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Decision Tree

### src/visualization.py
- **Purpose**: Data exploration and model evaluation visualizations
- **Key Functions**:
  - `plot_data_distribution()`: Feature distributions by species
  - `plot_correlation_matrix()`: Feature correlations
  - `plot_model_comparison()`: Model performance comparison
  - `create_summary_plots()`: Comprehensive visualization suite

## Usage

### Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete pipeline:
   ```bash
   python main.py
   ```

### Using Individual Modules
```python
from src.data_loader import load_iris_data, preprocess_data
from src.models import IrisClassifier
from src.visualization import create_summary_plots

# Load and preprocess data
df = load_iris_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Train models
classifier = IrisClassifier()
results = classifier.train_all_models(X_train, y_train)

# Create visualizations
create_summary_plots(df, results, y_test, classifier.best_model.predict(X_test))
```

## Features

### Enhanced Structure
- **Modular Design**: Separated concerns into logical modules
- **Reusable Components**: Functions can be used independently
- **Comprehensive Documentation**: Detailed docstrings and examples

### Advanced Analysis
- **Multiple Algorithms**: 5 different classification algorithms
- **Cross-Validation**: Robust model evaluation
- **Feature Importance**: Analysis for tree-based models
- **Comprehensive Visualizations**: 8+ different plot types

### Production Ready
- **Model Persistence**: Save and load trained models
- **Error Handling**: Graceful handling of missing files
- **Scalable**: Easy to extend with new algorithms

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization

### Additional Libraries
- **joblib**: Model persistence
- **jupyter**: Interactive notebooks

## Contributing

This project is a fork of the original work by venky14. Enhancements include:
- Modular code structure
- Comprehensive documentation
- Additional visualization capabilities
- Production-ready features

## License

This project maintains the same license as the original repository. 