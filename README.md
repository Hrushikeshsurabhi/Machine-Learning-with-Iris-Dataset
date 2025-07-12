# 🌸 Machine Learning with Iris Dataset

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-red.svg)](https://scikit-learn.org/)
[![Type](https://img.shields.io/badge/Type-Supervised-yellow.svg)](https://en.wikipedia.org/wiki/Supervised_learning)
[![Status](https://img.shields.io/badge/Status-Enhanced-green.svg)](https://github.com/Hrushikeshsurabhi/Machine-Learning-with-Iris-Dataset)
[![Fork](https://img.shields.io/badge/Fork-venky14-orange.svg)](https://github.com/venky14/Machine-Learning-with-Iris-Dataset)

> **Enhanced Version**: This is a forked and improved version of the original Iris dataset project by [venky14](https://github.com/venky14/Machine-Learning-with-Iris-Dataset), featuring a modular structure, comprehensive analysis, and production-ready code.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)

## 🌺 Introduction

The Iris dataset is a classic dataset for classification, machine learning, and data visualization. This enhanced version provides a comprehensive analysis with a modular, production-ready structure.

### Dataset Information
- **3 Classes**: Different Iris species (Setosa, Versicolor, Virginica)
- **4 Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **150 Samples**: 50 samples per species
- **Linearly Separable**: Iris Setosa is linearly separable from the other two species

### Enhanced Features
- 🏗️ **Modular Architecture**: Clean separation of concerns
- 🤖 **Multiple Algorithms**: 5 different classification models
- 📊 **Comprehensive Visualizations**: 8+ different plot types
- 🔄 **Cross-Validation**: Robust model evaluation
- 💾 **Model Persistence**: Save and load trained models
- 📚 **Detailed Documentation**: Complete API documentation

## 📁 Project Structure

```
Machine-Learning-with-Iris-Dataset/
├── 📊 data/                           # Data files
│   └── Iris.csv                      # Original Iris dataset
├── 📓 notebooks/                      # Jupyter notebooks
│   ├── Machine Learning with Iris Dataset.ipynb
│   └── Iris Species Dataset Visualization.ipynb
├── 🔧 src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── models.py                    # Machine learning models
│   └── visualization.py             # Visualization functions
├── 🤖 models/                        # Saved models (created during execution)
├── 📖 docs/                         # Documentation
│   └── README.md                    # Detailed documentation
├── 🚀 main.py                       # Main execution script
├── 📋 requirements.txt              # Python dependencies
└── 📄 README.md                     # This file
```

## ✨ Features

### 🎯 Machine Learning Models
- **Logistic Regression**: Linear classification
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Random Forest**: Ensemble learning
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Decision Tree**: Tree-based classification

### 📈 Analysis Capabilities
- **Data Exploration**: Comprehensive statistical analysis
- **Feature Importance**: Model interpretability
- **Model Comparison**: Performance benchmarking
- **Cross-Validation**: Robust evaluation
- **Confusion Matrix**: Detailed error analysis

### 🎨 Visualization Suite
- **Distribution Plots**: Feature distributions by species
- **Correlation Matrix**: Feature relationships
- **Pair Plots**: Multi-dimensional relationships
- **Box Plots**: Statistical summaries
- **Model Performance**: Comparison charts
- **Confusion Matrix**: Error visualization
- **Feature Importance**: Model interpretability

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Hrushikeshsurabhi/Machine-Learning-with-Iris-Dataset.git
cd Machine-Learning-with-Iris-Dataset
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Complete Pipeline
```bash
python main.py
```

This will:
- Load and preprocess the Iris dataset
- Train 5 different machine learning models
- Compare model performances
- Generate comprehensive visualizations
- Save the best performing model

## 💻 Usage Examples

### Basic Usage
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

### Advanced Usage
```python
# Custom preprocessing
X_train, X_test, y_train, y_test, scaler = preprocess_data(df, test_size=0.3, random_state=123)

# Train with custom parameters
classifier = IrisClassifier()
results = classifier.train_all_models(X_train, y_train, cv=10)

# Save and load models
classifier.save_model('my_iris_model.pkl')
classifier.load_model('my_iris_model.pkl')
```

## 📚 Documentation

- **[Detailed Documentation](docs/README.md)**: Complete module descriptions and API reference
- **[Jupyter Notebooks](notebooks/)**: Interactive examples and tutorials
- **[Source Code](src/)**: Well-documented source modules

## 🛠️ Dependencies

### Core Libraries
- **pandas** (≥1.3.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computing
- **scikit-learn** (≥1.0.0): Machine learning algorithms
- **matplotlib** (≥3.4.0): Basic plotting
- **seaborn** (≥0.11.0): Statistical data visualization

### Additional Libraries
- **joblib** (≥1.1.0): Model persistence
- **jupyter** (≥1.0.0): Interactive notebooks

## 🤝 Contributing

This project is a fork of the original work by [venky14](https://github.com/venky14/Machine-Learning-with-Iris-Dataset). 

### Enhancements Made
- ✅ Modular code structure
- ✅ Comprehensive documentation
- ✅ Additional visualization capabilities
- ✅ Production-ready features
- ✅ Enhanced error handling
- ✅ Model persistence functionality

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project maintains the same license as the original repository by venky14.

## 🙏 Acknowledgments

- **Original Author**: [venky14](https://github.com/venky14) for the foundational work
- **Dataset Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Libraries**: scikit-learn, pandas, matplotlib, seaborn communities

---

<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/Hrushikeshsurabhi">Hrushikeshsurabhi</a></p>
  <p>Forked from <a href="https://github.com/venky14/Machine-Learning-with-Iris-Dataset">venky14/Machine-Learning-with-Iris-Dataset</a></p>
</div>
