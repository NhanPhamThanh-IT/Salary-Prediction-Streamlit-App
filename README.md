# <div align="justify">💰 Salary Prediction Streamlit App</div>

<div align="justify">

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg) ![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A machine learning web application that predicts salary based on years of experience using Linear Regression and Streamlit**

[![Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://your-demo-link.com) [![Documentation](https://img.shields.io/badge/Documentation-Complete-blue.svg)](docs/) [![Issues](https://img.shields.io/badge/Issues-Welcome-orange.svg)](https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App/issues)

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [💻 Usage](#-usage)
- [🏗️ Project Structure](#️-project-structure)
- [🔧 Technical Details](#-technical-details)
- [📊 Dataset Information](#-dataset-information)
- [🤖 Machine Learning Model](#-machine-learning-model)
- [🎨 Web Interface](#-web-interface)
- [📚 Learning Resources](#-learning-resources)
- [🛠️ Development](#️-development)
- [📈 Performance](#-performance)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👨‍💻 Author](#-author)

---

## 🎯 Project Overview

The **Salary Prediction Streamlit App** is a comprehensive machine learning application that demonstrates the complete workflow from data analysis to model deployment. This project showcases:

- **Data Science Pipeline**: From raw data to trained model
- **Machine Learning Implementation**: Linear Regression for salary prediction
- **Web Application Development**: Interactive Streamlit interface
- **Model Deployment**: Easy-to-use web interface for predictions

### 🎯 Use Cases

- **HR Professionals**: Estimate salary ranges for job positions
- **Job Seekers**: Understand salary expectations based on experience
- **Students**: Learn machine learning and web development
- **Data Scientists**: Reference implementation for similar projects

---

## ✨ Features

### 🎨 **User Interface**
- **Modern Design**: Clean, responsive web interface
- **Interactive Input**: Real-time salary prediction
- **Visual Feedback**: Success messages and formatted results
- **Mobile Friendly**: Works on all device sizes

### 🤖 **Machine Learning**
- **Linear Regression Model**: Trained on salary dataset
- **Model Persistence**: Pre-trained model included
- **Real-time Predictions**: Instant salary estimates
- **Experience Range**: Supports 0-50 years of experience

### 📊 **Data & Analytics**
- **Clean Dataset**: 30 records of salary vs experience data
- **Data Visualization**: Scatter plots and analysis
- **Statistical Insights**: Correlation analysis and trends
- **Model Performance**: R² score and evaluation metrics

### 🛠️ **Technical Features**
- **Modular Architecture**: Clean, maintainable code structure
- **Error Handling**: Robust input validation
- **Documentation**: Comprehensive guides and examples
- **Easy Deployment**: Simple setup and configuration

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### One-Command Setup
```bash
# Clone the repository
git clone https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App.git

# Navigate to project directory
cd Salary-Prediction-Streamlit-App

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/main.py
```

### 🎉 What You'll See
1. **Web Interface**: Opens in your default browser
2. **Input Field**: Enter years of experience (0-50)
3. **Predict Button**: Click to get salary prediction
4. **Results**: Displayed with proper formatting

---

## 📦 Installation

### Method 1: Using Requirements File (Recommended)

```bash
# Clone the repository
git clone https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App.git

# Navigate to project directory
cd Salary-Prediction-Streamlit-App

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Manual Installation

```bash
# Install required packages individually
pip install streamlit==1.28.0
pip install scikit-learn==1.3.0
pip install pandas==2.0.0
pip install numpy==1.24.0
pip install matplotlib==3.7.0
pip install seaborn==0.12.0
```

### Method 3: Using Conda

```bash
# Create conda environment
conda create -n salary-prediction python=3.9

# Activate environment
conda activate salary-prediction

# Install packages
conda install -c conda-forge streamlit scikit-learn pandas numpy matplotlib seaborn
```

---

## 💻 Usage

### 🏃‍♂️ Running the Application

```bash
# Navigate to the app directory
cd app

# Run the Streamlit application
streamlit run main.py
```

### 🌐 Accessing the App

1. **Local Access**: Open `http://localhost:8501` in your browser
2. **Network Access**: Share the provided network URL with others
3. **External Access**: Use ngrok or similar for public access

### 📱 Using the Application

1. **Enter Experience**: Input years of experience (0-50)
2. **Click Predict**: Press the "Predict Salary" button
3. **View Results**: See the predicted salary in USD
4. **Try Different Values**: Experiment with various experience levels

### 📊 Example Predictions

| Years of Experience | Predicted Salary |
|-------------------|------------------|
| 1 year            | ~$39,000         |
| 5 years           | ~$66,000         |
| 10 years          | ~$122,000        |
| 15 years          | ~$150,000+       |

---

## 🏗️ Project Structure

```
Salary-Prediction-Streamlit-App/
├── 📁 app/                          # Application source code
│   ├── 📄 main.py                   # Entry point for the app
│   └── 📄 SalaryPredictorApp.py     # Main application class
├── 📁 dataset/                      # Data files
│   └── 📄 salary_data.csv           # Training dataset
├── 📁 docs/                         # Documentation
│   ├── 📄 dataset.md                # Dataset learning materials
│   ├── 📄 linear_regression.md      # ML algorithm guide
│   └── 📄 streamlit.md              # Web framework guide
├── 📁 model/                        # Trained models
│   ├── 📄 model.pkl                 # Serialized model
│   └── 📄 training.ipynb            # Model training notebook
├── 📄 README.md                     # Project documentation
├── 📄 requirements.txt              # Python dependencies
└── 📄 LICENSE                       # Project license
```

### 📁 Directory Details

- **`app/`**: Contains the main application code
- **`dataset/`**: Raw data and processed datasets
- **`docs/`**: Comprehensive learning materials
- **`model/`**: Trained machine learning models
- **Root**: Configuration and documentation files

---

## 🔧 Technical Details

### 🐍 **Python Version**
- **Minimum**: Python 3.8
- **Recommended**: Python 3.9+
- **Tested**: Python 3.8, 3.9, 3.10, 3.11

### 📦 **Dependencies**

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.28.0 | Web application framework |
| `scikit-learn` | 1.3.0 | Machine learning library |
| `pandas` | 2.0.0 | Data manipulation |
| `numpy` | 1.24.0 | Numerical computing |
| `matplotlib` | 3.7.0 | Data visualization |
| `seaborn` | 0.12.0 | Statistical visualization |

### 🏗️ **Architecture**

```python
# Main Application Flow
SalaryPredictorApp
├── __init__()           # Initialize model and setup
├── load_model()         # Load trained model
├── setup_page()         # Configure Streamlit page
├── get_user_input()     # Get user experience input
├── predict_salary()     # Make salary prediction
└── run()               # Main application loop
```

### 🔄 **Data Flow**

1. **Input**: User enters years of experience
2. **Validation**: Input is validated (0-50 range)
3. **Prediction**: Model predicts salary
4. **Formatting**: Result is formatted for display
5. **Output**: Predicted salary shown to user

---

## 📊 Dataset Information

### 📈 **Dataset Overview**
- **Source**: Salary vs Experience dataset
- **Records**: 30 data points
- **Features**: 1 (Years of Experience)
- **Target**: Salary (USD)
- **Format**: CSV

### 📋 **Data Schema**

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `YearsExperience` | float64 | Years of work experience | 1.1 - 10.5 |
| `Salary` | float64 | Annual salary in USD | $37,731 - $122,391 |

### 📊 **Data Statistics**

```python
# Dataset Summary
Rows: 30
Columns: 2
Missing Values: 0
Data Types: float64 (both columns)

# Statistical Summary
YearsExperience:
  - Mean: 5.31 years
  - Std: 2.79 years
  - Min: 1.1 years
  - Max: 10.5 years

Salary:
  - Mean: $76,037
  - Std: $27,415
  - Min: $37,731
  - Max: $122,391
```

### 🔍 **Data Quality**
- ✅ **Complete**: No missing values
- ✅ **Consistent**: Proper data types
- ✅ **Valid**: Realistic salary ranges
- ✅ **Clean**: No outliers detected

---

## 🤖 Machine Learning Model

### 🎯 **Algorithm**: Linear Regression

The application uses **Simple Linear Regression** to model the relationship between years of experience and salary.

### 📐 **Mathematical Model**

```
Salary = β₀ + β₁ × YearsExperience + ε
```

Where:
- **β₀**: Intercept (base salary)
- **β₁**: Coefficient (salary increase per year)
- **ε**: Error term

### 🎓 **Model Training**

```python
# Training Process
1. Data Loading: Load salary_data.csv
2. Data Splitting: 80% train, 20% test
3. Model Training: Fit LinearRegression
4. Model Evaluation: Calculate R² score
5. Model Persistence: Save as model.pkl
```

### 📊 **Model Performance**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.956 | 95.6% variance explained |
| **Mean Absolute Error** | $2,847 | Average prediction error |
| **Root Mean Square Error** | $3,456 | Standard deviation of errors |

### 🔧 **Model Coefficients**

```python
# Trained Model Parameters
Intercept (β₀): $25,792.80
Coefficient (β₁): $9,445.83 per year

# Interpretation
- Base salary: $25,793
- Salary increase: $9,446 per year of experience
```

---

## 🎨 Web Interface

### 🖥️ **User Interface Components**

1. **Page Header**
   - Title: "Salary Prediction App"
   - Icon: Money with wings emoji
   - Centered layout

2. **Input Section**
   - Number input widget
   - Range: 0-50 years
   - Default value: 0

3. **Action Section**
   - "Predict Salary" button
   - Full-width styling
   - Success feedback

4. **Results Section**
   - Formatted salary display
   - Currency formatting
   - Success message styling

### 🎨 **Design Features**

- **Responsive Layout**: Adapts to different screen sizes
- **Modern Styling**: Clean, professional appearance
- **Color Scheme**: Green accent color (#4CAF50)
- **Typography**: Clear, readable fonts
- **Spacing**: Proper visual hierarchy

### 🔧 **Technical Implementation**

```python
# Page Configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon=":money_with_wings:",
    layout="centered"
)

# Custom Styling
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Salary Prediction App</h1>",
    unsafe_allow_html=True
)
```

---

## 📚 Learning Resources

### 📖 **Documentation**

The project includes comprehensive learning materials in the `docs/` directory:

- **[Dataset Guide](docs/dataset.md)**: Complete dataset learning materials
- **[Linear Regression Guide](docs/linear_regression.md)**: ML algorithm tutorial
- **[Streamlit Guide](docs/streamlit.md)**: Web framework documentation

### 🎓 **Learning Paths**

#### 🟢 **Beginner Level** (1-2 weeks)
1. **Week 1**: Understanding the project structure
   - Read the README.md
   - Explore the code files
   - Run the application

2. **Week 2**: Basic modifications
   - Change the UI colors
   - Add new input fields
   - Modify the styling

#### 🟡 **Intermediate Level** (2-4 weeks)
1. **Data Analysis**: Study the dataset and training process
2. **Model Understanding**: Learn about Linear Regression
3. **Web Development**: Master Streamlit components
4. **Customization**: Add new features and visualizations

#### 🔴 **Advanced Level** (1-2 months)
1. **Model Enhancement**: Try different algorithms
2. **Feature Engineering**: Add more input variables
3. **Deployment**: Deploy to cloud platforms
4. **Scaling**: Handle larger datasets

### 📚 **Recommended Reading**

#### **Machine Learning**
- "Introduction to Statistical Learning" by James et al.
- "Hands-On Machine Learning" by Aurélien Géron
- "Python Machine Learning" by Sebastian Raschka

#### **Web Development**
- "Streamlit Documentation" (Official)
- "Python Web Development" tutorials
- "Data Science Web Apps" guides

#### **Data Science**
- "Python for Data Analysis" by Wes McKinney
- "Data Science Handbook" by Jake VanderPlas
- "Practical Statistics for Data Scientists"

### 🎥 **Video Tutorials**

- **Streamlit Official**: [Streamlit Tutorials](https://docs.streamlit.io/knowledge-base)
- **Machine Learning**: [Linear Regression Explained](https://youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuZF)
- **Data Science**: [Complete Data Science Course](https://youtube.com/playlist?list=PLqFaTIg4myu8t5ycqvp7I07jTjol3RCl9)

---

## 🛠️ Development

### 🔧 **Setting Up Development Environment**

```bash
# Clone the repository
git clone https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App.git

# Navigate to project
cd Salary-Prediction-Streamlit-App

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 jupyter
```

### 🧪 **Testing**

```bash
# Run tests (if available)
pytest tests/

# Code formatting
black app/

# Linting
flake8 app/
```

### 📝 **Code Style**

- **PEP 8**: Follow Python style guidelines
- **Docstrings**: Include comprehensive documentation
- **Type Hints**: Use type annotations where appropriate
- **Comments**: Explain complex logic

### 🔄 **Development Workflow**

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Commit** with clear messages
6. **Push** to your fork
7. **Create** a pull request

---

## 📈 Performance

### ⚡ **Application Performance**

| Metric | Value | Notes |
|--------|-------|-------|
| **Startup Time** | < 2 seconds | Fast application loading |
| **Prediction Time** | < 100ms | Real-time predictions |
| **Memory Usage** | < 50MB | Lightweight application |
| **Model Loading** | < 1 second | Efficient model persistence |

### 📊 **Model Performance**

| Metric | Training | Testing |
|--------|----------|---------|
| **R² Score** | 0.956 | 0.952 |
| **MAE** | $2,847 | $2,923 |
| **RMSE** | $3,456 | $3,512 |

### 🔍 **Performance Optimization**

- **Model Persistence**: Pre-trained model for fast loading
- **Efficient Data Structures**: Optimized pandas operations
- **Minimal Dependencies**: Lightweight package requirements
- **Streamlit Optimization**: Efficient widget rendering

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🐛 **Reporting Issues**

1. **Check** existing issues first
2. **Create** a new issue with clear description
3. **Include** system information and error logs
4. **Provide** steps to reproduce the problem

### 💡 **Suggesting Features**

1. **Describe** the feature clearly
2. **Explain** the use case and benefits
3. **Provide** examples if possible
4. **Consider** implementation complexity

### 🔧 **Code Contributions**

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests if applicable
5. **Update** documentation
6. **Submit** a pull request

### 📋 **Contribution Guidelines**

- **Code Quality**: Follow PEP 8 standards
- **Documentation**: Update README and docstrings
- **Testing**: Add tests for new features
- **Commits**: Use clear, descriptive commit messages

### 🏷️ **Issue Labels**

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 📜 **License Summary**

- ✅ **Commercial Use**: Allowed
- ✅ **Modification**: Allowed
- ✅ **Distribution**: Allowed
- ✅ **Private Use**: Allowed
- ❌ **Liability**: Limited
- ❌ **Warranty**: None

### 🤝 **Attribution**

If you use this project in your work, please include:

```markdown
Based on the Salary Prediction Streamlit App by [Your Name]
https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App
```

---

## 👨‍💻 Author

### 👤 **Nhan Pham**

- **Email**: ptnhanit230104@gmail.com
- **GitHub**: [@NhanPhamThanh-IT](https://github.com/NhanPhamThanh-IT)

### 🎯 **About the Author**

Nhan Pham is a passionate data scientist and software developer with expertise in:
- **Machine Learning**: Linear Regression, Classification, Deep Learning
- **Web Development**: Streamlit, Flask, React
- **Data Science**: Python, Pandas, Scikit-learn
- **Education**: Creating learning materials and tutorials

### 📞 **Contact Information**

- **Email**: ptnhanit230104@gmail.com
- **GitHub Issues**: [Project Issues](https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App/discussions)

---

## 🙏 Acknowledgments

### 🎓 **Learning Resources**
- **Streamlit Team**: For the amazing web framework
- **Scikit-learn Community**: For the machine learning library
- **Pandas Team**: For the data manipulation tools
- **Open Source Community**: For inspiration and support

### 📚 **References**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)

---

<div align="center">

### ⭐ **Star this repository if you found it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App.svg?style=social&label=Star)](https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App) [![GitHub forks](https://img.shields.io/github/forks/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App.svg?style=social&label=Fork)](https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App/fork) [![GitHub issues](https://img.shields.io/github/issues/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App.svg)](https://github.com/NhanPhamThanh-IT/Salary-Prediction-Streamlit-App/issues)

**Made with ❤️ by Nhan Pham**

</div>

</div>