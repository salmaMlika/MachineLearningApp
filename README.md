# Interactive Machine Learning Web App

This project showcases an interactive web application built with **Streamlit** and **Python**, enabling users to experiment with Machine Learning models without writing a single line of code.
The app allows users to select a classification algorithm and interactively adjust hyperparameters through an intuitive web interface.

---

## 📜 Project Overview

The goal of this project is to make Machine Learning accessible to everyone by providing an easy-to-use web application. With this app, users can:

- Select from multiple classification algorithms (Logistic Regression, Support Vector Machines, Random Forests).
- Adjust hyperparameter values interactively via sliders and dropdown menus.
- Visualize model performance metrics, making it easy to understand the impact of hyperparameters.

This application requires no prior programming knowledge, making it ideal for non-technical users and those new to Machine Learning.

---

## ✨ Features

1. **Algorithm Selection**: Choose between Logistic Regression, Support Vector Machines, and Random Forest classifiers.
2. **Hyperparameter Adjustment**: Intuitively tweak hyperparameters like regularization strength, tree depth, and more using interactive controls.
3. **Performance Metrics Visualization**: View key metrics such as accuracy, precision, recall, and F1-score.
4. **No Coding Required**: Designed for users with zero programming experience.

---

## 🚀 How to Use

### Prerequisites
Ensure you have Python installed along with the required libraries.
```bash
pip install -r requirements.txt

### Launch the App with Streamlit

To run the application, use the following command:

```bash
streamlit run app.py

http://localhost:8501

# 🛠️ How It Works

1. **Load Data**: The app includes a preloaded dataset for demonstration purposes.
2. **Choose an Algorithm**: Select a classification algorithm (e.g., Logistic Regression, Random Forest).
3. **Set Hyperparameters**: Use sliders and dropdowns to modify model settings interactively.
4. **Visualize Results**: View metrics and visualizations generated in real-time based on your chosen settings.
