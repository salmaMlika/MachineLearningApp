import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title("Binary classification web app")
    st.sidebar.title("Binary classification web app")
    st.markdown("are your mashrooms edible or poisonous ? 🍄")
    st.sidebar.markdown("are your mashrooms edible or poisonous ? 🍄")

    def load_data():
     data=pd.read_csv('/home/coder/Desktop/Project/mushrooms.csv')
     label=LabelEncoder()
     for col in data.columns:
        data[col]=label.fit_transform(data[col])
     return data

    df=load_data()

    if st.sidebar.checkbox("show raw data",False):
     st.subheader("mushroom data set (Classification)")
     st.write(df)




if __name__ == '__main__':
     main()


