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
    st.sidebar.markdown("are your maashrooms edible or poisonous ? 🍄")
    
    @st.cache_data
    def load_data():
     data=pd.read_csv('/home/coder/Desktop/Project/mushrooms.csv')
     label=LabelEncoder()
     for col in data.columns:
        data[col]=label.fit_transform(data[col])
     return data
    
    @st.cache_data
    def split(df):
       y=df.type
       x=df.drop(columns=['type']) 
       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
       return x_train,x_test,y_train,y_test
    
    def plot_metrics(metrics_list):
       if 'confusion_Metrics' in metrics_list:
          st.subheader("confusion Metrics")
          plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)
          st.pyplot()
       if 'ROC Curve' in metrics_list:
           st.subheader("ROC Curve")
           plot_roc_curve(x_test,y_test)
           st.pyplot()
       if 'Precision Recal curve' in metrics_list:
         st.subheader("Precision Recal curves")
         plot_precision_recall_curve(model,x_test,y_test)
         st.pyplot()
    df=load_data()
    x_train,x_test,y_train,y_test=split(df)
    class_names=['edible','poisonous']
    st.sidebar.subheader("chose classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support vector machine(SVM)","logistic Regression","random forrest"))
    if classifier =="Support vector machine(SVM)":
      st.sidebar.subheader("Model hyperparameters")
      C=st.sidebar.number_input("c (regularization parameter)",0.01,10.0,step=0.01,key='C')
      kernel=st.sidebar.radio("kernel",("rbf","linear"),key='kernel')
      gamma=st.sidebar.radio("gamma(kernel coefficient)",("scale","auto"),key='gamma')
   
      metrics=st.sidebar.multiselect("what metrics to plot",("Support vector machine(SVM)","logistic Regression","random forrest"))
   
      if st.sidebar.button("classify",key='classify'):
       st.subheader("support vector machine (SVM)results")
       model=SVC(C=C,kernel=kernel,gamma=gamma)
       model.fit(x_train,y_train)
       accuracy=model.score(x_test,y_test)
       st.write("Accuracy",accuracy.round(2))
       st.write("precision",precision_score(y_test,y_pred,labels=class_names).round(2))
       st.write("recall",recall_score(y_test,y_pred,labels=class_names).round(2))
       plot_metrics(metrics)




    if st.sidebar.checkbox("show raw data",False):
     st.subheader("mushroom data set (Classification)")
     st.write(df)




if __name__ == '__main__':
     main()


