import streamlit as st  
import pickle
import numpy as np 
import pandas as pd 

df=pd.read_csv("healthcare_dataset1.csv")
pipe=pickle.load(open("pipe.pkl","rb"))
st.title("Healthcare prediction")

age=st.number_input("Enter your age:")
gender=st.selectbox("Select your gender:",df['Gender'].unique())
Blood=st.selectbox("Select your blood type: ",df['Blood Type'].unique())
MedC=st.selectbox("Entr your Medical Condition:",df['Medical Condition'].unique())
InP=st.selectbox("Enter your Insurance Provider:",df['Insurance Provider'].unique())
bill=st.number_input("Enter your bill:")
At=st.selectbox("Enter your Admission Type:",df['Admission Type'].unique())
Medi=st.selectbox("Enter your Medication: ",df["Medication"].unique())

if st.button("Predict"):
    querry=np.array([age,gender,Blood,MedC,InP,bill,At,Medi])
    querry=querry.reshape(1,8)
    if pipe.predict(querry)==2:
        st.title("Normal")
    elif pipe.predict(querry)==1:
        st.title("Inconclusive")
    else:
        st.title("Abnormal")