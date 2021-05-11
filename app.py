import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('Naive_Bayes_education_model.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_excel('Classification Dataset2.xlsx')
# Extracting independent variable:
X=datadet.iloc[:,0:-1].values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(UserID, Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age):
  output= model.predict(sc.transform([[UserID,Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age]]))
  print("Disease", output)
  if output==[1]:
    prediction="Patient will have the disease."
  else:
    prediction="Patient will not have the disease."
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Science Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning MID TERM 1</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Disease Prediction using Naive Bayes")
    
    UserID = st.text_input("UserID","")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Glucose=st.number_input('Insert Glucose:')
    BP=st.number_input('Insert BP:')
    SkinThickness=st.number_input('Insert Skin Thickness:')
    Insulin=st.number_input('Insert Insulin:')
    BMI=st.number_input('Insert BMI:')
    PedigreeFunction=st.number_input('Insert Pedigree Function:')
    Age = st.number_input('Insert a Age:')
    result=""
    if st.button("Predict"):
      result=predict_note_authentication(UserID,Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Bhavya Maheshwari")
      st.subheader("PIET18CS036")

if __name__=='__main__':
  main()
