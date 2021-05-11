import streamlit as st
import pickle
import numpy as np
import matplotlob.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding',False)
#load pickle model
model=pickle.load(open('Naive_Bayes_education_model.pkl','rb'))
datadet=pd.read_xlsx('Classification Dataset2.xlsx')
X=datadet.iloc[:,0:-1].values
