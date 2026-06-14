import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
df1 = pd.read_csv('cleaned_carprice.csv') 
grouped=df1.groupby('company')




with open('CarPriceModel.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(layout='centered',page_title='Car price prediction')
st.title("Welcome to Car Price Predictor",text_alignment='center')
st.text("This app provides the estimated price of car you want to sell. Try it by filling the below information",text_alignment='justify')
company = st.selectbox("Select the company : ",(df1['company'].unique()))
model_name = st.selectbox("Select the model : ",(grouped.get_group(company)['name']))
year = st.selectbox("Select the year of purchase : ",(sorted(df1['year'].unique())))
fuel = st.selectbox("Select the fuel type : ",(df1['fuel_type'].unique()))
kms_input = st.text_input(label='Enter the kilometers driven:', label_visibility='visible')

col1, col2, col3 = st.columns([1, 1, 1])


with col2:
    if st.button("Predict Price"):
        input = pd.DataFrame([[str(model_name),str(company),year,int(kms_input),str(fuel)]],columns=['name','company','year','kms_driven','fuel_type'])
        prediction = model.predict(input)
        st.success(f'The estimated price is ₹{round(prediction[0])}',width='stretch')        
