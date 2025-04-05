import streamlit as st
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb


#load model
model = pickle.load(open('streamlit_app\churn_prediction_model.pkl','rb'))
x_train =pd.read_csv('H:/GitHub/ecommerce-churn-prediction-main/modelling/training_data.csv')
def pred(category,item,quantity,price,payment_method,card_type,city,province_state,country,gender,age,days_since_last_purchase,
          tenure,discount_used,purchase_frequency,avg_purchase_value,recency,purchase_per_tenure,discount_ratio):
    features=np.array([[category,item,quantity,price,payment_method,card_type,city,province_state,country,gender,age,days_since_last_purchase,
          tenure,discount_used,purchase_frequency,avg_purchase_value,recency,purchase_per_tenure,discount_ratio]])
    prediction= model.predict(features).reshape(1,-1)
    return prediction[0]
#web app
st.title('Customer Churn Prediction')

category=st.selectbox("category",x_train["category"].unique())
item=st.selectbox("item",x_train["item"].unique())
quantity=st.selectbox("quantity",x_train["quantity"].unique())
price=st.selectbox("price",x_train["price"].unique())
payment_method= st.selectbox("payment Method", x_train["payment_method"].unique())
card_type= st.selectbox("Card Type", x_train["card_type"].unique())
city=st.selectbox("city",x_train["city"].unique())
province_state=st.selectbox("province_state",x_train["province_state"].unique())
country=st.selectbox("country",x_train["country"].unique())
gender=st.selectbox("gender",x_train["gender"].unique())
age=st.selectbox("age",x_train["age"].unique())
days_since_last_purchase=st.selectbox("days_since_last_purchase",x_train["days_since_last_purchase"].unique())
tenure=st.selectbox("tenure",x_train["tenure"].unique())
discount_used=st.selectbox("discount_used",x_train["discount_used"].unique())
purchase_frequency=st.selectbox("purchase_frequency",x_train["purchase_frequency"].unique())
avg_purchase_value=st.selectbox("avg_purchase_value",x_train["avg_purchase_value"].unique())
recency=st.selectbox("recency",x_train["recency"].unique())
purchase_per_tenure=st.selectbox("purchase_per_tenure",x_train["purchase_per_tenure"].unique())
discount_ratio=st.selectbox("discount_ratio",x_train["discount_ratio"].unique())

result=pred(category,item,quantity,price,payment_method,card_type,city,province_state,country,gender,age,days_since_last_purchase,
tenure,discount_used,purchase_frequency,avg_purchase_value,recency,purchase_per_tenure,discount_ratio)

if st.button("predict"):
    st.write(result)