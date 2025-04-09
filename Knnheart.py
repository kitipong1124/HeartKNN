from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.title('การตรวจสอบโรคหัวใจด้วยเทคนิค Machine Learning 💀💀💀')
col1,col2 = st.columns(2)

with col1:
   st.header("Happy")
   st.image("./img/happy.jpg")

with col2:
   st.header("Dead")
   st.image("./img/end.jpg")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

