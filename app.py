import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import joblib as jb
import os
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

st.write("Chennai House Price Prediction System")

def inp(inpt,dicet):
    return dicet[inpt]

Area_dic={"Adyar": 0, "Anna Nagar": 1, "Chormpet": 2, "KK Nagar": 3, "Karapakkam": 4, "TNagar": 5, "Velachery": 6}
Sale_cond_dic={"Abnormal": 0, "AdjLand": 1, "Family": 2, "Normal Sale": 3, "Partial": 4}
Park_facil_dic={"No": 0, "Yes": 1}
Build_type_dic={"Commercial": 0, "House": 1, "Other": 2}
Utility_avail_dic={"AllPub": 0, "ELO": 1, "NoSeWa": 2, "NoSewr": 3}
Street_dic={"Gravel": 0, "NoAccess": 1, "Paved": 2}
MZZone_dic={"A": 0, "C": 1, "I": 2, "RH": 3, "RL": 4, "RM": 5}



model=jb.load('modelETR.pkl')
Ar_inp=st.selectbox("Area",['Adyar', 'Anna Nagar', 'Chormpet', 'KK Nagar', 'Karapakkam', 'TNagar', 'Velachery'])
Area=inp(Ar_inp,Area_dic)
INT_SQFT= st.number_input("INT_SQFT")
DIST_MAINROAD = st.number_input("DIST_MAINROAD")
N_BEDROOM=st.number_input("N_BEDROOM")
N_BATHROOM=st.number_input("N_BATHROOM")
N_ROOM=st.number_input("N_ROOM")
sale_c=st.selectbox("Sale Cond",['Abnormal', 'AdjLand', 'Family', 'Normal Sale', 'Partial'])
Sale_cond=inp(sale_c,Sale_cond_dic)
park_f=st.selectbox("Park Facility",['No', 'Yes'])
Park_facil=inp(park_f,Park_facil_dic)
buld_t=st.selectbox("Build Type",['Commercial', 'House', 'Other'])
Build_type=inp(buld_t,Build_type_dic)
ut_av=st.selectbox("Utility Avail",['AllPub', 'ELO', 'NoSeWa', 'NoSewr'])
Utility_avail=inp(ut_av,Utility_avail_dic)
strt=st.selectbox("Street",['Gravel', 'NoAccess', 'Paved'])
Street=inp(strt,Street_dic)
mzz=st.selectbox("MZ Zone",['A', 'C', 'I', 'RH', 'RL', 'RM'])
MZZone=inp(mzz,MZZone_dic)
QS_ROOMS=st.number_input("QS_ROOMS")
QS_BATHROOM=st.number_input("QS_BATHROOM")
QS_BEDROOM=st.number_input("QS_BEDROOM")
QS_OVERALL=st.number_input("QS_OVERALL")
REG_FEE=st.number_input("REG_FEE")
COMMIS=st.number_input("COMMIS")

def run(iar):
    pred = model.predict(iar)
    return pred
from sklearn.preprocessing import MinMaxScaler
mmscaler=MinMaxScaler(feature_range=(0,1))

inp_arr=[Area,INT_SQFT,DIST_MAINROAD,N_BEDROOM,N_BATHROOM,N_ROOM,Sale_cond,Park_facil,Build_type,Utility_avail,Street,MZZone,QS_ROOMS,QS_BATHROOM,QS_BEDROOM,QS_OVERALL,REG_FEE,COMMIS]
our_iip=np.array(inp_arr).reshape(-1,1)
if st.button("Run"):
    pred=model.predict([[Area,INT_SQFT,DIST_MAINROAD,N_BEDROOM,N_BATHROOM,N_ROOM,Sale_cond,Park_facil,Build_type,Utility_avail,Street,MZZone,QS_ROOMS,QS_BATHROOM,QS_BEDROOM,QS_OVERALL,REG_FEE,COMMIS]])
    st.write(pred[0])
else:
    st.write('Please Press Run')

