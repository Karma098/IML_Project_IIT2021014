import streamlit as st
import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Load the model and dataset
pipe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipe.pkl')
df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'df.pkl')

pipe = pickle.load(open(pipe_path, 'rb'))
df = pickle.load(open(df_path, 'rb'))

st.title("Laptop Predictor")

# Collect input
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight(in Kg)')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

# Label Encoding (assuming this is how your pipeline expects input)
company_encoder = LabelEncoder().fit(df['Company'])
type_encoder = LabelEncoder().fit(df['TypeName'])
cpu_encoder = LabelEncoder().fit(df['Cpu brand'])
gpu_encoder = LabelEncoder().fit(df['Gpu brand'])
os_encoder = LabelEncoder().fit(df['os'])

if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Encode categorical variables
    company_encoded = company_encoder.transform([company])[0]
    type_encoded = type_encoder.transform([type])[0]
    cpu_encoded = cpu_encoder.transform([cpu])[0]
    gpu_encoded = gpu_encoder.transform([gpu])[0]
    os_encoded = os_encoder.transform([os])[0]

    # Create the query array
    query = np.array([company_encoded, type_encoded, ram, weight, touchscreen, ips, ppi, cpu_encoded, hdd, ssd, gpu_encoded, os_encoded])
    query = query.reshape(1, -1)  # Reshape to (1, 12) for a single prediction

    try:
        prediction = str(int(np.exp(pipe.predict(query)[0])))
        st.title(f"The predicted price of this laptop can be around {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write(f"Query Shape: {query.shape}")
        st.write(f"Query Content: {query}")
