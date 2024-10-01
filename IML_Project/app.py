import streamlit as st
import pickle
import numpy as np
import os

# Load the model and data
pipe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipe.pkl')
df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'df.pkl')

pipe = pickle.load(open(pipe_path, 'rb'))
df = pickle.load(open(df_path, 'rb'))

# Encode categorical features (assuming you have encoders saved)
company_encoder = pickle.load(open('company_encoder.pkl', 'rb'))
type_encoder = pickle.load(open('type_encoder.pkl', 'rb'))
cpu_encoder = pickle.load(open('cpu_encoder.pkl', 'rb'))
gpu_encoder = pickle.load(open('gpu_encoder.pkl', 'rb'))
os_encoder = pickle.load(open('os_encoder.pkl', 'rb'))

st.title("Laptop Predictor")

# User inputs
company = st.selectbox('Brand', df['Company'].unique())
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 244, 32, 64])
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

if st.button('Predict Price'):
    try:
        # Encode categorical inputs
        company_encoded = company_encoder.transform([company])[0]
        type_encoded = type_encoder.transform([type])[0]
        cpu_encoded = cpu_encoder.transform([cpu])[0]
        gpu_encoded = gpu_encoder.transform([gpu])[0]
        os_encoded = os_encoder.transform([os])[0]

        # Convert other inputs
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Create input query array
        query = np.array([
            float(company_encoded),
            float(type_encoded),
            int(ram),
            float(weight),
            int(touchscreen),
            int(ips),
            float(ppi),
            float(cpu_encoded),
            int(hdd),
            int(ssd),
            float(gpu_encoded),
            float(os_encoded)
        ])

        # Reshape the query for the model
        query = query.reshape(1, -1)

        # Debugging: Check types of the query array
        print("Input types:", [type(x) for x in query.flatten()])

        # Predict
        prediction = str(int(np.exp(pipe.predict(query)[0])))
        st.title(f"The predicted price of this laptop can be around {prediction}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
