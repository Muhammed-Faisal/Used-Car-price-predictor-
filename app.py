import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

@st.cache_data
def load_data():
    # Load data
    df = pd.read_csv('df.csv')
    return df

@st.cache_resource
def load_model():
    # Download and load model
    model_url = 'https://drive.google.com/uc?export=download&id=1VC477JxlflH_IVbxda6PZh1ItR7yBAFQ'
    response = requests.get(model_url)
    response.raise_for_status()
    model = joblib.load(BytesIO(response.content))
    return model

# Load data and model
df = load_data()
model = load_model()

# Streamlit app code continues from here
st.title("**Instant Car Value Checker⚡**")
st.caption("*-by Md Faisal*")
st.caption("*Welcome to the instant car value checker app! This tool helps you estimate the value of your used car based on various factors.*")

# Sidebar
st.sidebar.header("Connect with me")
st.sidebar.markdown("""
<a href="https://www.linkedin.com/in/md-fsl" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" height="30">
</a>
<a href="https://github.com/Muhammed-Faisal" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" height="30">
</a>
<a href="https://www.kaggle.com/mdfaisal1" target="_blank">
<img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" width="30" height="30">
</a>
""", unsafe_allow_html=True)

st.sidebar.write("**Have any suggestions?**")
st.sidebar.write("*Please do let me know at mdf1234786143@gmail.com*")

# Initialize session state
if 'show_inputs' not in st.session_state:
    st.session_state.show_inputs = False

if 'inputs' not in st.session_state:
    st.session_state.inputs = {
        'brand': '',
        'name': '',
        'fuel_type': '',
        'transmission': '',
        'owner_type': '',
        'engine': 0,
        'kilometers_driven': 0
    }

def clear_inputs():
    for key in st.session_state.inputs.keys():
        st.session_state.inputs[key] = ''

if st.button("Enter Car Details"):
    st.session_state.show_inputs = True

if st.session_state.show_inputs:
    st.session_state.inputs['brand'] = st.selectbox('Brand : ', df['Brand'].sort_values().unique(), index=df['Brand'].sort_values().unique().tolist().index(st.session_state.inputs['brand']) if st.session_state.inputs['brand'] else 0, key='brand')
    name_options = df[df['Brand'] == st.session_state.inputs['brand']]['Name'].sort_values().unique() if st.session_state.inputs['brand'] else []
    st.session_state.inputs['name'] = st.selectbox('Model : ', name_options, index=name_options.tolist().index(st.session_state.inputs['name']) if st.session_state.inputs['name'] in name_options else 0, key='name')
    st.session_state.inputs['fuel_type'] = st.selectbox('Fuel Type : ', df['Fuel_Type'].sort_values().unique(), index=df['Fuel_Type'].sort_values().unique().tolist().index(st.session_state.inputs['fuel_type']) if st.session_state.inputs['fuel_type'] else 0)
    st.session_state.inputs['transmission'] = st.selectbox('Transmission : ', df['Transmission'].sort_values().unique(), index=df['Transmission'].sort_values().unique().tolist().index(st.session_state.inputs['transmission']) if st.session_state.inputs['transmission'] else 0)
    st.session_state.inputs['owner_type'] = st.selectbox('Owner Type : ', ['First', 'Second', 'Third', 'Fourth & Above'], index=['First', 'Second', 'Third', 'Fourth & Above'].index(st.session_state.inputs['owner_type']) if st.session_state.inputs['owner_type'] else 0)
    st.session_state.inputs['engine'] = st.number_input('Engine (cc) : ', min_value=int(df['Engine'].min()), max_value=int(df['Engine'].max()), value=int(st.session_state.inputs['engine']) if st.session_state.inputs['engine'] else int(df['Engine'].mean()))
    st.session_state.inputs['kilometers_driven'] = st.slider('Kilometers Driven : ', min_value=int(df['Kilometers_Driven'].min()), max_value=int(df['Kilometers_Driven'].max()), value=int(st.session_state.inputs['kilometers_driven']) if st.session_state.inputs['kilometers_driven'] else int(np.mean(df['Kilometers_Driven'])))

    if st.button("Know Your Car's Worth") and all(st.session_state.inputs.values()):
        input_data = pd.DataFrame({
            'Brand': [st.session_state.inputs['brand']],
            'Name': [st.session_state.inputs['name']],
            'Fuel_Type': [st.session_state.inputs['fuel_type']],
            'Transmission': [st.session_state.inputs['transmission']],
            'Owner_Type': [st.session_state.inputs['owner_type']],
            'Engine': [st.session_state.inputs['engine']],
            'Kilometers_Driven': [st.session_state.inputs['kilometers_driven']]
        })
        output = np.round(model.predict(input_data)[0], 2)
        st.write(f"You can sell this car for approximately ₹{output} lakhs.")
