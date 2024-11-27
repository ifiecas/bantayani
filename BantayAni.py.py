import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
url = 'https://raw.githubusercontent.com/ifiecas/simulated_cropdata/refs/heads/main/adjusted_farming_data.csv'
df = pd.read_csv(url)

# Data Cleaning: Handle Invalid and Extreme Values
df['Kabuuang_Ulan_mm'] = df['Kabuuang_Ulan_mm'].apply(lambda x: 0 if x < 0 else x)  # Replace negatives with 0
df['Kabuuang_Ulan_mm'] = df['Kabuuang_Ulan_mm'].clip(lower=0, upper=500)  # Clamp rainfall to 0–500 mm
df['Temperatura_Celsius'] = df['Temperatura_Celsius'].clip(lower=20, upper=40)  # Clamp temperature to 20–40°C

# Define features for predicting `Inaasahang Ani`
ani_features = [
    'Sukat_ng_Bukid_Ektarya', 'Planting_Month', 'Planting_Day', 
    'Kabuuang_Ulan_mm', 'Temperatura_Celsius'
]
X_ani = df[ani_features]
y_ani = df['Inaasahang_Ani_Sako']

# Train the Linear Regression model for `Inaasahang Ani`
X_ani_train, X_ani_test, y_ani_train, y_ani_test = train_test_split(X_ani, y_ani, test_size=0.2, random_state=42)
ani_model = LinearRegression()
ani_model.fit(X_ani_train, y_ani_train)

# Define features and target for predicting `Presyo kada Kilo`
price_features = ['Kabuuang_Ulan_mm', 'Temperatura_Celsius', 'Uri_ng_Palay_Freq', 'Bayan_Freq']
X_price = df[price_features]
y_price = df['Presyo_kada_Kilo']

# Train the Linear Regression model for `Presyo kada Kilo`
X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)
price_model = LinearRegression()
price_model.fit(X_price_train, y_price_train)

# Historical Weather Data (Example Averages for Each Planting Month)
weather_estimates = df.groupby('Planting_Month').agg({
    'Kabuuang_Ulan_mm': 'mean',
    'Temperatura_Celsius': 'mean'
}).to_dict(orient='index')

# Mappings for Uri ng Palay
uri_ng_palay_mapping = {
    913: "NSIC Rc438",
    930: "NSIC Rc354",
    947: "NSIC Rc238",
    948: "PSB Rc82",
    952: "NSIC Rc222",
    957: "NSIC Rc216",
    958: "NSIC Rc160",
    994: "NSIC Rc402"
}
uri_ng_palay_options = list(uri_ng_palay_mapping.values())

# Mappings for Bayan
bayan_mapping = {
    705: "Science City of Munoz",
    713: "Guimba",
    719: "Peñaranda",
    732: "Aliaga",
    744: "Zaragoza",
    745: "Gapan",
    794: "San Jose",
    852: "Cabanatuan",
    876: "Santa Rosa"
}
bayan_options = list(bayan_mapping.values())

# Streamlit App
st.title("🌾 BANTAY ANI: Gabay sa Panahon ng Pagtatanim 🌾")
st.write("Tulungan ang mga magsasaka na planuhin ang tamang araw ng pagtatanim para sa pinakamataas na ani.")

# Sidebar Instructions
st.sidebar.header("🌟 Mga Panuto")
st.sidebar.write("1. Piliin ang tamang petsa para sa pagtatanim.")
st.sidebar.write("2. I-adjust ang sukat ng iyong bukid kung kinakailangan.")
st.sidebar.write("3. Tingnan ang resulta ng iyong inaasahang ani, presyo, at kita.")
st.sidebar.markdown("---")

# Input Form
with st.form("prediction_form"):
    st.header("🔍 Detalye ng Pagtatanim at Sukat ng Bukid")

    # Date input for planting
    planting_date = st.date_input(
        "Petsa ng Pagtatanim:",
        value=pd.Timestamp("2025-01-01"),  # Default value
        min_value=pd.Timestamp("2025-01-01"),
        max_value=pd.Timestamp("2025-12-31")
    )
    planting_month = planting_date.month
    planting_day = planting_date.day

    # Input for Sukat ng Bukid
    field_size = st.number_input("Sukat ng Bukid (Ektarya):", min_value=0.1, step=0.1, value=1.0)

    # Dropdown for Uri ng Palay (Mapped)
    selected_uri_ng_palay = st.selectbox("Uri ng Palay:", options=uri_ng_palay_options)
    uri_ng_palay_freq = {v: k for k, v in uri_ng_palay_mapping.items()}[selected_uri_ng_palay]

    # Dropdown for Bayan (Mapped)
    selected_bayan = st.selectbox("Bayan:", options=bayan_options)
    bayan_freq = {v: k for k, v in bayan_mapping.items()}[selected_bayan]

    # Submit Button
    submitted = st.form_submit_button("📈 Alamin ang prediksyon ng iyong ani, presyo, at kita")

if submitted:
    # Retrieve estimated weather conditions based on planting month
    estimated_weather = weather_estimates.get(planting_month, {'Kabuuang_Ulan_mm': 300, 'Temperatura_Celsius': 27})
    rainfall = estimated_weather['Kabuuang_Ulan_mm']
    temperature = estimated_weather['Temperatura_Celsius']

    # Predict Inaasahang Ani with weather estimates
    ani_input = pd.DataFrame({
        'Sukat_ng_Bukid_Ektarya': [field_size],
        'Planting_Month': [planting_month],
        'Planting_Day': [planting_day],
        'Kabuuang_Ulan_mm': [rainfall],
        'Temperatura_Celsius': [temperature]
    })[ani_features]
    
    predicted_ani = ani_model.predict(ani_input)[0]

    # Adjust Inaasahang Ani based on weather extremes
    def calculate_adjustment(rainfall, temperature):
        rain_adjustment = 0
        temp_adjustment = 0

        # Rainfall adjustment
        if rainfall < 200 or rainfall > 400:
            rain_deviation = abs(rainfall - 300) / 100
            rain_adjustment = min(rain_deviation, 1) * np.random.uniform(0.1, 0.5)

        # Temperature adjustment
        if temperature < 25 or temperature > 30:
            temp_deviation = abs(temperature - 27.5) / 2.5
            temp_adjustment = min(temp_deviation, 1) * np.random.uniform(0.1, 0.5)

        # Combine adjustments
        return max(rain_adjustment, temp_adjustment)

    adjustment_percentage = calculate_adjustment(rainfall, temperature)
    adjusted_ani = predicted_ani * (1 - adjustment_percentage)

    # Predict Presyo kada Kilo
    price_input = pd.DataFrame({
        'Kabuuang_Ulan_mm': [rainfall],
        'Temperatura_Celsius': [temperature],
        'Uri_ng_Palay_Freq': [uri_ng_palay_freq],
        'Bayan_Freq': [bayan_freq]
    })[price_features]

    predicted_price = price_model.predict(price_input)[0]

    # Compute Inaasahang Kabuuang Kita
    total_income = adjusted_ani * 60 * predicted_price

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Display Results in Left Column
    with col1:
        st.header("📊 Resulta")
        st.success(f"🌾 Inaasahang Ani (Base): {predicted_ani:.2f} sako")
        st.success(f"🌾 Inaasahang Ani (Na-adjust): {adjusted_ani:.2f} sako")
        st.success(f"💰 Inaasahang Presyo kada Kilo: ₱{predicted_price:.2f}")
        st.success(f"💵 Inaasahang Kabuuang Kita: ₱{total_income:,.2f}")
        st.info(f"📉 Reduction due to forecasted weather extremes: {adjustment_percentage * 100:.2f}%")

    # Display Details of Features in Right Column
    with col2:
        st.header("🔍 Detalye")
        st.write(f"**Petsa ng Pagtatanim**: {planting_date}")
        st.write(f"**Sukat ng Bukid**: {field_size} ektarya")
        st.write(f"**Uri ng Palay**: {selected_uri_ng_palay}")
        st.write(f"**Bayan**: {selected_bayan}")
        st.write(f"**Kabuuang Ulan (mm) sa panahon ng anihan**: {rainfall}")
        st.write(f"**Temperatura (°C) sa panahon ng anihan**: {temperature}")
