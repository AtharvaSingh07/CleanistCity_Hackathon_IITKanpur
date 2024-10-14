import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from sklearnex import patch_sklearn
patch_sklearn()

# Load the dataset
data = pd.read_csv("C:/Users/Ajeet/BuzzEarthHack/Cleanest Cities Prediction/Cleanest_Cities_India.csv")

# API key for air quality
AIR_QUALITY_API_KEY = "2f667cec17cadcd56a29ea625750ae2c"

# Features definition
X = data[['2016_Score', '2017_Score', '2018_Score', '2019_Score_5000',
           '2020_Score_Max6000', '2022_Score_Max7500', '2023_Score_Max10000']]

# Calculate the target variable for 2024 prediction
data['2024_Score_Predicted'] = data['2023_Score_Max10000'] + (
        data['2023_Score_Max10000'] - data['2016_Score']) / 7  # Example logic

# Define the target variable
y = data['2024_Score_Predicted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize multiple models, including XGBoost
models = {
    'Gradient Boosting': GradientBoostingRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

# Train all models
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)

AQI_CATEGORIES = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

COMPONENT_DESCRIPTIONS = {
    'co': 'Carbon Monoxide',
    'no': 'Nitric Oxide',
    'no2': 'Nitrogen Dioxide',
    'o3': 'Ozone',
    'so2': 'Sulfur Dioxide',
    'pm2_5': 'Particulate Matter (PM2.5)',
    'pm10': 'Particulate Matter (PM10)',
    'nh3': 'Ammonia',
    'h2s': 'Hydrogen Sulfide',
    'c6h6': 'Benzene',
    'pbc': 'Lead',
    'ci': 'Chlorine'
}


# Function to fetch air quality data from the OpenWeatherMap API
def get_air_quality(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={AIR_QUALITY_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        air_quality_data = response.json()
        aqi = air_quality_data['list'][0]['main']['aqi']
        components = air_quality_data['list'][0]['components']
        return aqi, components
    return None, None


# Function to calculate confidence intervals using bootstrap sampling
def get_confidence_intervals(model, city_data_scaled, n_bootstrap=100, ci=0.95):
    bootstrap_preds = []
    for _ in range(n_bootstrap):
        X_resampled, y_resampled = resample(X_train_scaled, y_train, random_state=None)
        model.fit(X_resampled, y_resampled)
        bootstrap_pred = model.predict(city_data_scaled)
        bootstrap_preds.append(bootstrap_pred[0])

    lower_bound = np.percentile(bootstrap_preds, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrap_preds, (1 + ci) / 2 * 100)
    mean_pred = np.mean(bootstrap_preds)

    return mean_pred, lower_bound, upper_bound


# Function to calculate trend
def calculate_trend(city):
    city_data = data[data['City Name'] == city]
    if city_data.empty:
        return None

    years = ['2016_Score', '2017_Score', '2018_Score', '2019_Score_5000',
             '2020_Score_Max6000', '2022_Score_Max7500', '2023_Score_Max10000']
    scores = city_data[years].values.flatten()
    trend = np.polyfit(range(len(scores)), scores, 1)
    return trend[0]


# Function to plot trend
def plot_trend(city):
    city_data = data[data['City Name'] == city]
    if city_data.empty:
        return None

    years = ['2016_Score', '2017_Score', '2018_Score', '2019_Score_5000',
             '2020_Score_Max6000', '2022_Score_Max7500', '2023_Score_Max10000']
    formatted_labels = ['2016 Score', '2017 Score', '2018 Score', '2019 Score',
                        '2020 Score', '2022 Score', '2023 Score']
    scores = city_data[years].values.flatten()

    comparison_data = pd.DataFrame({
        'Year': formatted_labels,
        'Cleanliness Score': scores
    })

    # Line chart for trend analysis
    fig = px.line(comparison_data, x='Year', y='Cleanliness Score',
                  title=f'Cleanliness Score Trend for {city}',
                  labels={'Cleanliness Score': 'Cleanliness Score', 'Year': 'Year'},
                  markers=True)

    # Show the line chart in Streamlit
    st.plotly_chart(fig)

    # Bar chart with error bars (optional)
    bar_fig = px.bar(comparison_data, x='Year', y='Cleanliness Score',
                     title=f'Cleanliness Score Comparison for {city}',
                     labels={'Cleanliness Score': 'Cleanliness Score', 'Year': 'Year'},
                     error_y=[0] * len(scores))  # Add error bars if needed
    st.plotly_chart(bar_fig)


# Function to compare scores between two cities
def compare_scores(city1, city2):
    years = ['2016_Score', '2017_Score', '2018_Score', '2019_Score_5000',
             '2020_Score_Max6000', '2022_Score_Max7500', '2023_Score_Max10000']
    formatted_labels = ['2016 Score', '2017 Score', '2018 Score', '2019 Score',
                        '2020 Score', '2022 Score', '2023 Score']

    scores1 = data[data['City Name'] == city1][years].values.flatten()
    scores2 = data[data['City Name'] == city2][years].values.flatten()

    comparison_df = pd.DataFrame({
        'Year': formatted_labels,
        'City 1': scores1,
        'City 2': scores2
    })

    # Create a bar chart for comparison
    fig = px.bar(comparison_df, x='Year', y=['City 1', 'City 2'],
                 title='Cleanliness Score Comparison',
                 labels={'value': 'Cleanliness Score', 'variable': 'City'},
                 barmode='group',
                 height=400)

    # Show the bar chart in Streamlit
    st.plotly_chart(fig)


# Main function to run the Streamlit app
def run():
    # Streamlit application layout
    st.header("Cleanliness Score Prediction", divider="blue")
    st.write("Select two cities and a model to predict the cleanliness score for 2024, along with trend analysis.")

    st.sidebar.header("Additional Information", divider="blue")
    st.sidebar.write(
        "You can find additional information about air quality, cleanliness scores and trend analysis below:")
    st.sidebar.write("1. *Air Quality Index (AQI)* categories help in understanding the air pollution level.")
    st.sidebar.write("2. *Cleanliness Scores* are based on various parameters evaluated over the years.")
    st.sidebar.write("3. *Trend Analysis* shows the progress or decline in cleanliness over the years.")

    # Additional information about air quality indicators
    st.sidebar.subheader("Information about air quality components")
    st.sidebar.write("Here you can find details about the air quality indicators in μg/m3:")
    st.sidebar.write("1. *Carbon Monoxide (CO)*: A colorless, odorless gas produced by burning fossil fuels.")
    st.sidebar.write("2. *Nitric Oxide (NO)*: A gas that contributes to air pollution and smog formation.")
    st.sidebar.write(
        "3. *Nitrogen Dioxide (NO2)*: A toxic gas that can irritate airways and contribute to respiratory problems.")
    st.sidebar.write("4. *Ozone (O3)*: A reactive gas that can cause respiratory issues, especially in urban areas.")
    st.sidebar.write(
        "5. *Sulfur Dioxide (SO2)*: A gas produced by volcanic eruptions and industrial processes, leading to acid rain.")
    st.sidebar.write(
        "6. *Particulate Matter (PM2.5)*: Fine particles that can penetrate the lungs and enter the bloodstream.")
    st.sidebar.write("7. *Particulate Matter (PM10)*: Coarser particles that can affect respiratory health.")
    st.sidebar.write("8. *Ammonia (NH3)*: A compound that can contribute to the formation of particulate matter.")
    st.sidebar.write("9. *Trend Analysis* shows the progress or decline in cleanliness over the years.")

    # Dropdowns for city selection
    city1 = st.selectbox("Select City 1", data['City Name'].unique())
    city2 = st.selectbox("Select City 2", data['City Name'].unique())

    # Dropdown for model selection
    model_selection = st.selectbox("Select Model", list(models.keys()))

    # Button to trigger prediction
    if st.button("Predict"):
        if city1 not in data['City Name'].values or city2 not in data['City Name'].values:
            st.error("One or both cities not found.")
        else:
            progress_bar = st.progress(0, text="Please wait...")
            time.sleep(0.5)

            # Step 1: Fetch data for selected cities
            city_data1 = data[data['City Name'] == city1]
            city_data2 = data[data['City Name'] == city2]

            lat1, lon1 = city_data1.iloc[0]['lat'], city_data1.iloc[0]['lon']
            lat2, lon2 = city_data2.iloc[0]['lat'], city_data2.iloc[0]['lon']

            progress_bar.progress(20, text="Fetching air quality data...")

            # Step 2: Get air quality data
            aqi1, components1 = get_air_quality(lat1, lon1)
            aqi2, components2 = get_air_quality(lat2, lon2)

            progress_bar.progress(40, text="Scaling data...")

            # Step 3: Scale the city data
            scores1 = city_data1[['2016_Score', '2017_Score', '2018_Score',
                                  '2019_Score_5000', '2020_Score_Max6000',
                                  '2022_Score_Max7500', '2023_Score_Max10000']].values.flatten()
            scores2 = city_data2[['2016_Score', '2017_Score', '2018_Score',
                                  '2019_Score_5000', '2020_Score_Max6000',
                                  '2022_Score_Max7500', '2023_Score_Max10000']].values.flatten()

            city_data1_scaled = scaler.transform([scores1])
            city_data2_scaled = scaler.transform([scores2])

            progress_bar.progress(60, text="Making predictions...")
            time.sleep(0.5)

            # Step 4: Make predictions
            model = models[model_selection]
            mean_pred1, lower_bound1, upper_bound1 = get_confidence_intervals(model, city_data1_scaled)
            mean_pred2, lower_bound2, upper_bound2 = get_confidence_intervals(model, city_data2_scaled)

            accuracy1 = r2_score(y, model.predict(scaler.transform(X)))
            accuracy2 = accuracy1

            progress_bar.progress(80, text="Preparing results...")
            time.sleep(0.5)

            aqi_category1 = AQI_CATEGORIES.get(aqi1, "Unknown")
            aqi_category2 = AQI_CATEGORIES.get(aqi2, "Unknown")

            # Step 5: Display results
            st.success(f"{city1} Prediction: {mean_pred1:.2f} (95% CI: {lower_bound1:.2f} - {upper_bound1:.2f})")
            st.success(f"{city2} Prediction: {mean_pred2:.2f} (95% CI: {lower_bound2:.2f} - {upper_bound2:.2f})")
            st.write(f"Model Accuracy City 1: {accuracy1:.2f}")
            st.write(f"Model Accuracy City 2: {accuracy2:.2f}")
            st.write(f"AQI City 1: {aqi1} ({aqi_category1})")
            st.write(f"AQI City 2: {aqi2} ({aqi_category2})")

            progress_bar.progress(100, text="Prediction complete.")
            time.sleep(0.5)

            # Step 6: Display trend analysis and comparisons
            st.subheader(f"Trend Analysis for {city1}")
            plot_trend(city1)

            st.subheader(f"Trend Analysis for {city2}")
            plot_trend(city2)

            st.subheader("Comparison of Cleanliness Scores")
            compare_scores(city1, city2)

            # Create full names for components using COMPONENT_DESCRIPTIONS
            components1_full = {key: value for key, value in components1.items()}
            components2_full = {key: value for key, value in components2.items()}

            # Create DataFrames from components dictionaries
            components1_df = pd.DataFrame.from_dict(components1_full, orient='index', columns=['Value'])
            components2_df = pd.DataFrame.from_dict(components2_full, orient='index', columns=['Value'])

            # Display components from API
            st.write(f"Air Quality Components for {city1}:")
            st.dataframe(components1_df.rename(
                index={key: f"{COMPONENT_DESCRIPTIONS.get(key, key.title())} ({key})" for key in components1_df.index}))
            st.write(f"Air Quality Components for {city2}:")
            st.dataframe(components2_df.rename(
                index={key: f"{COMPONENT_DESCRIPTIONS.get(key, key.title())} ({key})" for key in components2_df.index}))


# Run the app
if __name__ == '__main__':
    run()
