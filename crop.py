import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.header("Crop and Fertilizer Recommendation System")
# Load your dataset
try:
    crop = pd.read_csv("data/Crop_recommendation.csv")
except Exception as e:
    st.error(f"Error loading dataset: {e}")

# Ensure the dataset is loaded properly
if crop.empty:
    st.error("Failed to load the dataset.")
else:
    st.write("Dataset loaded successfully")
    st.write(crop.head())  # Display a preview of the dataset

# Initialize session state for activity and header display
if 'activity' not in st.session_state:
    st.session_state.activity = "Overview"

# Activity selection with radio buttons
activity = st.radio(
    "Select an activity:", 
    ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"],
    index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"].index(st.session_state.activity)
)

# Debugging: Ensure session state is correctly updated
st.session_state.activity = activity
st.write(f"Current activity: {st.session_state.activity}")

# --- Overview Section ---
if st.session_state.activity == 'Overview':
    st.header("Overview of the Dataset")
    
    # Dataset Overview
    if st.checkbox('Show dataset preview'):
        st.write(crop.head())

    # Show dataset info and missing values
    if st.checkbox('Show dataset information'):
        st.write(crop.info())

    if st.checkbox('Show missing values'):
        st.write(crop.isnull().sum())

    if st.checkbox('Show duplicate values'):
        st.write(crop.duplicated().sum())

# --- Dataset Statistics Section ---
elif st.session_state.activity == 'Dataset Statistics':
    st.header("Dataset Statistics")

    # Statistics summary
    if st.checkbox('Show dataset statistics'):
        st.write(crop.describe())

    # Target feature distribution
    if st.checkbox('Show target feature distribution'):
        st.write(crop['label'].value_counts())

    # List features excluding target variable 'label'
    st.write("Features: ", crop.columns.tolist())

    # Correlation heatmap
    if st.checkbox('Show correlation heatmap'):
        num_cols = crop.select_dtypes(include=[np.number])
        corr = num_cols.corr()
        st.write(corr)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
        st.pyplot(plt)

# --- Visualize Features Section ---
elif st.session_state.activity == 'Visualize Features':
    st.header("Visualize Features")

    # Feature selection
    feature = st.selectbox('Select a feature to visualize', crop.columns.tolist())

    # Check if the feature exists in the data before plotting
    if feature in crop.columns:
        st.write(f"### Distribution of {feature}")
        plt.figure(figsize=(8, 6))
        sns.histplot(crop[feature], kde=True, bins=20)
        st.pyplot(plt)
    else:
        st.write(f"Feature '{feature}' not found in the dataset.")

# --- Filter and Analyze Crop Section ---
elif st.session_state.activity == 'Filter and Analyze Crop':
    st.header("Filter and Analyze Crop Data")

    # Filtering options
    filter_column = st.selectbox('Select a column to filter', crop.columns.tolist())
    filter_value = st.text_input(f'Enter value for {filter_column} to filter')

    # Apply filter
    if filter_value:
        filtered_data = crop[crop[filter_column].astype(str).str.contains(filter_value, case=False, na=False)]
        st.write(f"Filtered Data based on {filter_column} containing '{filter_value}':")
        st.write(filtered_data)
    else:
        st.write("No filter applied. Please enter a value to filter the data.")

# --- Download Dataset Section ---
elif st.session_state.activity == 'Download Dataset':
    st.header("Download the Processed Dataset")

    # Provide a download button
    st.write("Click the button below to download the dataset.")
    csv = crop.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='crop_recommendation_processed.csv',
        mime='text/csv'
    )

# --- Advanced Features Section ---
elif st.session_state.activity == 'Advanced Features':
    st.header("Advanced Features Section")

    # Input sliders for NPK, Temperature, and Humidity
    nitrogen = st.slider("Select Nitrogen Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
    phosphorus = st.slider("Select Phosphorus Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
    potassium = st.slider("Select Potassium Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
    
    temperature = st.slider("Select Temperature (°C)", min_value=15, max_value=40, step=1)
    humidity = st.slider("Select Humidity (%)", min_value=40, max_value=100, step=1)

    # Fertilizer recommendation logic based on input values
    if nitrogen > 0 and phosphorus > 0 and potassium > 0:
        fertilizer_recommendation = "Use a balanced NPK fertilizer (15-15-15)."
    elif nitrogen > 0 and phosphorus > 0:
        fertilizer_recommendation = "Use a Nitrogen-Phosphorus fertilizer."
    elif potassium > 0:
        fertilizer_recommendation = "Use a Potassium-based fertilizer."
    else:
        fertilizer_recommendation = "Use a general-purpose fertilizer."

    # Crop yield prediction (simple mock-up; in real-life, this would involve a trained ML model)
    predicted_yield = (nitrogen + phosphorus + potassium) * 0.1 + (temperature - 15) * 2 + (humidity - 40) * 0.5
    predicted_yield = max(0, predicted_yield)  # Ensuring yield can't be negative

    # Display results
    st.write(f"**Recommended Fertilizer:** {fertilizer_recommendation}")
    st.write(f"**Predicted Crop Yield:** {predicted_yield:.2f} units")

    # Visualization of NPK Levels and Temperature vs Yield (Mock-up visualization)
    st.write("### Visualizing the NPK and Temperature effects on Yield")
    data = pd.DataFrame({
        "Nitrogen": [nitrogen],
        "Phosphorus": [phosphorus],
        "Potassium": [potassium],
        "Temperature": [temperature],
        "Yield": [predicted_yield]
    })
    
    fig, ax = plt.subplots()
    ax.bar(data.columns[:-1], data.iloc[0, :-1], color=["#FF6347", "#FFD700", "#8A2BE2", "#4CAF50"])
    ax.set_title("Impact of NPK and Temperature on Crop Yield")
    st.pyplot(fig)
