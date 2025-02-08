# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# import ipywidgets as widgets
# from IPython.display import display

# # Load dataset
# crop = pd.read_csv("data/Crop_recommendation.csv")

# # Streamlit UI Setup
# st.title("Crop and Fertile System")

# # Check if 'label' column exists
# if 'label' not in crop.columns:
#     st.error("The 'label' column is missing from the dataset. Please check your CSV file.")
#     st.stop()  # Stop the app if the column is missing

# # Print column names to debug
# st.write("Columns in the dataset:", crop.columns)

# # List of features excluding the 'label' column (this list will be updated dynamically)
# features = crop.columns.to_list()
# features.remove('label')

# # Initialize session state for activity if it doesn't exist
# if 'activity' not in st.session_state:
#     st.session_state.activity = "Overview"  # Default to "Overview" on first load

# # Activity selection with radio buttons
# st.session_state.activity = st.radio(
#     "Select an activity:", 
#     ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset"],
#     index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset"].index(st.session_state.activity)
# )

# # Overview Section
# if st.session_state.activity == 'Overview':
#     st.header("Overview of the Dataset")
    
#     # Dataset Overview
#     if st.checkbox('Show dataset preview'):
#         st.write(crop.head())

#     # Show dataset info and missing values
#     if st.checkbox('Show dataset information'):
#         st.write(crop.info())

#     if st.checkbox('Show missing values'):
#         st.write(crop.isnull().sum())

#     if st.checkbox('Show duplicate values'):
#         st.write(crop.duplicated().sum())

# # Dataset Statistics Section
# elif st.session_state.activity == 'Dataset Statistics':
#     st.header("Dataset Statistics")

#     # Statistics summary
#     if st.checkbox('Show dataset statistics'):
#         st.write(crop.describe())

#     # Target feature distribution
#     if st.checkbox('Show target feature distribution'):
#         st.write(crop['label'].value_counts())

#     # List features excluding target variable 'label'
#     st.write("Features: ", features)

#     # Correlation heatmap
#     if st.checkbox('Show correlation heatmap'):
#         num_cols = crop.select_dtypes(include=[np.number])
#         corr = num_cols.corr()
#         st.write(corr)
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
#         st.pyplot(fig)

# # Visualize Features Section
# elif st.session_state.activity == 'Visualize Features':
#     st.header("Visualize Features")

#     # Buttons for different types of plots
#     if st.button("Show Histplot"):
#         feature = st.selectbox('Select feature for histplot:', features)
#         fig, ax = plt.subplots()
#         sns.histplot(crop[feature], ax=ax, kde=True, color='skyblue')
#         st.pyplot(fig)

#     if st.button("Show Scatterplot"):
#         feature_x = st.selectbox('Select X feature for scatterplot:', features)
#         feature_y = st.selectbox('Select Y feature for scatterplot:', features)
#         fig, ax = plt.subplots()
#         sns.scatterplot(x=crop[feature_x], y=crop[feature_y], ax=ax, color='salmon')
#         st.pyplot(fig)

#     if st.button("Show Boxplot"):
#         feature = st.selectbox('Select feature for boxplot:', features)
#         fig, ax = plt.subplots()
#         sns.boxplot(x=crop[feature], ax=ax, color='lightgreen')
#         st.pyplot(fig)

#     if st.button("Show Heatmap"):
#         num_cols = crop.select_dtypes(include=[np.number])
#         corr = num_cols.corr()
#         st.write(corr)
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
#         st.pyplot(fig)

#     if st.button("Show Violinplot"):
#         feature = st.selectbox('Select feature for violinplot:', features)
#         fig, ax = plt.subplots()
#         sns.violinplot(x=crop['label'], y=crop[feature], ax=ax, palette='muted')
#         st.pyplot(fig)

#     if st.button("Show Barplot"):
#         feature = st.selectbox('Select feature for barplot:', features)
#         fig, ax = plt.subplots()
#         sns.barplot(x=crop['label'], y=crop[feature], ax=ax, palette='Set2')
#         st.pyplot(fig)

# # Filter and Analyze Crop Section
# elif st.session_state.activity == 'Filter and Analyze Crop':
#     st.header("Filter and Analyze Crop Data")
    
#     # Encoding the 'label' column (before dropping 'label')
#     if st.checkbox('Show encoded target labels'):
#         crop_dict = {
#             'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
#             'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
#             'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
#             'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
#             'jute': 21, 'coffee': 22
#         }
#         crop['crop_no'] = crop['label'].map(crop_dict)
#         st.write(crop.head())

#     # Filter dataset based on user input (after encoding the 'label' column)
#     crop_choice = st.selectbox('Select a crop to analyze:', crop['label'].unique())

#     # Filter the dataset for the selected crop
#     filtered_crop = crop[crop['label'] == crop_choice]

#     # Display the filtered dataset
#     st.write(f"Filtered dataset for {crop_choice}:")
#     st.write(filtered_crop.head())

#     # User input for plot type
#     plot_type = st.selectbox('Select plot type:', ['Histogram', 'Boxplot', 'Scatterplot'])

#     # Display plots based on selection
#     if plot_type == 'Histogram':
#         feature = st.selectbox('Select feature for histogram:', features)
#         fig, ax = plt.subplots()
#         sns.histplot(crop[feature], ax=ax, kde=True, color='coral')
#         st.pyplot(fig)

#     elif plot_type == 'Boxplot':
#         feature = st.selectbox('Select feature for boxplot:', features)
#         fig, ax = plt.subplots()
#         sns.boxplot(x=crop[feature], ax=ax, color='plum')
#         st.pyplot(fig)

#     elif plot_type == 'Scatterplot':
#         feature_x = st.selectbox('Select X feature for scatterplot:', features)
#         feature_y = st.selectbox('Select Y feature for scatterplot:', features)
#         fig, ax = plt.subplots()
#         sns.scatterplot(x=crop[feature_x], y=crop[feature_y], ax=ax, color='mediumvioletred')
#         st.pyplot(fig)

# # Download Dataset Section
# elif st.session_state.activity == 'Download Dataset':
#     st.header("Download the Processed Dataset")
    
#     # Now drop the 'label' column (after it is no longer needed)
#     crop.drop('label', axis=1, inplace=True)

#     # Final Dataset
#     if st.checkbox('Show final dataset after dropping label'):
#         st.write(crop.tail())

#     # Function to convert dataframe to CSV for download
#     def convert_df_to_csv(df):
#         return df.to_csv(index=False).encode('utf-8')

#     # Add download button
#     if st.checkbox('Download final dataset'):
#         final_csv = convert_df_to_csv(crop)
#         st.download_button(
#             label="Download CSV",
#             data=final_csv,
#             file_name='final_crop_dataset.csv',
#             mime='text/csv'
#         )








# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.preprocessing import LabelEncoder
# import time
# from sklearn.model_selection import GridSearchCV

# # Load dataset
# crop = pd.read_csv("data/Crop_recommendation.csv")

# # Streamlit UI Setup
# st.set_page_config(page_title="Crop and Fertile System", page_icon="🌾", layout="wide")

# # Custom styling for the page (Black background with contrast colors)
# st.markdown("""
#     <style>
#     body {
#         background: linear-gradient(135deg, #333, #555);
#         color: white;
#         font-family: 'Arial', sans-serif;
#     }
#     .stButton>button {
#         background: linear-gradient(45deg, #32CD32, #3CB371, #2E8B57);  /* Green gradient */
#         color: white;
#         font-size: 18px;
#         border: none;
#         padding: 20px 40px;
#         border-radius: 6px;
#         cursor: pointer;
#         transition: all 0.3s ease-in-out;
#     }
#     .stButton>button:hover {
#         background-color: #228B22;  /* Darker green on hover */
#         transform: scale(1.05);
#     }
#     .stSelectbox, .stRadio {
#         width: 100%;
#         padding: 10px;
#         font-size: 16px;
#         border-radius: 6px;
#     }
#     .stTextInput>input {
#         background-color: #333;
#         color: white;
#         border: 1px solid #666;
#     }
#     .stTable th, .stTable td {
#         background-color: #444;
#     }
#     .stSidebar {
#         background: linear-gradient(135deg, #444, #666);
#         padding: 10px;
#     }
#     .stMarkdown {
#         font-size: 20px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Custom Title with Green color using markdown
# st.markdown("<h1 style='color: #32CD32; font-size: 40px; font-weight: bold;'>Crop and Fertile System 🌾</h1>", unsafe_allow_html=True)

# # Initializing options for the app
# if 'activity' not in st.session_state:
#     st.session_state.activity = "Overview"  # Default to "Overview" on first load

# # Activity selection with radio buttons (all options visible)
# st.session_state.activity = st.radio(
#     "Select an activity:",
#     ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Predict Crop", "Download Dataset", "Advanced Features"],
#     index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Predict Crop", "Download Dataset", "Advanced Features"].index(st.session_state.activity),
#     label_visibility="collapsed"
# )

# # Split Layout for better UX
# left_column, right_column = st.columns([1, 2])  # Adjust the width of columns

# # --- Overview Section ---
# if st.session_state.activity == 'Overview':
#     left_column.header("Overview of the Dataset")
    
#     if left_column.checkbox('Show dataset preview'):
#         left_column.write(crop.head())

#     if left_column.checkbox('Show dataset information'):
#         left_column.write(crop.info())

#     if left_column.checkbox('Show missing values'):
#         left_column.write(crop.isnull().sum())

#     if left_column.checkbox('Show duplicate values'):
#         left_column.write(crop.duplicated().sum())

# # --- Dataset Statistics Section ---
# elif st.session_state.activity == 'Dataset Statistics':
#     left_column.header("Dataset Statistics")
    
#     if left_column.checkbox('Show dataset statistics'):
#         left_column.write(crop.describe())

#     if left_column.checkbox('Show target feature distribution'):
#         left_column.write(crop['label'].value_counts())

#     if left_column.checkbox('Show correlation heatmap'):
#         num_cols = crop.select_dtypes(include=[np.number])
#         corr = num_cols.corr()
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
#         right_column.pyplot(fig)

# # --- Visualize Features Section ---
# elif st.session_state.activity == 'Visualize Features':
#     left_column.header("Visualize Features")
    
#     # Buttons for different types of plots
#     plot_type = left_column.selectbox('Select plot type:', ['Histplot', 'Boxplot', 'Scatterplot', 'Heatmap', 'Pairplot'])
#     feature = left_column.selectbox('Select feature:', crop.columns)

#     # Example: Histplot
#     if plot_type == 'Histplot':
#         fig, ax = plt.subplots()
#         sns.histplot(crop[feature], ax=ax, kde=True, color='skyblue')
#         right_column.pyplot(fig)
    
#     # Example: Scatterplot
#     if plot_type == 'Scatterplot':
#         feature_y = left_column.selectbox('Select Y feature for scatterplot:', crop.columns)
#         fig, ax = plt.subplots()
#         sns.scatterplot(x=crop[feature], y=crop[feature_y], ax=ax, color='salmon')
#         right_column.pyplot(fig)

#     # Example: Boxplot
#     if plot_type == 'Boxplot':
#         fig, ax = plt.subplots()
#         sns.boxplot(x=crop[feature], ax=ax, color='lightgreen')
#         right_column.pyplot(fig)

#     # Example: Pairplot (for multiple features)
#     if plot_type == 'Pairplot':
#         fig = sns.pairplot(crop[crop.columns[:5]])  # Just show a subset of features for simplicity
#         right_column.pyplot(fig)

# # --- Predict Crop Section ---
# elif st.session_state.activity == 'Predict Crop':
#     left_column.header("Predict Crop Based on Input")
    
#     # Machine Learning Model Setup
#     X = crop.drop(columns=["label"])
#     y = crop["label"]
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # User selects model
#     model_type = left_column.selectbox("Choose Model", ["Random Forest", "SVM"])
#     if model_type == "Random Forest":
#         model = RandomForestClassifier()
#     elif model_type == "SVM":
#         model = SVC()
    
#     # Train model
#     model.fit(X_train, y_train)
    
#     # Prediction and evaluation
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     left_column.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     cm_display = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
#     cm_display.plot(cmap='Blues')
#     right_column.pyplot()
    
#     # User input for prediction
#     user_input = left_column.text_input('Enter feature values comma-separated').split(',')
#     if user_input:
#         user_input = [float(x) for x in user_input]
#         prediction = model.predict([user_input])
#         right_column.write(f"Predicted Crop: {prediction[0]}")

# # --- Advanced Features Section ---
# elif st.session_state.activity == 'Advanced Features':
#     left_column.header("Advanced Machine Learning Features")
    
#     # Hyperparameter tuning with GridSearchCV
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10]
#     }
    
#     if left_column.button("Tune Model with GridSearchCV"):
#         grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
#         grid_search.fit(X_train, y_train)
#         left_column.write(f"Best Parameters: {grid_search.best_params_}")
#         left_column.write(f"Best Score: {grid_search.best_score_}")
    
# # --- Download Dataset Section ---
# elif st.session_state.activity == 'Download Dataset':
#     left_column.header("Download the Processed Dataset")
#     crop.drop('label', axis=1, inplace=True)

#     # Function to convert dataframe to CSV for download
#     def convert_df_to_csv(df):
#         return df.to_csv(index=False).encode('utf-8')

#     if left_column.checkbox('Download final dataset'):
#         final_csv = convert_df_to_csv(crop)
#         left_column.download_button(
#             label="Download CSV",
#             data=final_csv,
#             file_name='final_crop_dataset.csv',
#             mime='text/csv'
#         )













# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st

# # Load dataset from the 'data/' folder
# crop = pd.read_csv("data/Crop_recommendation.csv")  # Update the path if needed

# # Streamlit UI Setup
# st.title("Crop and Fertile System")

# # Ensure 'label' column exists
# if 'label' not in crop.columns:
#     st.error("The 'label' column is missing from the dataset. Please check your CSV file.")
#     st.stop()  # Stop execution if the column is missing

# # List of features excluding the 'label' column
# features = crop.columns.to_list()
# features.remove('label')

# # Initialize session state for activity if it doesn't exist
# if 'activity' not in st.session_state:
#     st.session_state.activity = "Overview"  # Default to "Overview" on first load

# # Activity selection with radio buttons
# st.session_state.activity = st.radio(
#     "Select an activity:", 
#     ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"],
#     index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"].index(st.session_state.activity)
# )

# # Debugging statement to check activity state
# st.write(f"Selected activity: {st.session_state.activity}")

# # --- Overview Section ---
# if st.session_state.activity == 'Overview':
#     st.header("Overview of the Dataset")
    
#     # Dataset Overview
#     if st.checkbox('Show dataset preview'):
#         st.write(crop.head())

#     # Show dataset info and missing values
#     if st.checkbox('Show dataset information'):
#         st.write(crop.info())

#     if st.checkbox('Show missing values'):
#         st.write(crop.isnull().sum())

#     if st.checkbox('Show duplicate values'):
#         st.write(crop.duplicated().sum())

# # --- Dataset Statistics Section ---
# elif st.session_state.activity == 'Dataset Statistics':
#     st.header("Dataset Statistics")

#     # Statistics summary
#     if st.checkbox('Show dataset statistics'):
#         st.write(crop.describe())

#     # Target feature distribution
#     if st.checkbox('Show target feature distribution'):
#         st.write(crop['label'].value_counts())

#     # List features excluding target variable 'label'
#     st.write("Features: ", features)

#     # Correlation heatmap
#     if st.checkbox('Show correlation heatmap'):
#         num_cols = crop.select_dtypes(include=[np.number])
#         corr = num_cols.corr()
#         st.write(corr)
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
#         st.pyplot(plt)

# # --- Visualize Features Section ---
# elif st.session_state.activity == 'Visualize Features':
#     st.header("Visualize Features")

#     # Add a few plot options for demonstration
#     if st.button("Show Histplot"):
#         feature = st.selectbox('Select feature for histplot:', features)
#         fig, ax = plt.subplots()
#         sns.histplot(crop[feature], ax=ax, kde=True, color='skyblue')
#         st.pyplot(fig)

# # --- Filter and Analyze Crop Section ---
# elif st.session_state.activity == 'Filter and Analyze Crop':
#     st.header("Filter and Analyze Crop Data")

#     # Encoding the 'label' column (before dropping 'label')
#     if st.checkbox('Show encoded target labels'):
#         crop_dict = {
#             'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
#             'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
#             'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
#             'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
#             'jute': 21, 'coffee': 22
#         }
#         crop['crop_no'] = crop['label'].map(crop_dict)
#         st.write(crop.head())

#     # Filter dataset based on user input
#     nitrogen = st.slider("Select Nitrogen Level", 0.0, 200.0, 100.0)
#     phosphorus = st.slider("Select Phosphorus Level", 0.0, 200.0, 100.0)
#     potassium = st.slider("Select Potassium Level", 0.0, 200.0, 100.0)

#     # Apply filter on the dataset based on user input
#     filtered_data = crop[
#         (crop['N'] >= nitrogen - 10) & (crop['N'] <= nitrogen + 10) & 
#         (crop['P'] >= phosphorus - 10) & (crop['P'] <= phosphorus + 10) & 
#         (crop['K'] >= potassium - 10) & (crop['K'] <= potassium + 10)
#     ]

#     # Display the filtered data
#     if not filtered_data.empty:
#         st.write(f"Found {filtered_data.shape[0]} matching crops:")
#         st.dataframe(filtered_data)
#     else:
#         st.write("No crops found with the selected criteria.")

# # --- Download Dataset Section ---
# elif st.session_state.activity == 'Download Dataset':
#     st.header("Download the Processed Dataset")
    
#     # Drop the 'label' column
#     crop.drop('label', axis=1, inplace=True)

#     # Display the dataset after dropping 'label'
#     if st.checkbox('Show final dataset after dropping label'):
#         st.write(crop.tail())

#     # Function to convert dataframe to CSV for download
#     def convert_df_to_csv(df):
#         return df.to_csv(index=False).encode('utf-8')

#     if st.checkbox('Download final dataset'):
#         final_csv = convert_df_to_csv(crop)
#         st.download_button(
#             label="Download CSV",
#             data=final_csv,
#             file_name='final_crop_dataset.csv',
#             mime='text/csv'
#         )

# # --- Advanced Features Section ---
# elif st.session_state.activity == 'Advanced Features':
#     st.header("Advanced Features Section")
#     st.write("Here you can add additional advanced features as needed.")









# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# import time

# # Load dataset from the 'data/' folder
# crop = pd.read_csv("data/Crop_recommendation.csv")  # Update the path if needed

# # Streamlit UI Setup
# st.set_page_config(page_title="Crop and Fertilizer System", page_icon="🌱", layout="wide")

# # Dark theme setup
# st.markdown("""
#     <style>
#     body {
#         background-color: #121212;
#         color: white;
#     }
#     .stButton>button {
#         background-color: #1db954;
#         color: white;
#     }
#     .stTextInput>div>div>input {
#         background-color: #333333;
#         color: white;
#     }
#     .stSelectbox>div>div>input {
#         background-color: #333333;
#         color: white;
#     }
#     .stRadio>div>div>input {
#         background-color: #333333;
#         color: white;
#     }
#     .stCheckbox>div>div>input {
#         background-color: #333333;
#         color: white;
#     }
#     .stMarkdown {
#         color: white;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Ensure 'label' column exists
# if 'label' not in crop.columns:
#     st.error("The 'label' column is missing from the dataset. Please check your CSV file.")
#     st.stop()  # Stop execution if the column is missing

# # List of features excluding the 'label' column
# features = crop.columns.to_list()
# features.remove('label')

# # Initialize session state for activity if it doesn't exist
# if 'activity' not in st.session_state:
#     st.session_state.activity = "Overview"  # Default to "Overview" on first load

# # Initialize session state for header display if it doesn't exist
# if 'header_displayed' not in st.session_state:
#     st.session_state.header_displayed = False

# # Activity selection with radio buttons
# st.session_state.activity = st.radio(
#     "Select an activity:", 
#     ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"],
#     index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"].index(st.session_state.activity)
# )

# # Color changing header text
# def changing_header():
#     colors = ["#FF6347", "#1DB954", "#4CAF50", "#FFD700", "#8A2BE2", "#FF1493"]
#     for color in colors:
#         st.markdown(f"<h1 style='color:{color};'>Fertilizer and Crop Recommendation System</h1>", unsafe_allow_html=True)
#         # time.sleep(1)

# # --- Overview Section ---
# if st.session_state.activity == 'Overview':
#     # Check if the header has been displayed before
#     if not st.session_state.header_displayed:
#         changing_header()  # Display header with color change
#         st.session_state.header_displayed = True  # Mark the header as displayed
    
#     st.header("Overview of the Dataset")
    
#     # Dataset Overview
#     if st.checkbox('Show dataset preview'):
#         st.write(crop.head())

#     # Show dataset info and missing values
#     if st.checkbox('Show dataset information'):
#         st.write(crop.info())

#     if st.checkbox('Show missing values'):
#         st.write(crop.isnull().sum())

#     if st.checkbox('Show duplicate values'):
#         st.write(crop.duplicated().sum())

# # --- Dataset Statistics Section ---
# elif st.session_state.activity == 'Dataset Statistics':
#     # Check if the header has been displayed before
#     if not st.session_state.header_displayed:
#         changing_header()  # Display header with color change
#         st.session_state.header_displayed = True  # Mark the header as displayed
    
#     st.header("Dataset Statistics")

#     # Statistics summary
#     if st.checkbox('Show dataset statistics'):
#         st.write(crop.describe())

#     # Target feature distribution
#     if st.checkbox('Show target feature distribution'):
#         st.write(crop['label'].value_counts())

#     # List features excluding target variable 'label'
#     st.write("Features: ", features)

#     # Correlation heatmap
#     if st.checkbox('Show correlation heatmap'):
#         num_cols = crop.select_dtypes(include=[np.number])
#         corr = num_cols.corr()
#         st.write(corr)
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
#         st.pyplot(plt)

# # --- Visualize Features Section ---
# elif st.session_state.activity == 'Visualize Features':
#     # Check if the header has been displayed before
#     if not st.session_state.header_displayed:
#         changing_header()  # Display header with color change
#         st.session_state.header_displayed = True  # Mark the header as displayed
    
#     st.header("Visualize Features")

#     # Select plot type and feature
#     plot_type = st.selectbox("Select plot type", ["Bar Plot", "Line Plot", "Box Plot", "Scatter Plot", "Pairplot", "Heatmap"])
#     feature = st.selectbox('Select feature to visualize:', features)

#     # Show corresponding plot based on the selected plot type
#     if plot_type == "Bar Plot":
#         st.write(f"Bar plot of {feature}")
#         fig, ax = plt.subplots()
#         crop[feature].value_counts().plot(kind='bar', ax=ax, color='skyblue')
#         st.pyplot(fig)

#     elif plot_type == "Line Plot":
#         st.write(f"Line plot of {feature}")
#         fig, ax = plt.subplots()
#         crop[feature].plot(kind='line', ax=ax, color='green')
#         st.pyplot(fig)

#     elif plot_type == "Box Plot":
#         st.write(f"Box plot of {feature}")
#         fig, ax = plt.subplots()
#         sns.boxplot(x=crop[feature], ax=ax, color='orange')
#         st.pyplot(fig)

#     elif plot_type == "Scatter Plot":
#         scatter_feature = st.selectbox('Select another feature for scatter plot:', features)
#         st.write(f"Scatter plot of {feature} vs {scatter_feature}")
#         fig, ax = plt.subplots()
#         ax.scatter(crop[feature], crop[scatter_feature], color='purple')
#         st.pyplot(fig)

#     elif plot_type == "Pairplot":
#         st.write(f"Pairplot of {feature}")
#         fig = sns.pairplot(crop[features], hue="label")
#         st.pyplot(fig)

#     elif plot_type == "Heatmap":
#         st.write(f"Heatmap of correlation")
#         corr = crop[features].corr()
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)

# # --- Filter and Analyze Crop Section ---
# elif st.session_state.activity == 'Filter and Analyze Crop':
#     # Check if the header has been displayed before
#     if not st.session_state.header_displayed:
#         changing_header()  # Display header with color change
#         st.session_state.header_displayed = True  # Mark the header as displayed
    
#     st.header("Filter and Analyze Crop Data")

#     # Encoding the 'label' column (before dropping 'label')
#     if st.checkbox('Show encoded target labels'):
#         crop_dict = {
#             'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
#             'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
#             'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
#             'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
#             'jute': 21, 'coffee': 22
#         }
#         crop['crop_no'] = crop['label'].map(crop_dict)
#         st.write(crop.head())

#     # Filter dataset based on user input
#     nitrogen = st.slider("Select Nitrogen Level", 0.0, 200.0, 100.0)
#     phosphorus = st.slider("Select Phosphorus Level", 0.0, 200.0, 100.0)
#     potassium = st.slider("Select Potassium Level", 0.0, 200.0, 100.0)

#     # Apply filter on the dataset based on user input
#     filtered_data = crop[
#         (crop['N'] >= nitrogen - 10) & (crop['N'] <= nitrogen + 10) & 
#         (crop['P'] >= phosphorus - 10) & (crop['P'] <= phosphorus + 10) & 
#         (crop['K'] >= potassium - 10) & (crop['K'] <= potassium + 10)
#     ]

#     # Display the filtered data
#     if not filtered_data.empty:
#         st.write(f"Found {filtered_data.shape[0]} matching crops:")
#         st.dataframe(filtered_data)
#     else:
#         st.write("No crops found with the selected criteria.")

# # --- Download Dataset Section ---
# elif st.session_state.activity == 'Download Dataset':
#     # Check if the header has been displayed before
#     if not st.session_state.header_displayed:
#         changing_header()  # Display header with color change
#         st.session_state.header_displayed = True  # Mark the header as displayed
    
#     st.header("Download the Processed Dataset")
    
#     # Drop the 'label' column
#     crop.drop('label', axis=1, inplace=True)

#     # Display the dataset after dropping 'label'
#     if st.checkbox('Show final dataset after dropping label'):
#         st.write(crop.tail())

#     # Function to convert dataframe to CSV for download
#     def convert_df_to_csv(df):
#         return df.to_csv(index=False).encode('utf-8')

#     if st.checkbox('Download final dataset'):
#         final_csv = convert_df_to_csv(crop)
#         st.download_button(
#             label="Download CSV",
#             data=final_csv,
#             file_name='final_crop_dataset.csv',
#             mime='text/csv'
#         )

# # --- Advanced Features Section ---
# elif st.session_state.activity == 'Advanced Features':
#     # Check if the header has been displayed before
#     if not st.session_state.header_displayed:
#         changing_header()  # Display header with color change
#         st.session_state.header_displayed = True  # Mark the header as displayed
    
#     st.header("Advanced Features Section")

#     # Fertilizer recommendation based on Nitrogen, Phosphorus, Potassium
#     def recommend_fertilizer(nitrogen, phosphorus, potassium):
#         # Example fertilizer recommendation based on N, P, K levels
#         if nitrogen < 50 and phosphorus < 50 and potassium < 50:
#             return "Recommendation: Use a balanced NPK fertilizer (10-10-10)."
#         elif nitrogen > 150:
#             return "Recommendation: Use a high-nitrogen fertilizer (20-10-10)."
#         elif phosphorus > 150:
#             return "Recommendation: Use a high-phosphorus fertilizer (10-20-10)."
#         elif potassium > 150:
#             return "Recommendation: Use a high-potassium fertilizer (10-10-20)."
#         else:
#             return "Recommendation: Use a balanced NPK fertilizer (15-15-15)."

#     nitrogen = st.slider("Select Nitrogen Level", 0.0, 200.0, 100.0)
#     phosphorus = st.slider("Select Phosphorus Level", 0.0, 200.0, 100.0)
#     potassium = st.slider("Select Potassium Level", 0.0, 200.0, 100.0)

#     # Show fertilizer recommendation
#     st.write(recommend_fertilizer(nitrogen, phosphorus, potassium))

#     # Predicting crop yield based on NPK and temperature
#     def predict_crop_yield(nitrogen, phosphorus, potassium, temperature, humidity):
#         # Simple mock model for crop yield prediction
#         yield_pred = (nitrogen * 0.3) + (phosphorus * 0.2) + (potassium * 0.2) + (temperature * 0.15) + (humidity * 0.15)
#         return f"Predicted crop yield: {yield_pred:.2f} units"

#     temperature = st.slider("Select Temperature (°C)", 15, 40, 25)
#     humidity = st.slider("Select Humidity (%)", 40, 100, 60)

#     # Show crop yield prediction
#     st.write(predict_crop_yield(nitrogen, phosphorus, potassium, temperature, humidity))










# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your dataset
# crop = pd.read_csv("data/Crop_recommendation.csv")

# # st.markdown("""
# #     <h1 style="text-align: center; font-size: 40px; color: #4CAF50;">Crop and Fertilizer Recommendation System</h1>
# # """, unsafe_allow_html=True)



# # Initialize session state for activity and header display
# if 'activity' not in st.session_state:
#     st.session_state.activity = "Overview"

# # Activity selection with radio buttons
# st.session_state.activity = st.radio(
#     "Select an activity:", 
#     ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"],
#     index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"].index(st.session_state.activity)
# )

# # --- Overview Section ---
# if st.session_state.activity == 'Overview':
#     st.header("Overview of the Dataset")
    
#     # Dataset Overview
#     if st.checkbox('Show dataset preview'):
#         st.write(crop.head())

#     # Show dataset info and missing values
#     if st.checkbox('Show dataset information'):
#         st.write(crop.info())

#     if st.checkbox('Show missing values'):
#         st.write(crop.isnull().sum())

#     if st.checkbox('Show duplicate values'):
#         st.write(crop.duplicated().sum())

# # --- Dataset Statistics Section ---
# elif st.session_state.activity == 'Dataset Statistics':
#     st.header("Dataset Statistics")

#     # Statistics summary
#     if st.checkbox('Show dataset statistics'):
#         st.write(crop.describe())

#     # Target feature distribution
#     if st.checkbox('Show target feature distribution'):
#         st.write(crop['label'].value_counts())

#     # List features excluding target variable 'label'
#     st.write("Features: ", crop.columns.tolist())

#     # Correlation heatmap
#     if st.checkbox('Show correlation heatmap'):
#         num_cols = crop.select_dtypes(include=[np.number])
#         corr = num_cols.corr()
#         st.write(corr)
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
#         st.pyplot(plt)

# # --- Visualize Features Section ---
# elif st.session_state.activity == 'Visualize Features':
#     st.header("Visualize Features")

#     # Placeholder for feature visualization code (you can add your visualizations here)

# # --- Filter and Analyze Crop Section ---
# elif st.session_state.activity == 'Filter and Analyze Crop':
#     st.header("Filter and Analyze Crop Data")

#     # Placeholder for filtering and analysis code (you can add your filters and analyses here)

# # --- Download Dataset Section ---
# elif st.session_state.activity == 'Download Dataset':
#     st.header("Download the Processed Dataset")

#     # Placeholder for dataset download code (you can allow users to download processed dataset here)

# # --- Advanced Features Section ---
# elif st.session_state.activity == 'Advanced Features':
#     st.header("Advanced Features Section")

#     # Input sliders for NPK, Temperature, and Humidity
#     nitrogen = st.slider("Select Nitrogen Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
#     phosphorus = st.slider("Select Phosphorus Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
#     potassium = st.slider("Select Potassium Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
    
#     temperature = st.slider("Select Temperature (°C)", min_value=15, max_value=40, step=1)
#     humidity = st.slider("Select Humidity (%)", min_value=40, max_value=100, step=1)

#     # Fertilizer recommendation logic based on input values
#     if nitrogen > 0 and phosphorus > 0 and potassium > 0:
#         fertilizer_recommendation = "Use a balanced NPK fertilizer (15-15-15)."
#     elif nitrogen > 0 and phosphorus > 0:
#         fertilizer_recommendation = "Use a Nitrogen-Phosphorus fertilizer."
#     elif potassium > 0:
#         fertilizer_recommendation = "Use a Potassium-based fertilizer."
#     else:
#         fertilizer_recommendation = "Use a general-purpose fertilizer."

#     # Crop yield prediction (simple mock-up; in real-life, this would involve a trained ML model)
#     predicted_yield = (nitrogen + phosphorus + potassium) * 0.1 + (temperature - 15) * 2 + (humidity - 40) * 0.5
#     predicted_yield = max(0, predicted_yield)  # Ensuring yield can't be negative

#     # Display results
#     st.write(f"**Recommended Fertilizer:** {fertilizer_recommendation}")
#     st.write(f"**Predicted Crop Yield:** {predicted_yield:.2f} units")

#     # Visualization of NPK Levels and Temperature vs Yield (Mock-up visualization)
#     st.write("### Visualizing the NPK and Temperature effects on Yield")
#     data = pd.DataFrame({
#         "Nitrogen": [nitrogen],
#         "Phosphorus": [phosphorus],
#         "Potassium": [potassium],
#         "Temperature": [temperature],
#         "Yield": [predicted_yield]
#     })
    
#     fig, ax = plt.subplots()
#     ax.bar(data.columns[:-1], data.iloc[0, :-1], color=["#FF6347", "#FFD700", "#8A2BE2", "#4CAF50"])
#     ax.set_title("Impact of NPK and Temperature on Crop Yield")
#     st.pyplot(fig)










# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your dataset
# crop = pd.read_csv("data/Crop_recommendation.csv")

# # Initialize session state for activity and header display
# if 'activity' not in st.session_state:
#     st.session_state.activity = "Overview"

# # Activity selection with radio buttons
# st.session_state.activity = st.radio(
#     "Select an activity:", 
#     ["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"],
#     index=["Overview", "Dataset Statistics", "Visualize Features", "Filter and Analyze Crop", "Download Dataset", "Advanced Features"].index(st.session_state.activity)
# )

# # --- Overview Section ---
# if st.session_state.activity == 'Overview':
#     st.header("Overview of the Dataset")
    
#     # Dataset Overview
#     if st.checkbox('Show dataset preview'):
#         st.write(crop.head())

#     # Show dataset info and missing values
#     if st.checkbox('Show dataset information'):
#         st.write(crop.info())

#     if st.checkbox('Show missing values'):
#         st.write(crop.isnull().sum())

#     if st.checkbox('Show duplicate values'):
#         st.write(crop.duplicated().sum())

# # --- Dataset Statistics Section ---
# elif st.session_state.activity == 'Dataset Statistics':
#     st.header("Dataset Statistics")

#     # Statistics summary
#     if st.checkbox('Show dataset statistics'):
#         st.write(crop.describe())

#     # Target feature distribution
#     if st.checkbox('Show target feature distribution'):
#         st.write(crop['label'].value_counts())

#     # List features excluding target variable 'label'
#     st.write("Features: ", crop.columns.tolist())

#     # Correlation heatmap
#     if st.checkbox('Show correlation heatmap'):
#         num_cols = crop.select_dtypes(include=[np.number])
#         corr = num_cols.corr()
#         st.write(corr)
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlation'})
#         st.pyplot(plt)

# # --- Visualize Features Section ---
# elif st.session_state.activity == 'Visualize Features':
#     st.header("Visualize Features")

#     # Feature selection
#     feature = st.selectbox('Select a feature to visualize', crop.columns.tolist())

#     # Plotting the feature
#     st.write(f"### Distribution of {feature}")
#     plt.figure(figsize=(8, 6))
#     sns.histplot(crop[feature], kde=True, bins=20)
#     st.pyplot(plt)

# # --- Filter and Analyze Crop Section ---
# elif st.session_state.activity == 'Filter and Analyze Crop':
#     st.header("Filter and Analyze Crop Data")

#     # Filtering options
#     filter_column = st.selectbox('Select a column to filter', crop.columns.tolist())
#     filter_value = st.text_input(f'Enter value for {filter_column} to filter')

#     # Apply filter
#     if filter_value:
#         filtered_data = crop[crop[filter_column].astype(str).str.contains(filter_value, case=False, na=False)]
#         st.write(f"Filtered Data based on {filter_column} containing '{filter_value}':")
#         st.write(filtered_data)
#     else:
#         st.write("No filter applied. Please enter a value to filter the data.")

# # --- Download Dataset Section ---
# elif st.session_state.activity == 'Download Dataset':
#     st.header("Download the Processed Dataset")

#     # Provide a download button
#     st.write("Click the button below to download the dataset.")
#     csv = crop.to_csv(index=False)
#     st.download_button(
#         label="Download CSV",
#         data=csv,
#         file_name='crop_recommendation_processed.csv',
#         mime='text/csv'
#     )

# # --- Advanced Features Section ---
# elif st.session_state.activity == 'Advanced Features':
#     st.header("Advanced Features Section")

#     # Input sliders for NPK, Temperature, and Humidity
#     nitrogen = st.slider("Select Nitrogen Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
#     phosphorus = st.slider("Select Phosphorus Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
#     potassium = st.slider("Select Potassium Level (kg/ha)", min_value=0.00, max_value=200.00, step=0.1)
    
#     temperature = st.slider("Select Temperature (°C)", min_value=15, max_value=40, step=1)
#     humidity = st.slider("Select Humidity (%)", min_value=40, max_value=100, step=1)

#     # Fertilizer recommendation logic based on input values
#     if nitrogen > 0 and phosphorus > 0 and potassium > 0:
#         fertilizer_recommendation = "Use a balanced NPK fertilizer (15-15-15)."
#     elif nitrogen > 0 and phosphorus > 0:
#         fertilizer_recommendation = "Use a Nitrogen-Phosphorus fertilizer."
#     elif potassium > 0:
#         fertilizer_recommendation = "Use a Potassium-based fertilizer."
#     else:
#         fertilizer_recommendation = "Use a general-purpose fertilizer."

#     # Crop yield prediction (simple mock-up; in real-life, this would involve a trained ML model)
#     predicted_yield = (nitrogen + phosphorus + potassium) * 0.1 + (temperature - 15) * 2 + (humidity - 40) * 0.5
#     predicted_yield = max(0, predicted_yield)  # Ensuring yield can't be negative

#     # Display results
#     st.write(f"**Recommended Fertilizer:** {fertilizer_recommendation}")
#     st.write(f"**Predicted Crop Yield:** {predicted_yield:.2f} units")

#     # Visualization of NPK Levels and Temperature vs Yield (Mock-up visualization)
#     st.write("### Visualizing the NPK and Temperature effects on Yield")
#     data = pd.DataFrame({
#         "Nitrogen": [nitrogen],
#         "Phosphorus": [phosphorus],
#         "Potassium": [potassium],
#         "Temperature": [temperature],
#         "Yield": [predicted_yield]
#     })
    
#     fig, ax = plt.subplots()
#     ax.bar(data.columns[:-1], data.iloc[0, :-1], color=["#FF6347", "#FFD700", "#8A2BE2", "#4CAF50"])
#     ax.set_title("Impact of NPK and Temperature on Crop Yield")
#     st.pyplot(fig)











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
