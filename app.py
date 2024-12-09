# Necessary imported modules
import matplotlib.pyplot as plt
import streamlit as st 
import pandas as pd 
import numpy as np 
import altair as alt
import seaborn as sns
import plotly.express as px
import pickle
import numpy as np


import joblib
import time 
import os 

# Modules for the Machine Learning pipeline
from ydata_profiling import ProfileReport # type: ignore
from streamlit_ydata_profiling import st_profile_report
from streamlit_option_menu import option_menu
from sklearn.datasets import load_breast_cancer

# ML modules
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, predict_model, save_model, load_model, pull


#title of the web page of the breast cancer detection system
st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon=":home:"
)

#using sidebar functions to create the sidebar using streamlit
with st.sidebar:
    #logo of the app
    st.image('logo.jpg')
    #title of the app
    st.title("Cancer Detection Web App")
    #navigation bar
    # st.sidebar.title("Navigation")
    # creating a radio button to select the page
    choice_dashboard = st.sidebar.radio(
        "Navigate to:",
        ["Home","Upload", "Profiling", "Machine Learning", "Diagnosis"]
    )
    # provide some information about the app
    # st.info("This application allows you to build an automated ML pipeline using Stream Lit, Pandas Profiling and PyCaret")

# Possible output in the cancer detection
class_list = ["Benign", "Malignant"]

# all value for detection of the cancer tumor
input_data = {
    'mean_radius' : [0],
    'mean_texture' : [0],
    'mean_perimeter' : [0],
    'mean_area' : [0],
    'mean_smoothness' : [0],
    'mean_compactness' : [0], 
    'mean_concavity' : [0],
    'mean_concave points' : [0],
    'mean_symmetry' : [0],
    'mean_fractal_dimension' : [0],
    'radius_error' : [0],
    'texture_error' : [0],
    'perimeter_error' : [0],
    'area_error' : [0],
    'smoothness_error' : [0],
    'compactness_error' : [0],
    'concavity_error' : [0],
    'concave_points_error' : [0],
    'symmetry_error' : [0],
    'fractal_dimension_error' : [0],
    'worst_radius' : [0],
    'worst_texture' : [0],
    'worst_perimeter' : [0],
    'worst_area' : [0],
    'worst_smoothness' : [0],
    'worst_compactness' : [0],
    'worst_concavity' : [0],
    'worst_concave points' : [0],
    'worst_symmetry' : [0],
    'worst_fractal dimension' : [0],
}


# Load the joblib model
model = joblib.load('model_joblib')

# Define function to predict health
def predict_health(input_data):
    # Ensure features are in the correct order
    feature_names = ['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','mean_compactness','mean_concavity','mean_concave_points','mean_symmetry','mean_fractal_dimension','radius_error','texture_error','perimeter_error','area_error','smoothness_error','compactness_error','concavity_error','concave_points_error','symmetry_error','fractal_dimension_error','worst_radius','worst_texture','worst_perimeter','worst_area','worst_smoothness','worst_compactness','worst_concavity','worst_concave_points','worst_symmetry','worst_fractal_dimension'
]    
    # Prepare the data in the correct order
    X_new = np.array([[input_data[feature] for feature in feature_names]])
    print("Shape of X_new:", X_new.shape)
    # Make predictions
    prediction = model.predict(X_new.reshape(1, -1))
    
    # Return the prediction
    return class_list[int(prediction)]


# creating sourcedata csv file for training and testing, if present then leave it
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# main function
if __name__ == "__main__":
    
    # dashboard navigation
    if choice_dashboard == "Home":
        
        st.title('✅ Cancer Prediction Dashboard')
        alt.themes.enable("dark")

        with st.sidebar:
            st.info("This is a dashboard to visualize the cancer data")
        color_theme_list = ['blues', 'cividis', 'greens', 'inferno',
                            'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
        
        df_reshaped = pd.read_csv("data/actual.csv")
        
        @st.cache_data
        def cancer_data():
            df = pd.read_csv("data/transformed_data.csv")
            return df.set_index("Region")

        try:
            df = cancer_data()
            countries = st.multiselect (
                "Choose countries", list(df.index), ["India", "China"]
            )
            if not countries:
                st.error("Please select at least one country.")
            else:
                data = df.loc[countries]
                st.write("### Cancer Data Worldwide (per 100 peoples)", data.sort_index())

                data = data.T.reset_index()
                data = pd.melt(data, id_vars=["index"]).rename(
                    columns={"index": "year", "value": "neoplasms_per_100_peoples"}
                )

                chart = (
                    alt.Chart(data)
                    .mark_area(opacity=0.3)
                    .encode(
                    x="year:T",
                    y=alt.Y("neoplasms_per_100_peoples", stack=None),
                    color= "Region:N"
                    )
                )

                st.altair_chart(chart, use_container_width=True)


        except RuntimeError as r:
            st.error("""**There is a bug.
                     Connection error: %s
                     """
                     % e.reason
                     )
            
        def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
            heatmap = alt.Chart(input_df).mark_rect().encode(
                y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18,
                        titlePadding=15, titleFontWeight=900, labelAngle=0)),
                x=alt.X(f'{input_x}:O', axis=alt.Axis(
                    title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
                color=alt.Color(f'max({input_color}):Q',
                                legend=None,
                                scale=alt.Scale(scheme=input_color_theme)),
                stroke=alt.value('black'),
                strokeWidth=alt.value(0.25),
            ).properties(width=900
                         ).configure_axis(
                labelFontSize=12,
                titleFontSize=12
            )
            # height=300
            return heatmap
        col1, col2 = st.columns([0.6,0.4])
        
        
        with col1:
            selected_color_theme = st.selectbox(
            'Select a color theme', color_theme_list)
            heatmap = make_heatmap(df_reshaped, 'Year', 'Entity',
                               'neoplasms_per_100_peoples', selected_color_theme)
            st.altair_chart(heatmap, use_container_width=True)
        
        with col2:
            # selected_color_theme = st.selectbox('Select a color theme', color_theme_list)
            
            with st.expander('About', expanded=True):
                st.write('''
                - Data: [U C Irvine] (Dataset obtained from Scikit Learn (Dataset)).
                - :orange[**Cancer DataFrame Graph**]: Demographic data of cancer patients (pre 100 person).
                - :orange[**HeatMap Trend**]: HeatMap generation of the cancer data (year wise).
                - :Orange[**Web Developer**]: Sk Sofiquee Fiaz).
                ''')
                
    elif choice_dashboard == "Upload":
        with st.sidebar:
            st.info("This application allows you to build an automated ML pipeline using Stream Lit, Pandas Profiling and PyCaret")
        st.title("Choose a ML Algorithm!!!")
        if "visibility" not in st.session_state:
            st.session_state.visibility = "visible"
            st.session_state.disabled = False
        col1, col2 =  st.columns(2)
        with col1:
            # st.header("Machine Learning")
            # st.radio(                                         #radio features
            #     "Set selectbox label visibility 👉",
            #     key="visibility",
            #     options=["visible", "hidden", "collapsed"],
            # )
            option1 = st.selectbox(
                "Which option you are going to opt.. for",
                ("Machine Learning", "Deep Learning"),
                index=None,
                placeholder="Select a model...."
            )
        with col2:
            if option1 == "Machine Learning":
                option2 = st.selectbox(
                    "Select an algorithm....",
                    ("Support Vector Machine", "Decision Tree"),
                    index=None,
                    placeholder="Select an algorithm...."
                )
            else:
                option2 = st.selectbox(
                    "Select an algorithm....",
                    ("Convolutional Neural Network", "KNN"),
                    index=None,
                    placeholder="Select an algorithm...."
                )
        st.write("You selected: ", option1, option2)
        st.title("Upload Your Data for Modelling!!!")
        file = st.file_uploader("Upload Your Dataset Here...")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv("sourcedata.csv", index=None)
            st.dataframe(df)
    
    elif choice_dashboard == "Profiling":
        st.title("Automated Exploratory Data Analysis")
        profile_report = df.profile_report()
        st_profile_report(profile_report)
        
    elif choice_dashboard == "Machine Learning":
        st.title("Machine Learning")
        st.write("Select an option from the sidebar to proceed")
        target = st.selectbox("Select your target", df.columns)
        
        if st.button("Train model"):
            st.info("Training the model")
            with st.spinner('Wait for it...'):
                time.sleep(5)
            st.success('Done!')
            setup(df, target=target, verbose=False)
            setup_df = pull()
            st.info("This is the ML Experiment settings")
            st.dataframe(setup_df)
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()
            best_model = compare_models()
            compare_df = pull()
            st.success("This is the comparison of the models")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, "best_model")
                
    elif choice_dashboard == "Diagnosis":
        #creating a select box to select the page
        with st.sidebar:
            st.info("This application allows you to build an automated ML pipeline using Stream Lit, Pandas Profiling and PyCaret")
            
        st.title("User Input")
        st.write("Please enter the following details")
        
        left, mid, right = st.columns(3)
        
        # target = st.number_input("Target", min_value=0, max_value=1, step=1, value=0)
        with left:
            mean_radius = st.number_input("Mean Radius", min_value=-100.0, max_value=100.0, step=0.5, value=0.0)
            mean_texture = st.number_input("Mean Texture", min_value=-100.0, max_value=100.0, step=0.1, value=0.0)
            mean_perimeter = st.number_input("Mean Perimeter", min_value=-100.0, max_value=1050.0, step=50.0, value=0.0)
            mean_area = st.number_input("Mean Area", min_value=-100.0, max_value=5.0, step=0.1, value=0.0)
            mean_smoothness = st.number_input("Mean Smoothness", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            mean_compactness = st.number_input("Mean Compactness", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            mean_concavity = st.number_input("Mean Concavity", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            mean_concave_points = st.number_input("Mean Concave Points", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            mean_symmetry = st.number_input("Mean Symmetry", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
        with mid:
            radius_error = st.number_input("Radius Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            texture_error = st.number_input("Texture Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            perimeter_error = st.number_input("Perimeter Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            area_error = st.number_input("Area Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            smoothness_error = st.number_input("Smoothness Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            compactness_error = st.number_input("Compactness Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            concavity_error = st.number_input("Concavity Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            concave_points_error = st.number_input("Concave Points Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            symmetry_error = st.number_input("Symmetry Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            fractal_dimension_error = st.number_input("Fractak Dimension Error", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
        with right:
            worst_radius = st.number_input("Worst Radius", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_texture = st.number_input("Worst Texture", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_perimeter = st.number_input("Worst Perimeter", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_area = st.number_input("Worst Area", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_smoothness = st.number_input("Worst Smoothness", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_compactness = st.number_input("Worst Compactness", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_concavity = st.number_input("Worst Concavity", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_concave_points = st.number_input("Worst Concave Points", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_symmetry = st.number_input("Worst Symmetry", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=-100.0, max_value=100.0, step=0.2, value=0.0)
            
        # input_data["target"] = [target]
        input_data["mean_radius"] = [mean_radius]
        input_data["mean_texture"] = [mean_texture]
        input_data["mean_perimeter"] = [mean_perimeter]
        input_data["mean_area"] = [mean_area]
        input_data["mean_smoothness"] = [mean_smoothness]
        input_data["mean_compactness"] = [mean_compactness]
        input_data["mean_concavity"] = [mean_concavity]
        input_data["mean_concave_points"] = [mean_concave_points]
        input_data["mean_symmetry"] = [mean_symmetry]
        input_data["mean_fractal_dimension"] = [mean_fractal_dimension]
        input_data["radius_error"] = [radius_error]
        input_data["texture_error"] = [texture_error]
        input_data["perimeter_error"] = [perimeter_error]
        input_data["area_error"] = [area_error]
        input_data["smoothness_error"] = [smoothness_error]
        input_data["compactness_error"] = [compactness_error]
        input_data["concavity_error"] = [concavity_error]
        input_data["concave_points_error"] = [concave_points_error]
        input_data["symmetry_error"] = [symmetry_error]
        input_data["fractal_dimension_error"] = [fractal_dimension_error]
        input_data["worst_radius"] = [worst_radius]
        input_data["worst_texture"] = [worst_texture]
        input_data["worst_perimeter"] = [worst_perimeter]
        input_data["worst_area"] = [worst_area]
        input_data["worst_smoothness"] = [worst_smoothness]
        input_data["worst_compactness"] = [worst_compactness]
        input_data["worst_concavity"] = [worst_concavity]
        input_data["worst_concave_points"] = [worst_concave_points]
        input_data["worst_symmetry"] = [worst_symmetry]
        input_data["worst_fractal_dimension"] = [worst_fractal_dimension]
        
        button = st.button("Predict")
        
        if button and len(input_data.values()) > 32:
            predicted_class = predict_health(input_data=input_data)
            st.info(f"### Predicted Class : {predicted_class}")
            
        with open("best_model.pkl", "rb") as f:
            st.download_button("Download the model", f, "trained_model.pkl", "application/octet-stream")
