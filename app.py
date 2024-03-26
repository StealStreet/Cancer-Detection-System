import streamlit as st 
import pandas as pd 
import numpy as np 
import altair as alt
import time 
import os 

from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Upload", "Profiling", "User Input", "Download"],
    icons=["none", "upload", "person-lines-fill", "input-cursor", "download"],
    default_index = 0,
    orientation = "horizontal"
)

if selected == "Upload":
    st.title("Choose a ML Algorithm!!!")
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    col1, col2 =  st.columns(2)

    with col1:
        # st.header("Machine Learning")
        # st.radio(
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