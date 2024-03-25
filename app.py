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