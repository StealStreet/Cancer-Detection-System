import streamlit as st 
import pandas as pd 
import numpy as np 
import altair as alt
import plotly.express as px
import time 
import os 

from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from streamlit_option_menu import option_menu


with st.sidebar:
    # st.image('data\logo.jpg')
    # st.title("Cancer Detection Web App")
    # selected = st.radio(
    #     "Navigation", ["Dashboard", "Upload", "Profiling", "User Input", "Download"])
    # st.info("This application allows you to build an automated ML pipeline using Stream Lit, Pandas Profiling and PyCaret")
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Upload", "Profiling", "User Input", "Download"],
        icons=["house", "upload", "person-lines-fill",
               "input-cursor", "download"],
        default_index=0,
        # orientation="horizontal"
    )


if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if selected == "Dashboard":
    # st.set_page_config(
    #     page_title="US Population Dashboard",
    #     page_icon="😡",
    #     layout="wide",
    #     initial_sidebar_state="expanded")

    alt.themes.enable("dark")

    df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')

    with st.sidebar:
        st.title('✅ US Population Dashboard')

        year_list = list(df_reshaped.year.unique())[::-1]

        selected_year = st.selectbox(
            'Select a year', year_list, index=len(year_list)-1)
        df_selected_year = df_reshaped[df_reshaped.year == selected_year]
        df_selected_year_sorted = df_selected_year.sort_values(
            by="population", ascending=False)

        color_theme_list = ['blues', 'cividis', 'greens', 'inferno',
                            'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
        selected_color_theme = st.selectbox(
            'Select a color theme', color_theme_list)


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


    def make_choropleth(input_df, input_id, input_column, input_color_theme):
        choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
                                   color_continuous_scale=input_color_theme,
                                   range_color=(
                                       0, max(df_selected_year.population)),
                                   scope="usa",
                                   labels={'population': 'Population'}
                                   )
        choropleth.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=350
        )
        return choropleth


    def calculate_population_difference(input_df, input_year):
      selected_year_data = input_df[input_df['year'] == input_year].reset_index()
      previous_year_data = input_df[input_df['year']
                                    == input_year - 1].reset_index()
      selected_year_data['population_difference'] = selected_year_data.population.sub(
          previous_year_data.population, fill_value=0)
      return pd.concat([selected_year_data.states, selected_year_data.id, selected_year_data.population, selected_year_data.population_difference], axis=1).sort_values(by="population_difference", ascending=False)


    def make_donut(input_response, input_text, input_color):
      if input_color == 'blue':
          chart_color = ['#29b5e8', '#155F7A']
      if input_color == 'green':
          chart_color = ['#27AE60', '#12783D']
      if input_color == 'orange':
          chart_color = ['#F39C12', '#875A12']
      if input_color == 'red':
          chart_color = ['#E74C3C', '#781F16']

      source = pd.DataFrame({
          "Topic": ['', input_text],
          "% value": [100-input_response, input_response]
      })
      source_bg = pd.DataFrame({
          "Topic": ['', input_text],
          "% value": [100, 0]
      })

      plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
          theta="% value",
          color=alt.Color("Topic:N",
                          scale=alt.Scale(
                              # domain=['A', 'B'],
                              domain=[input_text, ''],
                              # range=['#29b5e8', '#155F7A']),  # 31333F
                              range=chart_color),
                          legend=None),
      ).properties(width=130, height=130)

      text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32,
                            fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
      plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
          theta="% value",
          color=alt.Color("Topic:N",
                          scale=alt.Scale(
                              # domain=['A', 'B'],
                              domain=[input_text, ''],
                              range=chart_color),  # 31333F
                          legend=None),
      ).properties(width=130, height=130)
      return plot_bg + plot + text


    def format_number(num):
        if num > 1000000:
            if not num % 1000000:
                return f'{num // 1000000} M'
            return f'{round(num / 1000000, 1)} M'
        return f'{num // 1000} K'


    col = st.columns((1.5, 4.5, 2), gap='medium')


    with col[0]:
        st.markdown('#### Gains/Losses')

        df_population_difference_sorted = calculate_population_difference(
            df_reshaped, selected_year)

        if selected_year > 2010:
            first_state_name = df_population_difference_sorted.states.iloc[0]
            first_state_population = format_number(
                df_population_difference_sorted.population.iloc[0])
            first_state_delta = format_number(
                df_population_difference_sorted.population_difference.iloc[0])
        else:
            first_state_name = '-'
            first_state_population = '-'
            first_state_delta = ''
        st.metric(label=first_state_name, value=first_state_population,
                  delta=first_state_delta)

        if selected_year > 2010:
            last_state_name = df_population_difference_sorted.states.iloc[-1]
            last_state_population = format_number(
                df_population_difference_sorted.population.iloc[-1])
            last_state_delta = format_number(
                df_population_difference_sorted.population_difference.iloc[-1])
        else:
            last_state_name = '-'
            last_state_population = '-'
            last_state_delta = ''
        st.metric(label=last_state_name, value=last_state_population,
                  delta=last_state_delta)

        st.markdown('#### States Migration')

        if selected_year > 2010:
            # Filter states with population difference > 50000
            # df_greater_50000 = df_population_difference_sorted[df_population_difference_sorted.population_difference_absolute > 50000]
            df_greater_50000 = df_population_difference_sorted[
                df_population_difference_sorted.population_difference > 50000]
            df_less_50000 = df_population_difference_sorted[
                df_population_difference_sorted.population_difference < -50000]

            # % of States with population difference > 50000
            states_migration_greater = round(
                (len(df_greater_50000)/df_population_difference_sorted.states.nunique())*100)
            states_migration_less = round(
                (len(df_less_50000)/df_population_difference_sorted.states.nunique())*100)
            donut_chart_greater = make_donut(
                states_migration_greater, 'Inbound Migration', 'green')
            donut_chart_less = make_donut(
                states_migration_less, 'Outbound Migration', 'red')
        else:
            states_migration_greater = 0
            states_migration_less = 0
            donut_chart_greater = make_donut(
                states_migration_greater, 'Inbound Migration', 'green')
            donut_chart_less = make_donut(
                states_migration_less, 'Outbound Migration', 'red')

        migrations_col = st.columns((0.2, 1, 0.2))
        with migrations_col[1]:
            st.write('Inbound')
            st.altair_chart(donut_chart_greater)
            st.write('Outbound')
            st.altair_chart(donut_chart_less)

    with col[1]:
        st.markdown('#### Total Population')

        choropleth = make_choropleth(
            df_selected_year, 'states_code', 'population', selected_color_theme)
        st.plotly_chart(choropleth, use_container_width=True)

        heatmap = make_heatmap(df_reshaped, 'year', 'states',
                               'population', selected_color_theme)
        st.altair_chart(heatmap, use_container_width=True)

    with col[2]:
        st.markdown('#### Top States')

        st.dataframe(df_selected_year_sorted,
                     column_order=("states", "population"),
                     hide_index=True,
                     width=None,
                     column_config={
                         "states": st.column_config.TextColumn(
                             "States",
                         ),
                         "population": st.column_config.ProgressColumn(
                             "Population",
                             format="%f",
                             min_value=0,
                             max_value=max(df_selected_year_sorted.population),
                         )}
                     )

        with st.expander('About', expanded=True):
            st.write('''
                - Data: [U.S. Census Bureau](<https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html>).
                - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
                - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
                ''')

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

if selected == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if selected == "User Input":
    st.title("User Input")


if selected == "Download":
    pass