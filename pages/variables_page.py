from sim_classes import *
from general_funcs import *
from general_funcs import *
import streamlit as st

import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import streamlit.components.v1 as components
from st_aggrid import AgGrid, GridOptionsBuilder

def app():

    st.session_state.sim.update_attrs_from_disk()

    variable = st.selectbox("Variable to inspect: ", options = get_filenames_from_dir("data", '.json'))

    if variable == "global_vars":

        simple_global_vars = {key:value for key,value in st.session_state.sim.global_vars.items() if \
            ( (not(isinstance(value, dict))) and (not(isinstance(value, list))) and (not(isinstance(value, dict))) ) }

        complex_global_vars = st.session_state.sim.global_vars

        for key in simple_global_vars.keys(): del complex_global_vars[key]

        complex_global_vars['simple_global_vars'] = simple_global_vars

        global_variable = st.selectbox("Global Variable to inspect: ", options = complex_global_vars.keys())
        
        setattr(st.session_state.sim, "current_edit_var", {f"{global_variable}":complex_global_vars[global_variable]})
    
    elif variable == "trajectory_data":

        edge_variable = st.selectbox("Edge Variable to inspect: ", options = get_filenames_from_dir("edge_var_tables", ".csv"))
        df = pd.read_csv([filepath for filepath in get_filepaths_from_dir("edge_var_tables") if edge_variable in filepath][0])
        if "Unnamed: 0" in df.columns:
            df = df.set_index(df["Unnamed: 0"].values)
            df = df.drop(columns=["Unnamed: 0"])

        setattr(st.session_state.sim, "current_edit_var", {f"{edge_variable}":df})

    else: setattr(st.session_state.sim, "current_edit_var", {f"{variable}": getattr(st.session_state.sim, f"{variable}")})

    draw_editable_table(st.session_state.sim.current_edit_var)

    # st.button("Save changes to disk", on_click=)