from sim_classes import *
from general_funcs import *
import streamlit as st

import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import streamlit.components.v1 as components


def app():


    variable = st.selectbox("Variable to inspect: ", options = get_filenames_from_dir("data", '.json'))

    st.write(getattr(st.session_state.sim, f"{variable}"))