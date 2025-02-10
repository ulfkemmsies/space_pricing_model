from email.policy import default
from bleach import clean
from requests import options
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network


import time

from sim_classes import *
from general_funcs import *


def app():

    # state = st.session_state

    #Create the all-encompassing simulation object
    st.session_state.sim = Price_Sim()

    #Draw graph from the saved trajectory and vehicle data
    st.session_state.sim.graph_drawer(old_graph=None, dict_of_dicts=st.session_state.sim.trajectory_data)

    #This is just for weird situations
    st.button("Press to check for changes in variables to re-draw graph", on_click=st.session_state.sim.graph_drawer(old_graph=st.session_state.sim.graph, dict_of_dicts=st.session_state.sim.trajectory_data))

    #Choose which fuel sources should have their outflow graph shown
    source_graph_to_vis = st.multiselect("Fuel sources' graphs to be drawn", st.session_state.sim.global_vars['fuel_sources'])
    
    #Make sure the available nodes actually update and are reflected in the calculation
    def update_nodes(active_nodes):
        st.session_state.sim.global_vars.update({"active_nodes": active_nodes})
        st.session_state.sim.graph_drawer(dict_of_dicts=st.session_state.sim.trajectory_data)

    active_nodes = st.multiselect("The locations where there are infrastructure nodes (mainly for refuelling)", options=st.session_state.sim.global_vars["all_nodes"], default=st.session_state.sim.global_vars["active_nodes"])
    update_nodes(active_nodes=active_nodes)

    #Create two columns for a better fit
    col1, col2 = st.columns(2)

    #Add initial fuel prices at each source, also the LEO launch cost to spare us having to calculate it
    st.session_state.sim.global_vars['earth_to_LEO_cost'] = col1.number_input(f"Earth to LEO launch cost ($/kg)", 0, 50000, int(st.session_state.sim.global_vars['earth_to_LEO_cost']), 100)
    st.session_state.sim.global_vars['local_fuel_prices']['Earth'] = col2.number_input(f"Price of propellant on Earth ($/kg)", 0, 50000, int(st.session_state.sim.global_vars['local_fuel_prices']['Earth']), 100)
    
    #Use the LEO price calculated from Earth launch instead of user-selected one
    LEO_calc = st.checkbox("Use LEO launch cost?")

    #Add a initial fuel price input for each fuel source
    cols = [col1, col2]
    i= 0
    local_fuel_prices = dict()
    for fuel_source in st.session_state.sim.global_vars['fuel_sources']:
        col =  cols[i]
        local_fuel_prices[fuel_source] = col.number_input(f"Initial fuel price at {fuel_source}", 0, 50000, st.session_state.sim.global_vars['local_fuel_prices'][f"{fuel_source}"], 100)
        i += 1
    
    # Calculate the LEO price calculated from Earth launch instead of user-selected one
    if LEO_calc:
        local_fuel_prices["LEO"] = st.session_state.sim.global_vars['earth_to_LEO_cost'] * st.session_state.sim.global_vars['local_fuel_prices']['Earth']

    #Be sure to update the local fuel prices from our input
    st.session_state.sim.global_vars.update({"local_fuel_prices": local_fuel_prices})

    #Converge on a stable price state before merging the fuel source subgraphs
    st.session_state.sim.graph.converge_prices(st.session_state.sim.graph.subgraphs, st.session_state.sim.global_vars["local_fuel_prices"])

    #Possibly redundant but added this to avoid the error of the best price not propagating because of changed attribute names
    st.session_state.sim.graph.get_best_prices(st.session_state.sim.graph.subgraphs)

    #Takes the subgraphs (or any graph), merges them, and plots them with Pyvis
    def draw_graph():

        if len(source_graph_to_vis) != 0: #only attempt to draw graph once something is selected

            st.session_state.sim.global_vars['draw_best_path'] = st.checkbox("Draw best path between two nodes?")

            if set(source_graph_to_vis) == set(st.session_state.sim.graph.subgraphs.keys()):
                #if all fuel sources selected, merge them and copy their graphs into Pyvis for visualization
                st.session_state.sim.graph.merged_graph = st.session_state.sim.graph.merge_subgraphs(st.session_state.sim.graph.subgraphs)

                if st.session_state.sim.global_vars['draw_best_path']:

                    best_path_fuel_source = st.selectbox("Fuel Source for best path?", options=source_graph_to_vis)
                    best_path_subgraph = st.session_state.sim.graph.subgraphs[best_path_fuel_source]

                    non_terminal_nodes = [node for node in best_path_subgraph.nodes.keys() if len(list(nx.descendants(best_path_subgraph, node))) != 0]
                    
                    col3, col4 = st.columns(2)
                    st.session_state.sim.global_vars['best_path_source_node'] = col3.selectbox("Source node", options=non_terminal_nodes)
                    st.session_state.sim.global_vars['best_path_target_node'] = col4.selectbox("Target node", options= list(nx.descendants(best_path_subgraph, st.session_state.sim.global_vars['best_path_source_node'])))

                    graph_to_draw = st.session_state.sim.graph.draw_best_path(
                        graph_in = best_path_subgraph,
                        source = st.session_state.sim.global_vars['best_path_source_node'],
                        target = st.session_state.sim.global_vars['best_path_target_node'],
                        graph_out = st.session_state.sim.graph.merged_graph,
                        is_already_flow_subgraph=True)

                    st.session_state.sim.graph.merged_graph = graph_to_draw

                space_net = st.session_state.sim.graph.graph_to_pyvis(st.session_state.sim.graph.merged_graph, merged=True)
            else: #Just copy a single graph into Pyvis (stylishly)

                if st.session_state.sim.global_vars['draw_best_path']:
                    
                    graph_to_draw = st.session_state.sim.graph.subgraphs[source_graph_to_vis[0]]

                    non_terminal_nodes = [node for node in graph_to_draw.nodes.keys() if len(list(nx.descendants(graph_to_draw, node))) != 0]
                    
                    col3, col4 = st.columns(2)
                    st.session_state.sim.global_vars['best_path_source_node'] = col3.selectbox("Source node", options=non_terminal_nodes)
                    st.session_state.sim.global_vars['best_path_target_node'] = col4.selectbox("Target node", options= list(nx.descendants(graph_to_draw, st.session_state.sim.global_vars['best_path_source_node'])))

                    graph_to_draw = st.session_state.sim.graph.draw_best_path(
                        graph_in = graph_to_draw,
                        source = st.session_state.sim.global_vars['best_path_source_node'],
                        target = st.session_state.sim.global_vars['best_path_target_node'],
                        graph_out = graph_to_draw,
                        is_already_flow_subgraph=True)

                    st.session_state.sim.graph.subgraphs[source_graph_to_vis[0]] = graph_to_draw

                space_net = st.session_state.sim.graph.graph_to_pyvis(st.session_state.sim.graph.subgraphs[source_graph_to_vis[0]])
            
            space_net.set_edge_smooth('dynamic')
            # Generate network with specific layout settings
            space_net.force_atlas_2based(gravity=-50, central_gravity=0.001,
                            spring_length=500, spring_strength=0.01,
                            damping=0.95, overlap=0)

            space_net.toggle_physics(True)

            # Save and read graph as HTML file (on Streamlit Sharing)
            try:
                path = './tmp'
                space_net.save_graph(f'{path}/pyvis_graph.html')
                HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')

            # Save and read graph as HTML file (locally)
            except:
                path = './html_files'
                space_net.save_graph(f'{path}/pyvis_graph.html')
                HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')

            # Load HTML into HTML component for display on Streamlit
            components.html(HtmlFile.read(), width=899, height=695)

        # state = st.session_state
        # state.sync()

    draw_graph()

    # state = st.session_state
    # state.sync()