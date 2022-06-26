from audioop import reverse
from general_funcs import *
import pandas as pd
import json
import os

from configparser import NoOptionError
from xmlrpc.client import Boolean
from scipy import stats
import numpy as np
import pandas as pd
from itertools import chain
from operator import attrgetter
import math as m
import networkx as nx
from operator import mul
from functools import reduce
from deepdiff import DeepDiff
from pyvis.network import Network
import streamlit as st

class Graph:

    def __init__(self, dict_of_dict=None):
        
        if dict_of_dict == None:
            self.g_main = nx.MultiDiGraph()
        elif type(dict_of_dict) == dict:
            self.g_main = nx.MultiDiGraph(nx.from_dict_of_dicts(dict_of_dict, create_using=nx.MultiDiGraph))
            self.trajectory_data = dict_of_dict

        self.aerobraking_mass_penalty = None
        self.global_vars = None

# ---------------------------------------------------------

    def update_nodes(self, graph, nodes):
        nodes_to_delete = [node for node in graph.nodes.keys() if node not in nodes]
        graph.remove_nodes_from(nodes_to_delete)

    def system_state_to_dict(self, subgraphs, year):
        
        fuel_sources = list(subgraphs.keys())
        nodes = subgraphs[fuel_sources[0]].nodes.keys()

        out_dict = dict()
        out_dict['year'] = year
        for fuel_source in fuel_sources:
            for node in nodes:
                if node != fuel_source:
                    out_dict[f"{node}_{fuel_source}"] = subgraphs[fuel_source].nodes[node]["fuel_price"]
                else:
                    out_dict[f"{node}_local"] = subgraphs[fuel_source].nodes[node]["fuel_price"]

        return out_dict

# ---------------------------------------------------------

    def assign_mass_fractions(self, graph):
        for edge in graph.edges.items():

            attrs = edge[1]
            edge_item = edge[0]
            
            vehicle = self.vehicles[attrs['vehicle']]
            gross_sens, prop_sens = self.calc_mass_sens_terms(attrs['vehicle'])
            
            Isp = vehicle["engine"]["Isp"]
            aerobraking = vehicle["aerobraking"]
            
            directionality = attrs["directionality"]
            dV = attrs["dV"]

            mass_frac = m.exp((1000*dV)/(9.807*Isp))
            
            Minit_Mpay = self.calc_Minit_Mpay(prop_sens, gross_sens, mass_frac, directionality, aerobraking)
            Mprop_Mpay = self.calc_Mprop_Mpay(prop_sens, gross_sens, mass_frac, directionality, aerobraking)
            Mdry_Mpay = self.calc_Mdry_Mpay(prop_sens, gross_sens, mass_frac, directionality, aerobraking)

            attr_dict = {edge_item: {"Mprop_Mpay": Mprop_Mpay, "Minit_Mpay": Minit_Mpay, "Mdry_Mpay": Mdry_Mpay}}

            nx.set_edge_attributes(graph, attr_dict)
        
    def calc_Mprop_Mpay(self, prop_sens_term, gross_sens_term, mass_frac_term, direct, aerobraking):

        Minit_Mpay = self.calc_Minit_Mpay(prop_sens_term, gross_sens_term, mass_frac_term, direct, aerobraking)

        if aerobraking == False: aerocheck = self.global_vars["aerobraking_mass_penalty"]
        else: aerocheck = 1

        if direct == 1: mass_frac_term_2 = 1
        elif direct == 2: mass_frac_term_2 = mass_frac_term

        return (Minit_Mpay * (1 - 1/(mass_frac_term * mass_frac_term_2))) - (1 - 1/mass_frac_term_2)

    def calc_Minit_Mpay(self, prop_sens_term, gross_sens_term, mass_frac_term, direct, aerobraking):

        if aerobraking == False: aerocheck = self.global_vars["aerobraking_mass_penalty"]
        else: aerocheck = 1

        if direct == 1: mass_frac_term_2 = 1
        elif direct == 2: mass_frac_term_2 = mass_frac_term

        return mass_frac_term * (1 - prop_sens_term*(mass_frac_term_2 - 1)) * 1/(1 - (prop_sens_term + gross_sens_term)*mass_frac_term * mass_frac_term_2 + prop_sens_term) * (self.global_vars["aerobraking_mass_penalty"]/aerocheck)

    def calc_Mdry_Mpay(self, prop_sens_term, gross_sens_term, mass_frac_term, direct, aerobraking):

        Minit_Mpay = self.calc_Minit_Mpay(prop_sens_term, gross_sens_term, mass_frac_term, direct, aerobraking)
        Mprop_Mpay = self.calc_Mprop_Mpay(prop_sens_term, gross_sens_term, mass_frac_term, direct, aerobraking)

        return gross_sens_term * Minit_Mpay + prop_sens_term * Mprop_Mpay

    def calc_mass_sens_terms(self, vehicle_name:str):

        vehicle_data = self.vehicles[vehicle_name]
        engine_data = self.engines[vehicle_data["engine"]]
        propellant_data = self.propellants[engine_data['propellant']]

        prop_sens = (engine_data['MXR'] * (propellant_data['f_oxtank'] / propellant_data['oxidizer_density'] + \
            (propellant_data['f_fueltank'] / propellant_data['fuel_density']))) / ( (1+ engine_data['MXR']) * (1 - propellant_data['f_ullage']) )  ])

        if T2W not in vehicle_data.keys():
            T2W = self.global_vars['T2W_default']

        T2W_eng = (engine_data['thrust_vac']*1000)/ (engine_data['dry_mass'] * self.global_vars['g0'])
        
        gross_sens = (T2W * (1 + engine_data['f_TSW'] * T2W)) / T2W_eng

        return gross_sens, prop_sens

# ---------------------------------------------------------

    def best_fuel_flow_subgraphs(self, graph, sources, fuel_sources=None):
        
        if fuel_sources == None:
            fuel_sources = [None]
        else:
            pass
        
        subgraphs = dict()
        
        for fuel_source in fuel_sources:
            
            all_paths = []
            
            for source in sources:

                paths = []
                targets = list(graph.nodes.keys())
                targets.remove(source)

                paths = []

                for target in targets:
                    
                    if fuel_source == None:
                        subg = self.fuel_flow_path(graph, source)
                    else:
                        subg = self.fuel_flow_path(graph, fuel_source)

                    if (source, target) in subg.edges:
                        best_path = self.find_best_path(graph, source, target, fuel_source)
                        paths.append(best_path)
                    else:
                        pass
            
                all_paths.append(paths)
            
            flat_list = [item for sublist in all_paths for item in sublist]
            all_paths = [path_item[0] for path_item in flat_list]
            
            all_edges = []
            
            for path in all_paths:
                for i in range(len(path)-1):
                    new_edge = (path[i], path[i+1])
                    all_edges.append(new_edge)
            
            fuel_subgraph_edges = set(all_edges)
            
            edges_to_delete = set([edge[:2] for edge in graph.edges]) - fuel_subgraph_edges

            fuel_subgraph = graph.copy()
            fuel_subgraph.remove_edges_from(edges_to_delete)
            
            subgraphs[f"{fuel_source}"] = nx.DiGraph(fuel_subgraph)
        
        return subgraphs       
            
    def find_best_path(self, graph, source, target, fuel_source=None, is_already_flow_subgraph = False):
        total_Ks = []
        
        if is_already_flow_subgraph == False:
            if fuel_source == None:
                subg = self.fuel_flow_path(graph, source)
            else:
                subg = self.fuel_flow_path(graph, fuel_source)
            
            if (source, target) in subg.edges:
                pass
            else:
                raise ValueError("Flow in wrong direction!")
                
            edges_to_delete = [edge[:2] for edge in graph.edges] - subg.edges

            source_flow_graph = graph.copy()
            source_flow_graph.remove_edges_from(edges_to_delete)
        else:
            source_flow_graph = graph.copy()

        for path in nx.all_simple_paths(source_flow_graph, source, target):
            if  is_already_flow_subgraph == False:
                total_mass = reduce(mul, ((graph[start][end][0]['Mprop_Mpay']+1) for start, end in zip(path[:-1], path[1:])), 1)

            else:
                total_mass = reduce(mul, ((graph[start][end]['Mprop_Mpay']+1) for start, end in zip(path[:-1], path[1:])), 1)

            for start, end in zip(path[:-1], path[1:]):
                total_Ks.append([path, total_mass])

            paths_and_Ks = np.array(total_Ks)
            best_path = paths_and_Ks[paths_and_Ks[:,1].argmin()]

            total_Ks = []

            return list(best_path)

    def draw_best_path(self, graph_in, source, target, graph_out=None, color=None, is_already_flow_subgraph=False):
        
        if graph_out == None:
            graph_out = graph_in

        if color == None:
            color = "#BF0000"

        best_path = self.find_best_path(graph_in, source, target, is_already_flow_subgraph=is_already_flow_subgraph)[0]
        best_path_edges = []

        for i in range(len(best_path)-1):
            new_edge = (best_path[i], best_path[i+1])
            best_path_edges.append(new_edge)

        for u, v, data in graph_out.edges(data=True):
            for edge in best_path_edges:
                if (u, v) == edge:
                    data['color'] = color

        return graph_out

    def fuel_flow_path(self, graph, source):
        subg = nx.DiGraph(graph.copy())
        
        targets = list(graph.nodes.keys())
        targets.remove(source)
        
        for edge in subg.out_edges(source):
            dist = graph[source][edge[1]][0]['Mprop_Mpay']+1
            nx.set_node_attributes(subg, {edge[1]: dist}, name=f"dist_from_{source}")
        
        subg.clear_edges()
        
        for target in targets:
            subg.add_edge(source, target)
            
            subtargets = list(graph.nodes.keys())
            subtargets.remove(source)
            subtargets.remove(target)
            
            target_dist = subg.nodes[target][f"dist_from_{source}"]

            for subtarget in subtargets:
                if subg.nodes[subtarget][f"dist_from_{source}"] > target_dist:
                    subg.add_edge(target, subtarget)
            
        return subg

# ---------------------------------------------------------

    def calc_price_increase(self, edge, orig_node):

        #Get information from linked dicts
        vehicle_data = self.vehicles[edge['vehicle']]
        engine_data = self.engines[vehicle_data["engine"]]
        
        #Get engine parameters
        v_e = engine_data['Isp'] * self.global_vars['g0']
        thrust_vac = engine_data['thrust_vac']
        if "burn_lifetime" not in engine_data.keys():
            burn_lifetime = self.global_vars['burn_lifetime_default']
        
        #Get vehicle parameters
        initial_cost = vehicle_data['initial_cost']
        repair_factor_fixed = vehicle_data["repair_factor_fixed"]
        repair_factor_var = vehicle_data["repair_factor_var"]

        #Get node parameters
        if "best_price" not in orig_node.keys(): best_price = orig_node['fuel_price']
        else: best_price = orig_node['best_price']
        fuel_price = orig_node['fuel_price']

        #Get edge parameters
        dV = edge['dV']
        profit_margin = edge['profit_margin']
        Minit_Mpay = edge["Minit_Mpay"]
        Mprop_Mpay = edge["Mprop_Mpay"]
        Mdry_Mpay = edge["Mdry_Mpay"]

        #Calculate total new price
        Ctot_Mpay = initial_cost * ((v_e * (1 - m.exp(-(dV*1000)/v_e)) /(thrust_vac * burn_lifetime))* Minit_Mpay +\
            (repair_factor_fixed * (1 + repair_factor_var * dV * 1000) * Mdry_Mpay)) + best_price * Mprop_Mpay
        new_price = profit_margin * (fuel_price + Ctot_Mpay)

        return new_price

    def set_initial_price_conditions(self, subgraphs, local_fuel_prices):
        
        for fuel_source in subgraphs.keys():
            subgraph = subgraphs[fuel_source]

            nx.set_node_attributes(subgraph, {fuel_source: {"fuel_price": local_fuel_prices[fuel_source]}})
            
            for node in [sorted(generation) for generation in nx.topological_generations(subgraph)]:
                
                new_prices = {}

                current_node = subgraph.nodes[node[0]]
                children = subgraph.adj[node[0]]

                for child in children.keys():
                    edge = children[child]                    
                    new_price = self.calc_price_increase(edge, current_node)
                    
                    if child in new_prices.keys():
                        if new_price < new_prices[child]:
                            new_prices[child] = new_price
                        else:
                            pass
                    else:
                        new_prices[child] = new_price
                
                nx.set_node_attributes(subgraph, new_prices, "fuel_price")

    def get_best_prices(self, subgraphs):
        
        fuel_sources = list(subgraphs.keys())
        nodes = subgraphs[fuel_sources[0]].nodes.keys()
        best_prices = {}
        
        for node in nodes:
            best_price = min([subgraphs[subgraph].nodes[node]["fuel_price"] for subgraph in subgraphs.keys()])
            best_prices[node] = best_price
        
        for subgraph in subgraphs.keys():
            nx.set_node_attributes(subgraphs[subgraph], best_prices, name="best_price")

    def update_prices(self, subgraphs, merged=None):

        all_new_prices = {}
        changing = []
        for fuel_source in subgraphs.keys():
            subgraph = subgraphs[fuel_source]

            new_prices = {}
            for node in [sorted(generation) for generation in nx.topological_generations(subgraph)]:
                
                current_node = subgraph.nodes[node[0]]
                children = subgraph.adj[node[0]]

                for child in children.keys():

                    edge = children[child]                    
                    new_price = self.calc_price_increase(edge, current_node)

                    if child in new_prices.keys():
                        if new_price < new_prices[child]:
                            new_prices[child] = new_price
                    else:
                        new_prices[child] = new_price
            
            current_prices = nx.get_node_attributes(subgraph, "fuel_price")
            new_prices[fuel_source] = current_prices[fuel_source]

            diff = DeepDiff(current_prices, new_prices, significant_digits=3, ignore_numeric_type_changes=True)

            if diff == {}: delta = False
            else: delta = True

            nx.set_node_attributes(subgraph, new_prices, name="fuel_price")
        
        changing = any(changing)
        return changing
        
    def converge_prices(self, subgraphs, local_fuel_prices, merged=None):
        
        if merged==None:
            self.set_initial_price_conditions(subgraphs, local_fuel_prices)

            changing = True
            iteration = 1
            while changing:
                self.get_best_prices(subgraphs)
                
                changing = self.update_prices(subgraphs)
        
        else:

            fuel_sources = []    
            for u, v, data in subgraphs.edges(data=True):
                fuel_sources.append(data['fuel_source'])
            fuel_sources = set(fuel_sources)

            changing = True
            iteration = 1
            while changing:
                best_prices = {}

                for node in subgraphs.nodes:
                    best_price = min([subgraphs.nodes[node][f"fuel_price_{fuel_source}"] for fuel_source in fuel_sources])
                    best_prices[node] = best_price

                nx.set_node_attributes(subgraphs, best_prices, name="best_price")

                
                
                changing = []
                for fuel_source in fuel_sources:
                    subgraph = subgraphs
                    new_prices = {}
                    for node in [sorted(generation) for generation in nx.topological_generations(subgraph)]:
                        
                        current_node = subgraph.nodes[node[0]]
                        children = subgraph.adj[node[0]]

                        for child in children.keys():

                            edge = children[child]                    
                            new_price = current_node["best_price"] * edge["Mprop_Mpay"] + current_node["fuel_price"]

                            if child in new_prices.keys():
                                if new_price < new_prices[child]:
                                    new_prices[child] = new_price
                            else:
                                new_prices[child] = new_price
                    
                    current_prices = nx.get_node_attributes(subgraph, "fuel_price")
                    new_prices[fuel_source] = current_prices[fuel_source]

                    diff = DeepDiff(current_prices, new_prices, significant_digits=3, ignore_numeric_type_changes=True)

                    if diff == {}: delta = False
                    else: delta = True

                    nx.set_node_attributes(subgraph, new_prices, name="fuel_price")
                
                changing = any(changing)
                return changing
            
# ---------------------------------------------------------

    def graph_to_pyvis(self, graph, net=None, group=None, merged=None):
        
        
        if net == None:
            net = Network(directed=True, height="700px", width="900px", bgcolor='#222222', font_color='white')
        else:
            net = net

        if not merged:
            for node in graph.nodes.keys():

                if "fuel_price" in graph.nodes[node].keys():

                    net.add_node(node,
                    label = node+"\n"+str(round(graph.nodes[node]['fuel_price']/1000,1))+" k$",
                    title = json.dumps(graph.nodes[node]),
                    group = group,
                    size = graph.nodes[node]['fuel_price']/500
                    )
                
                else:
                    net.add_node(node,
                    title = json.dumps(graph.nodes[node]),
                    group = group
                )

            for edge in graph.edges.items():
                attrs = edge[1]
                edge_item = edge[0]

                full_label = f"dV:{round(attrs['dV'], 2)}\n"+"k: "+str(round(attrs['Mprop_Mpay'], 2))

                if "color" in attrs.keys():

                    net.add_edge(edge_item[0], edge_item[1],
                    title = json.dumps(attrs),
                    label = str(full_label),
                    value = attrs['Mprop_Mpay']*10,
                    color = attrs['color']
                    )
                
                else:
                    net.add_edge(edge_item[0], edge_item[1],
                        title = json.dumps(attrs),
                        label = str(full_label),
                        value = attrs['Mprop_Mpay']*10
                    )

        elif merged:

            fuel_sources = []    
            for u, v, data in graph.edges(data=True):
                fuel_sources.append(data['fuel_source'])
            fuel_sources = set(fuel_sources)

            for node in graph.nodes.keys():

                net.add_node(node,
                label = node+"\n"+str(f"{graph.nodes[node]['cheaper_source']}: ")+str(round(graph.nodes[node]['best_price']/1000,1))+" k$",
                title = attr_dict_to_str(graph.nodes[node]),
                size = graph.nodes[node]['best_price']/250,
                color = graph.nodes[node]['color'],
                font='50px arial black'
                )

            for edge in graph.edges.items():
                attrs = edge[1]
                edge_item = edge[0]

                full_label = f"dV:{round(attrs['dV'], 2)}km/s\n"+"k: "+\
                    str(round(attrs['Mprop_Mpay'], 2))

                for attr in attrs.keys():
                    if type(attrs[attr]) == float:
                        attrs[attr] = round(attrs[attr],2)

                net.add_edge(edge_item[0], edge_item[1],
                    title = attr_dict_to_str(attrs),
                    label = str(full_label),
                    width = attrs['Mprop_Mpay']*5,
                    color = attrs['color'],
                    arrowStrikethrough = False
                )
            

        return net

    def merge_subgraphs(self, subgraphs):

        marked_subgraphs = []

        color_palette = {}
        colors = ["#BF0000", "#0000BF", "#00BF00"]

        for fuel_source in subgraphs.keys():

            color_palette[f"{fuel_source}"] = colors.pop()

            subgraph = subgraphs[fuel_source]

            new_subgraph = subgraph.copy()

            for u, data in new_subgraph.nodes(data=True):
                data[f"fuel_price_{fuel_source}"] = data.pop("fuel_price")

            for u, v, data in new_subgraph.edges(data=True):
                data["fuel_source"] = f"{fuel_source}"

            marked_subgraphs.append(new_subgraph)

        for i in range(len(marked_subgraphs)-1):
            marked_subgraphs[0].update(marked_subgraphs[i+1])

            merged_graph = marked_subgraphs[0]

        for fuel_source in subgraphs.keys():

            for u, data in merged_graph.nodes(data=True):
                if round(data[f"fuel_price_{fuel_source}"],2) == round(data['best_price'],2):
                    data['cheaper_source'] = f"{fuel_source}"
                    data['color'] = color_palette[f'{fuel_source}']

        # st.write([(node, merged_graph.nodes[node]) for node in merged_graph.nodes if "cheaper_source" not in merged_graph.nodes[node].keys()])    

        for u, v, data in merged_graph.edges(data=True):
            data["color"] = color_palette[data['fuel_source']]

        return merged_graph





class Price_Sim:

    """ A class containing all of the data and methods for the global simulation. This is the class that interfaces with Streamlit in order to configure\
        all the parameters needed for the simulation.
    
    Attributes:
        
        """

    def __init__(self):

        """Initialize instance with X,Y,Z.
            Args:"""
        
        self.update_attrs_from_disk()

    def save_attribute_to_json(self, attribute):
        if not type(getattr(self, attribute)) == dict:
            raise Exception("Saved attribute must be a dict!")
        dict_out = getattr(self, attribute)
        with open(f"{attribute}.json", "w") as outfile:
            json.dump(dict_out, outfile)

    def update_attr_from_json(self, attribute, filename):
        if not type(getattr(self, attribute)) == dict:
            raise Exception("Updated attribute must be a dict!")
        dict_new = load_dict_from_json(filename)
        getattr(self, attribute).update(dict_new)

    def update_attrs_from_disk(self, attribute_list=None):

        #Create all the dicts that will hold our info and update them from disk
        subfolder_path = get_subfolder_path("data")
        
        if attribute_list == None:

            for var_name in ["global_var_distros",
                            "local_var_distros",
                            "global_vars",
                            "events",
                            "propellants",
                            "engines",
                            "vehicles",
                            "trajectory_data"]:
                setattr(self, var_name, {})
                self.update_attr_from_json(var_name, subfolder_path +"/"+ var_name)
        
        else:

            for var_name in attribute_list:
                setattr(self, var_name, {})
                self.update_attr_from_json(var_name, subfolder_path +"/"+ var_name)

    def graph_drawer(self, old_graph=None, dict_of_dicts=None):

        #notice changes in nodes, trajectory data (thus vehicles), aerobraking constant
        if old_graph != None:
            nodes_changed = old_graph.g_main.nodes.keys() != self.global_vars["active_nodes"]
            trajectory_data_changed = old_graph.trajectory_data != dict_of_dicts

            if any([nodes_changed, trajectory_data_changed]):

                self.graph = Graph(dict_of_dicts)
                self.graph.vehicles = self.vehicles
                self.graph.global_vars = self.global_vars
                
                self.graph.update_nodes(self.graph.g_main, self.global_vars["active_nodes"])

                self.graph.assign_mass_fractions(self.graph.g_main)

                self.graph.subgraphs = self.graph.best_fuel_flow_subgraphs(graph=self.graph.g_main, sources=self.graph.g_main.nodes.keys(), fuel_sources=self.global_vars["fuel_sources"])
        
            else:
                self.graph = old_graph
        
        elif old_graph == None:
            self.graph = Graph(dict_of_dicts)
            
            self.graph.propellants = self.propellants
            self.graph.engines = self.engines
            self.graph.vehicles = self.vehicles
            self.graph.global_vars = self.global_vars
            
            self.graph.update_nodes(self.graph.g_main, self.global_vars["active_nodes"])
            
            self.graph.assign_mass_fractions(self.graph.g_main)

            self.graph.subgraphs = self.graph.best_fuel_flow_subgraphs(graph=self.graph.g_main, sources=self.graph.g_main.nodes.keys(), fuel_sources=self.global_vars["fuel_sources"])

    
    

class Sim_Iteration(Price_Sim):
    """ A class containing all of the data and methods for the current Monte Carlo sampling iteration in the simulation.
    Each iteration contains all the time periods of the total time being simulated.
    
    Attributes:
        
        """
    def __init__(self):
        
        """Initialize instance with X,Y,Z.
            Args:""" 

        self.global_var_samples = self.sample_distros(self.global_var_distros)

    def sample_distros(self, distros):
        global_var_samples = {}
        return global_var_samples

    def simulate_over_time(self, start_year, end_year):
        self.system_state_time_series = pd.DataFrame()


        for year in range(start_year, end_year):

            year_sim = Timeperiod(year)

            self.system_state_time_series.append(year_sim.system_state, ignore_index=True)

        

class Timeperiod(Sim_Iteration):
    """ A class containing all of the data and methods for the current timestep in the simulation.
    
    Attributes:
        
        """
    
    def __init__(self,
        year):
        
        self.year = year

        self.local_var_samples = self.sample_distros(self.local_var_distros)

        self.graph_drawer(old_graph=self.graph, dict_of_dict=self.trajectory_data)
        self.graph.converge_prices(self.graph.subgraphs, local_fuel_prices=self.local_var_samples.local_fuel_prices)

        self.system_state = self.graph.system_state_to_dict(self.graph.subgraphs, self.year)

    

