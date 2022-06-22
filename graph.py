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

from general_funcs import *

class Graph:

    def __init__(self, dict_of_dict=None):
        
        if dict_of_dict == None:
            self.g_main = nx.MultiDiGraph()
        elif type(dict_of_dict) == dict:
            self.g_main = nx.MultiDiGraph(nx.from_dict_of_dicts(dict_of_dict))
            self.trajectory_data = dict_of_dict

        self.aerobraking_mass_penalty = None
    
    def update_nodes(self, graph, nodes):
        present_nodes = list(graph.nodes.keys())
        nodes_to_keep = present_nodes & nodes
        nodes_to_delete = set(present_nodes) - nodes_to_keep
        graph.remove_nodes_from(list(nodes_to_delete))

    def assign_prop_pay(self, graph):
        for edge in graph.edges.items():
            attrs = edge[1]
            print(attrs['vehicle'])
            edge_item = edge[0]
            vehicle = self.vehicles[attrs.vehicle]

            self.prop_sens= vehicle.prop_sens
            self.gross_sens = vehicle.gross_sens
            self.directionality = attrs.directionality
            self.aerobraking = vehicle.aerobraking
            self.Isp = vehicle.Isp
            self.dV = attrs.dV
                
            mass_frac = m.exp((1000*self.dV)/(9.807*self.Isp))
            
            prop_pay = self.calc_prop_pay(self.prop_sens, self.gross_sens, mass_frac, self.directionality, self.aerobraking)
            
            attr_dict = {edge_item: {"prop_pay": prop_pay}}

            nx.set_edge_attributes(self.g_main, attr_dict)
        
    def calc_prop_pay(self, prop_sens_term, gross_sens_term, mass_frac_term, direct, aerobraking):

        if direct == 1:
            return (1- 1/mass_frac_term) * (1/(1/mass_frac_term)) * (1- (gross_sens_term + prop_sens_term)*mass_frac_term + prop_sens_term) * (aerobraking * 1.15)

        elif direct == 2:
            return ((1- (1/mass_frac_term**2)) * (1/(mass_frac_term * (1 - prop_sens_term*(mass_frac_term - 1)))) * (1 - (prop_sens_term + gross_sens_term)*mass_frac_term**2 + prop_sens_term) * (aerobraking * 1.15) ) - (1- (1/mass_frac_term))
    
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
            
    def find_best_path(self, graph, source, target, fuel_source=None):
        total_Ks = []
        
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
        
        for path in nx.all_simple_paths(source_flow_graph, source, target):
            total_mass = reduce(mul, ((graph[start][end][0]['prop_pay']+1) for start, end in zip(path[:-1], path[1:])), 1)

            for start, end in zip(path[:-1], path[1:]):
                total_Ks.append([path, total_mass])

            paths_and_Ks = np.array(total_Ks)
            best_path = paths_and_Ks[paths_and_Ks[:,1].argmin()]
            total_Ks = []

            return list(best_path)
    
    def fuel_flow_path(self, graph, source):
        subg = nx.DiGraph(graph.copy())
        
        targets = list(graph.nodes.keys())
        targets.remove(source)
        
        for edge in subg.out_edges(source):
            dist = graph[source][edge[1]][0]['prop_pay']+1
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
                    new_price = current_node["fuel_price"] * (edge["prop_pay"] + 1)
                    
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

    def update_prices(self, subgraphs):

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
                    new_price = current_node["best_price"] * edge["prop_pay"] + current_node["fuel_price"]

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
        
    def converge_prices(self, subgraphs, local_fuel_prices):
        self.set_initial_price_conditions(subgraphs, local_fuel_prices)

        changing = True
        iteration = 1
        while changing:
            self.get_best_prices(subgraphs)
            
            changing = self.update_prices(subgraphs)
    
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

    
                

    