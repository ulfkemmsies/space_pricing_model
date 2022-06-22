from configparser import NoOptionError
from xmlrpc.client import Boolean
from scipy import stats
import numpy as np
import pandas as pd
from itertools import chain
from operator import attrgetter
import math as m

from sim_classes import *
from general_funcs import *
from vehic_classes import *


class Location:
    """ A location in cislunar space with a specific gravitational pull.
    
    Attributes:
        
        """

    def __init__(self, name, g = None):
        self.name = name
        if (g is not None) and (g != 0):
            self.type = "body"
            self.g = g
        elif g == 0:
            self.type = "orbit"

class Node:

    def __init__(self,
        name,
        location: Location,
        is_fuel_source = False,
        ):

        self.name = name
        self.location = location
        self.is_fuel_source = is_fuel_source
        self.storage_space = None

class Edge:


    def __init__(self,
        origin: Node,
        target: Node,
        dV = None,
        aerobraking = False):

        self.origin = origin
        self.target = target
        self.dV = dV
        self.aerobraking = aerobraking
        self.name = str(origin.name) +","+str(target.name)

        self.prop_pay = None
        

