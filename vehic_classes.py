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
from graph_classes import *


class Propellant:
    """ A type of rocket propellant and the tanks that hold it. For now, it is assumed to be chemical
    (but if you have another kind of propellant that e.g. does not use oxidizer, then just set that to zero.)
    
    Attributes:
        fuel=None,
        oxidizer=None,
        fuel_density=None,
        oxidizer_density=None,
        f_fueltank=None,
        f_oxtank=None,
        f_ullage=None """

    def __init__(self,
        name,
        fuel=None,
        oxidizer=None,
        fuel_density=None,
        oxidizer_density=None,
        f_fueltank=None,
        f_oxtank=None,
        f_ullage=None) :

        self.name = name
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.fuel_density = fuel_density
        self.oxidizer_density = oxidizer_density
        self.f_fueltank = f_fueltank
        self.f_oxtank = f_oxtank
        self.f_ullage = f_ullage

class Engine:
    """ A rocket engine model that uses a certain kind of propellant.
    Keep in mind that non-matching engine-propellant combinations are not prohibited but will create unrealistic models.
    
    Attributes:
        propellant: the propellant used by the engine.
        thrust_vac: the thrust of the engine in vacuum [kN].
        Isp: the specific impulse of the engine [s].
        dry_mass: the propellantless mass of the engine [kg].
        MXR: the mixture rate (ratio) between the oxidizer and fuel masses (m_ox/m_fuel)"""

    def __init__(self,
        name,
        propellant: Propellant,
        thrust_vac=None,
        Isp=None,
        dry_mass=None,
        MXR=None,
        f_TSW = 0.003):

        self.name = name
        self.propellant = propellant
        self.thrust_vac = thrust_vac
        self.Isp = Isp
        self.dry_mass = dry_mass
        self.MXR = MXR
        self.f_TSW = f_TSW

    

class Vehicle():
    """ Create a spacecraft that uses a certain engine and propellant.
    
    Attributes:
        
        """

    def __init__(self,
        name,
        engine: Engine,
        T2W = None,
        aerobraking = None,
        directionality = None,
        ):

        self.name = name
        self.engine = engine
        self.T2W = T2W
        self.g0 = 9.807 # m/s2
        self.aerobraking_init_mass_increase = 1.15

        self.gross_sens_term = None
        # self.M_init = (self.engine.thrust_vac*1000) / (self.T2W * g0) Leave this for later

    def calc_prop_sens_term(self):
        return (self.engine.MXR * (self.engine.propellant.f_oxtank/self.engine.propellant.oxidizer_density)+ (self.engine.propellant.f_fueltank/self.engine.propellant.fuel_density))/((1+self.engine.MXR)*(1-self.engine.propellant.f_ullage))

    def calc_gross_sens_term(self, T2W, f_TSW, T2W_eng):
        return T2W * (1 + f_TSW* T2W)/T2W_eng

    def calc_prop_pay(self, prop_sens_term, gross_sens_term, mass_frac, directionality, aerobraking):
        
        if directionality == 1:
            return (1- 1/mass_frac) * (1/(1/mass_frac)) * (1- (gross_sens_term + prop_sens_term)*mass_frac + prop_sens_term) * (aerobraking * self.aerobraking_init_mass_increase)

        elif directionality == 2:
            return ((1- (1/mass_frac**2)) * (1/(mass_frac * (1 - prop_sens_term*(mass_frac - 1)))) * (1 - (prop_sens_term + gross_sens_term)*mass_frac**2 + prop_sens_term) * (aerobraking * self.aerobraking_init_mass_increase) ) - (1- (1/mass_frac))


    def size_vehicle(self):

        self.calc_prop_sens_term()
        self.calc_gross_sens_term()


    