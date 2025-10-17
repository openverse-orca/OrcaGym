import sys
import os
import grpc
import numpy as np
import json
from datetime import datetime

class OrcaGymOptConfig:

    def __init__(self, opt_config: dict):
        self.opt_config = opt_config.copy()
        self.timestep = opt_config['timestep']
        self.apirate = opt_config['apirate']
        self.impratio = opt_config['impratio']
        self.tolerance = opt_config['tolerance']
        self.ls_tolerance = opt_config['ls_tolerance']
        self.noslip_tolerance = opt_config['noslip_tolerance']
        self.ccd_tolerance = opt_config['ccd_tolerance']
        self.gravity = opt_config['gravity']
        self.wind = opt_config['wind']
        self.magnetic = opt_config['magnetic']
        self.density = opt_config['density']
        self.viscosity = opt_config['viscosity']
        self.o_margin = opt_config['o_margin']
        self.o_solref = opt_config['o_solref']
        self.o_solimp = opt_config['o_solimp']
        self.o_friction = opt_config['o_friction']
        self.integrator = opt_config['integrator']
        self.cone = opt_config['cone']
        self.jacobian = opt_config['jacobian']
        self.solver = opt_config['solver']
        self.iterations = opt_config['iterations']
        self.ls_iterations = opt_config['ls_iterations']
        self.noslip_iterations = opt_config['noslip_iterations']
        self.ccd_iterations = opt_config['ccd_iterations']
        self.disableflags = opt_config['disableflags']
        self.enableflags = opt_config['enableflags']
        self.disableactuator = opt_config['disableactuator']
        self.sdf_initpoints = opt_config['sdf_initpoints']
        self.sdf_iterations = opt_config['sdf_iterations']
        self.filterparent = opt_config.get('filterparent', True)

