import sys
import os
import grpc
import numpy as np

class OrcaGymData:
    def __init__(self, model):
        self.model = model
        self.qpos = np.zeros(model.nq)
        self.qvel = np.zeros(model.nv)

    def update(self, qpos, qvel):
        self.qpos = qpos
        self.qvel = qvel
