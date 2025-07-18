import sys
import os
import grpc
import numpy as np

class OrcaGymData:
    def __init__(self, model):
        self.model = model
        self.qpos = np.zeros(model.nq)
        self.qvel = np.zeros(model.nv)
        self.qacc = np.zeros(model.nv)
        self.qfrc_bias = np.zeros(model.nv)
        self.time = 0

    def update_qpos_qvel_qacc(self, qpos, qvel, qacc):
        self.qpos = qpos
        self.qvel = qvel
        self.qacc = qacc

    def update_qfrc_bias(self, qfrc_bias):
        self.qfrc_bias = qfrc_bias

