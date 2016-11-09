#!/usr/bin/env python

import src.network as network

from . import args

class NeuralNet:
    def __init__(self):
        self.args = args

    def run(self):
        network.startNetwork(self)
