from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter
import random


class MPCDNet(nn.Module):
    """
    S & T Models
    """
    def __init__(
        self,
        s_network,
        t_network,
        m=0.98,
        
    ):
        super(MPCDNet, self).__init__()
        self.m = m

        # create the encoders
        self.s_network = s_network
        self.t_network = t_network

        # freeze key model
        self.t_network.requires_grad_(False)

    @torch.no_grad()
    def _ema_model_update(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.s_network.parameters(), self.t_network.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, x):
        if self.training:
            ## Compute query features
            s_output = self.s_network(x)

            ## EMA update of the Teacher Model
            with torch.no_grad():
                self._ema_model_update()

            # dequeue and enqueue will happen outside
            return s_output
        else:
            return self.t_network(x)
    
    @property
    def decoder(self):
        return self.s_network.decoder