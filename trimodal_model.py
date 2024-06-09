import torch
import torch.distributed as dist
import torchmetrics
import json
import math
import numpy as np
import os
import copy
import time
import pandas as pd
import random

from tqdm import tqdm
from protein_encoder import ProteinEncoder
from molecule_encoder import SmallMoleculeEncoder
from text_encoder import TextEncoder
from abstract_model import AbstractModel
from model_interface import register_model
from torch.nn.functional import normalize, cross_entropy
from sklearn.metrics import roc_auc_score


def multilabel_cross_entropy(logits, labels):
    """
    Compute cross entropy loss for multilabel classification。 See "https://arxiv.org/pdf/2208.02955.pdf"
    Args:
        logits: [num_samples, num_classes]
        labels: [num_samples, num_classes]
    """

    loss = 0
    for pred, label in zip(logits, labels):
        pos_logits = pred[label == 1]
        neg_logits = pred[label == 0]

        diff = neg_logits.unsqueeze(-1) - pos_logits
        loss += torch.log(1 + torch.exp(diff).sum())

    return loss / len(logits)

@register_model
class TrimodalModel(AbstractModel):
    def __init__(self,
                 protein_config: str,
                 text_config: str,
                 sm_config: str,
                 repr_dim: int = 256,
                 temperature: float = 1.0,
                 gradient_checkpointing: bool = False,
                 **kwargs):
        self.protein_config = protein_config
        self.text_config = text_config
        self.sm_config = sm_config
        self.repr_dim = repr_dim
        self.temperature = temperature
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(**kwargs)
    
    def initialize_metrics(self, stage: str) -> dict:
        return_dict = {
            f"{stage}_protein_sm_acc": torchmetrics.Accuracy(),
            f"{stage}_sm_text_acc": torchmetrics.Accuracy(),
        }

        return return_dict

    def initialize_model(self):
        # Initialize encoders
        self.protein_encoder = ProteinEncoder(self.protein_config,
                                              self.repr_dim,
                                              self.gradient_checkpointing)
        
        self.text_encoder = TextEncoder(self.text_config,
                                        self.repr_dim,
                                        self.gradient_checkpointing)
        
        self.sm_encoder = SmallMoleculeEncoder(self.sm_config,
                                              self.repr_dim,
                                              self.gradient_checkpointing)

        # Learnable temperature
        self.temperature = torch.nn.Parameter(torch.tensor(self.temperature))
        
        # self.model is used for saving and loading
        self.model = torch.nn.ParameterList([self.temperature,
                                             self.protein_encoder,
                                             self.text_encoder,
                                             self.sm_encoder])
    
    def get_text_repr(self, texts: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        return self.text_encoder.get_repr(texts, batch_size, verbose)
    
    def get_sm_repr(self, molecules: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        return self.sm_encoder.get_repr(molecules, batch_size, verbose)
    
    def get_protein_repr(self, proteins: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        return self.protein_encoder.get_repr(proteins, batch_size, verbose)

    def forward(self, protein_inputs: dict, text_inputs: dict, sm_inputs: dict):
        """
        Args:
            protein_inputs: A dictionary for protein encoder
            structure_inputs: A dictionary for structure encoder
            text_inputs   : A dictionary for text encoder
        """
        protein_repr = self.protein_encoder(protein_inputs)
        text_repr = self.text_encoder(text_inputs)
        sm_repr = self.sm_encoder(sm_inputs)
        outputs = [protein_repr, text_repr, sm_repr]
        return outputs

    def loss_func(self, stage: str, outputs, labels):
        text_repr, protein_repr, sm_repr = outputs

        text_repr = normalize(text_repr, dim=-1)
        protein_repr = normalize(protein_repr, dim=-1)
        sm_repr = normalize(sm_repr, dim=-1)

        pairs = [
            ["protein", "sm"],
            ["sm", "text"]
        ]

        loss_list = []
        for k, v in pairs:
            sim = eval(f"{k}_repr") @ eval(f"{v}_repr").T
            sim = sim / self.temperature
            loss = cross_entropy(sim, labels)
            
            self.metrics[stage][f"{stage}_{k}_{v}_acc"].update(sim.detach(), labels)
            loss_list.append(loss)

        loss = sum(loss_list) / len(loss_list)
        
        # if stage == "train":
        log_dict = self.get_log_dict(stage)
        log_dict[f"{stage}_loss"] = loss
        self.log_info(log_dict)
        
        self.reset_metrics(stage)
        
        return loss

