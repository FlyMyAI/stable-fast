import logging
import torch.nn as nn
from torch._prims_common import suggest_memory_format
from .. import torch_ops as TTO

logger = logging.getLogger()


class TritonLoRACompatibleConv(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def set_lora_layer(self, lora_layer):
        if hasattr(self.module, 'set_lora_layer'):
            self.module.set_lora_layer(lora_layer)

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonLoRACompatibleLinear(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def set_lora_layer(self, lora_layer):
        if hasattr(self.module, 'set_lora_layer'):
            self.module.set_lora_layer(lora_layer)

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)
