import torch
from .registry import Registry
# from models.mambaModels.mambaConfig import Mamba2Config

META_ARCHITECTURE = Registry(name="ARCHITECTURE")

def build_model(config, vocab):
    model = META_ARCHITECTURE.get(config.architecture)(config, vocab)
    model = model.to(torch.device(config.device))
    
    return model

# def build_mamba(config, vocab):
#     mambaconfig = Mamba2Config(d_model=config.model.d_model, n_layers=config.model.n_layers, d_head=config.model.head, mup=True, mup_base_width=64)
#     model = META_ARCHITECTURE.get(config.architecture)(mambaconfig, vocab)
#     model = model.to(torch.device(config.device))
    
#     return model