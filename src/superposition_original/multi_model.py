from dataclasses import dataclass, replace
from typing import Optional

import einops
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

# @dataclass
# class ManyModelConfig:
#     n_feat: int
#     n_hidden: int
#     # train many models with different 
#     # sparsity in one go
#     n_model: int

class MultiModel(nn.Module):
    def __init__(self,
                n_model,
                n_feat,
                n_hidden,
                # feat_prob: Optional[torch.Tensor]=None,
                # importance: Optional[torch.Tensor]=None,
                device='cuda'):

        # 1 - feat_prob  = sparsity_prob

        super().__init__()
        self.w_3t = nn.Parameter(
            torch.empty(
                (n_model,
                n_feat,
                n_hidden),
                device=device
            )
        )

        nn.init.xavier_normal_(self.w_3t)
        self.b_3t = nn.Parameter(
            torch.zeros(
                (n_model, 1 ,n_feat),
                device=device
            )
        )

        # if feat_prob is None:
        #     feat_prob = torch.ones(())
        # self.feat_prob = feat_prob.to(device)

        # if importance is None:
        #     importance = torch.ones(())
        # self.importance = importance.to(device)


    def forward(self, feat_3t):
        # feat_3t: [n_model, n_data_points, n_feat]
        # self.w_3t: [n_model, n_feat, n_hidden]

        # multi_model: y = RELU(x * w * w_T  + b)
        
        hidden_map_3t = torch.einsum(
            "mdf,mfh->dmh", # on feat
            feat_3t, 
            self.w_3t
        )

        # hidden_3t: [n_model, n_data_points, n_hidden]
        # self.w_3t: [n_model, n_feat, n_hidden]

        # no need transpose with einsum method
        feat_map_3t = torch.einsum(
            "dmh,mfh->mdf", #  on n_hidden, -> (n_model, n_data, n_feat)
            hidden_map_3t,
            self.w_3t
        )

        # b_3t = (n_model, 1,  n_feat)
        add_bias = feat_map_3t + self.b_3t
        cut_off = F.relu(add_bias)

        return cut_off

def generate_data( 
    n_model: int,

    n_data_point: int,

    n_feat: int,
    feat_prob_3t: Float[Tensor, "n_model n_data_point n_feat"],

    SEED:int=0,
    device: str = "cuda",
) -> Float[
    Tensor, "n_model n_data_point n_feat"
]:
    # torch.manual_seed(SEED)

    x_3t = torch.rand(
        (n_model,
            n_data_point,
            n_feat),
            device=device
    )

    filter_cond_3t = torch.rand(
        (n_model, n_data_point, n_feat),
        device=device
    ) <= feat_prob_3t

    sparse_x_3t = torch.where(
        filter_cond_3t,
        x_3t,
        torch.zeros((), device=device),
    )

    return sparse_x_3t



