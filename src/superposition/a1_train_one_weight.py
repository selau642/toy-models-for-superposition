from pathlib import Path
from typing import List

import torch

from src.superposition.toynet import OneWeightLinearNet
from src.superposition.training import train_toy_model

if __name__ == "__main__":
    sparsity_list = [0, 0.7, 0.9, 0.99, 0.999]
    n_data = 1024
    n_feat = 20
    n_hidden = 5
    device = "cpu"  
    assert device in ("cpu", "cuda", "mps")

    two_w_net_name = 'one_w_net'

    # y = RELU(x * w1 * w2  + b)
    two_w_net = OneWeightLinearNet(
        n_feat=n_feat,
        n_hidden=n_hidden
    )

    # test run 
    n_epoch = 10_000

    SEED = 0
    torch.manual_seed(SEED)

    for sparsity in sparsity_list:

        two_w_net_folder_path_str = f'src/superposition/weights/one_w/s_{sparsity}'

        folder_path = Path(two_w_net_folder_path_str)
        folder_path.mkdir(parents=True, exist_ok=True)

        print("Training OneWeightLinearNet with sparsity: ", sparsity)

        train_lost_list, final_model_state_dict = train_toy_model(
            toy_model=two_w_net,

            save_folder_str=two_w_net_folder_path_str,
            save_model_name=two_w_net_name,

            n_epoch=n_epoch,
            n_data=n_data,    
            n_feat=n_feat,
            sparsity=sparsity,
            device=device,
        )

