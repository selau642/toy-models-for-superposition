import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from jaxtyping import Float
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm

from src.superposition.generate_x import generate_sparse_x_2t, normalize_matrix


def train_toy_model(
    toy_model,

    save_folder_str: str,
    save_model_name:str,

    n_epoch:int,
    n_feat:int,
    n_data:int,
    sparsity: float,

    lr:float=0.001,
    weight_decay:float=0.01,
    device: str = "cuda",

) -> Tuple[List[float], Dict[str, Tensor]]:
    """

    train model and 
    returns training_lost_list, 

    """
    toy_model = toy_model.to(device)
    optim = torch.optim.AdamW(
        toy_model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    epoch_loss_list = []
    # lost for each epoch

    weight_t = torch.tensor([
        0.7 ** i for i in range(n_feat)
    ]).to(device)

    for i in tqdm(range(n_epoch)):
        sparse_x_2t = generate_sparse_x_2t(
            n_feat=n_feat,
            n_data=n_data,
            frac_is_zero=sparsity
        )
        # sparse_x_2t = (data, feat)

        norm_x_2t = normalize_matrix(sparse_x_2t)

        norm_x_2t = norm_x_2t.to(device)
        actual_2t = norm_x_2t.to(device)

        optim.zero_grad()
        pred_2t: Float[Tensor, 'n_feat n_data'] = toy_model(
            norm_x_2t.float()
        )
        loss_t = mean_weighted_square_error_loss(
            pred_2t=pred_2t,
            actual_2t=actual_2t,
            weight_t=weight_t
        )

        loss_t.backward()
        optim.step()
        epoch_loss_list.append(loss_t.item())

        if i % 2000 == 0:
            # print('epoch: ', i)
            # print('loss: ', loss_t.item())

            file_path_str = f'{save_folder_str}/{save_model_name}_{i}.pt'

            save_model(
                pt_model_state_dict=toy_model.state_dict(),
                file_path_str=file_path_str
                )

    final_model_state_dict = toy_model.state_dict()
    final_file_path_str = f'{save_folder_str}/{save_model_name}_final.pt'

    epoch_loss_json_str = f'{save_folder_str}/epoch_loss_list.json'

    epoch_loss_dict = {
        'epoch_loss_list': epoch_loss_list
    } 

    save_dict_to_json(
        json_file_name_str=epoch_loss_json_str,
        data_dict=epoch_loss_dict
    )

    save_model(
        pt_model_state_dict=final_model_state_dict,
        file_path_str=final_file_path_str,
    )

    # save checkpoint
    return epoch_loss_list, final_model_state_dict

    
def mean_weighted_square_error_loss(
    pred_2t: Float[Tensor, 'n_feat n_data'],
    actual_2t: Float[Tensor, 'n_feat n_data'],
    weight_t: Float[Tensor, 'n_feat']
) -> Float[Tensor, '1']:
    sqr_err_loss_t = (actual_2t - pred_2t) ** 2
    w_se_loss_t = weight_t * sqr_err_loss_t
    # check above
    mw_se_loss_t = w_se_loss_t.mean()
    return mw_se_loss_t


def save_model(
    pt_model_state_dict,
    file_path_str:str,
    metadata_dict: Optional[
        Dict[str, Any]
    ]=None
):   
    save_file(
        tensors=pt_model_state_dict,
        filename=file_path_str,
        metadata=metadata_dict
    )


def load_model(
    file_path_str: str
) -> Dict[str, Any]:
    tensor_dict = {}

    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in f.keys():
            tensor_dict[key] = f.get_tensor(key)

    return tensor_dict


def load_partial_model(
    file_path_str: str,
    model_key_list: List[str]
) -> Dict[str, Any]:
    tensor_dict = {}
    with safe_open(
        file_path_str,
        framework='pt',
        device='cpu'
    ) as f:
        for key in model_key_list:
            tensor_dict[key] = f.get_tensor(key)
    
    return tensor_dict


def save_dict_to_json(
    json_file_name_str: str,
    data_dict: Dict[str, Any]
):
    with open(json_file_name_str, 'w') as f:
        json.dump(
            data_dict, f, 
            ensure_ascii=False, 
            indent=4
        )


def load_dict_from_json(
    json_file_name_str: str,
):
    with open(json_file_name_str, 'r') as f:
        data_dict = json.load(f)
    return data_dict


if __name__ == "__main__":
    data_dict = {"test": ['a', 12]}
    json_file_name_str = 'test.json'

    save_dict_to_json(
        json_file_name_str=json_file_name_str,
        data_dict=data_dict
    )

    data_from_dict = load_dict_from_json(
        json_file_name_str=json_file_name_str
    )

    print(data_from_dict)