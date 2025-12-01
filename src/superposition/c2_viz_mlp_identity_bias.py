import matplotlib.pyplot as plt
from src.superposition.training import \
    load_model
import numpy as np

if __name__ == "__main__":
    sparse_list = [0, 0.7, 0.9, 0.99, 0.999]

    fig, ax = plt.subplots(
        1, 10,
        tight_layout=True,
        figsize=(15, 3),
        gridspec_kw={
            'width_ratios': [20, 1] * 5
        }       
    )
    
    for i, sparsity in enumerate(sparse_list):
        file_path_str = (
            'src/superposition/weights/'
            f'mlp/s_{sparsity}/mlp_final.pt'
        )

        model_dict = load_model(
            file_path_str=file_path_str
        )   

        w1 = model_dict['w1_2t']
        # w1 = (20, 5)
        w2 = model_dict['w2_2t']
        # w2 = (5, 20)
        w1_dot_w2 = w1 @ w2
        
        b = model_dict['b_1t']

        w1_fig_index = 2 * i 
        b_fig_index = w1_fig_index + 1

        im = ax[w1_fig_index].imshow(
            w1_dot_w2,
            vmin=-1.3,
            vmax=1.3,
            cmap="RdBu"
        )

        ax[w1_fig_index].set_title(
            f'1-S={1-sparsity:.03f}'
        )

        b_max = np.abs(b).max() * 1.3
        ax[b_fig_index].imshow(
            b.reshape(-1,1),
            vmin=-b_max,
            vmax=b_max,
            cmap="RdBu"
        )

        ax[w1_fig_index].set_xticks([])
        ax[w1_fig_index].set_yticks([])


        ax[b_fig_index].set_xticks([])
        ax[b_fig_index].set_yticks([])

    plt.savefig(
        f'src/superposition/viz/c2_mlp_identity_n_bias.png'
    )

