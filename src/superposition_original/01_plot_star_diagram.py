from matplotlib import colors as mcolors
from matplotlib import collections as mc
import matplotlib.pyplot as plt
from cycler import cycler

import numpy as np

from src.model_io import load_model

def plot_star_diagram():
    

    multi_model_state_dict = load_model(
        # file_path_str="src/superposition_original/weights/multi_model.pth"
        file_path_str="src/superposition_original/weights/small_5_to_2_model.safetensors"
    )

    w_3t = multi_model_state_dict['w_3t'].detach()

    n_model, n_feat, n_hidden = w_3t.shape
    # n_model=10, n_feat=5, n_hidden=2

    feat_weight_np= (
        0.9 ** np.arange(n_feat)
    )[:]

    plt.rcParams["axes.prop_cycle"] = \
        cycler(
            "color",
            plt.cm.viridis(
                feat_weight_np                
            )
        )    
    
    plt.rcParams["figure.dpi"] = 200

    fig, ax_list = plt.subplots(
        1, n_model, 
        figsize=(2*n_model, 2)
    )

    for i, ax in enumerate(ax_list):
        w_model_2t = w_3t[i].detach().numpy()
        # w_model_2t: (feat, hidden) = (5, 2)
        color_list = [
            mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']
        ]
        ax.scatter(
            w_model_2t[:, 0],
            # x coordinate
            w_model_2t[:, 1],
            # y coordinate
            c=color_list[0:n_feat]
        )

        ax.set_aspect('equal')

        feat_line_3np = np.stack(
            (   
                np.zeros_like(w_model_2t), 
                w_model_2t
            ),
            axis=1
        )

        # feat_line_3np: (feat, 2, hidden)
        #
        # feat_line_3np[0] 
        # = [[x,y],[0,0]] 
        # = line from (0,0) to (x,y)
        # where 0 is feature number 0, 
        #

        ax.add_collection(
            mc.LineCollection(
               feat_line_3np, 
               colors=color_list
            ) 
        )

        z = 1.5
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z,z))
        ax.set_ylim((-z,z))

        ax.tick_params(
            left=True,
            labelleft=False,

            right=False,
            
            bottom=True,
            labelbottom=False
        )

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        for spine in ['bottom', 'left']:
            ax.spines[spine].set_position('center')

    plt.savefig(
        f'src/superposition_original/viz/01_scatter_plot.png'
    )



if __name__ == "__main__":
    plot_star_diagram()