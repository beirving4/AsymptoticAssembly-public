
import numpy as np
import matplotlib.pyplot as plt


def display_mf_model_parameter_evolution(
        param_evo: dict[str, np.ndarray],
        is_model_fitted: dict[float, bool],
        cutoff_time: float = 0.0, # I used 0.4 in the original code
        plot_loglog: bool = False
    ) -> None: 
    

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 12), sharex=True, gridspec_kw={"hspace": 0.0}
    )

    for idx, (key, value) in enumerate(param_evo.items()):
        ax = axes[idx // 2, idx % 2]

        for scale_factor, parameter in value:
            if scale_factor < cutoff_time: continue
            if is_model_fitted[scale_factor]:
                if plot_loglog:
                    ax.loglog(
                        scale_factor, parameter, marker="o", color="tab:blue"
                    )
                else:
                    ax.semilogx(
                        scale_factor, parameter, marker="o", color="tab:blue"
                    )
    


        if idx in [2, 3]:
            ax.set_xlabel(r"${\rm Scale}$ ${\rm Factor}$, $a$", fontsize=20)
            ax.tick_params(axis='x', labelsize=20)


        ax.set_ylabel(key, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)


    plt.tight_layout()
    plt.show()