import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_lineage_file(video_name, descriptor, final_frame):
    """
    Devuelve un dict {leaf_mask: ndarray[:final_frame]} con la señal cruda
    (sin normalizar) de cada linaje almacenado en
        data/results/<video_name>/lineages_<descriptor>_intensities_<video_name>.txt
    """
    path = os.path.join(
        "data", "results", video_name,
        f"lineages_{descriptor}_intensities_{video_name}.txt"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    data = {}
    with open(path) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            leaf_str, vec_str = ln.split(":", 1)
            leaf = int(leaf_str.strip())
            vec  = np.array([float(v) if v.lower() != "nan" else np.nan
                             for v in vec_str.split()])
            data[leaf] = vec[:final_frame]     # recorta/pad según final_frame
    return data


# ----------------------------------------------------------------------
# Función principal
# ----------------------------------------------------------------------
def plot_all_normalized_lineages_expColors(lineages_dict,
                                 descriptor,
                                 control_array,
                                 final_frame,
                                 y_max=5,
                                 pretreat_len=10,
                                 color_map=None):
    """
    • lineages_dict : {"exp0": [2, 3, 5], "exp4": [3, 4]}
    • control_array : array/Serie longitud = final_frame (ya suavizado)
    • color_map     : dict opcional {"exp0": "tab:blue", ...}
                      si None → se genera con tab10.
    """

    frames   = np.arange(final_frame)
    control  = pd.Series(control_array, index=frames)

    # -------- asignar un color por experimento ------------------------ #
    exps = list(lineages_dict.keys())
    if color_map is None:
        base_colors = plt.cm.tab10.colors
        color_map = {exp: base_colors[i % len(base_colors)]
                     for i, exp in enumerate(exps)}

    # -------- crear la figura ---------------------------------------- #
    plt.figure(figsize=(10, 6))
    legend_handles = []

    for exp in exps:
        color = color_map[exp]
        raw_data = read_lineage_file(exp, descriptor, final_frame)

        for leaf in lineages_dict[exp]:
            vec = raw_data[leaf]

            # --- normalización a pre-tratamiento ---------------------- #
            baseline = np.nanmean(vec[:pretreat_len])
            if np.isnan(baseline) or baseline == 0:
                print(f"⚠️  {exp}-{leaf} omitido (baseline NaN/0)")
                continue
            vec_norm = (vec / baseline) / control

            label = f"{exp}-{leaf}"
            plt.plot(frames, vec_norm, lw=1, color=color, label=label)

        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=exp))

    # -------- embellecer -------------------------------------------- #
    plt.xlabel("t")
    plt.ylabel("Intensidad normalizada")
    plt.xlim(0, final_frame - 1)
    plt.ylim(0.5, y_max)
    # leyenda: una sola entrada por linaje (mismo color para un experimento)
    plt.legend(handles=legend_handles, fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_all_normalized_lineages_idColors(lineages_dict,
                                 descriptor,
                                 control_array,
                                 final_frame,
                                 df_clusters,
                                 y_max=5,
                                 pretreat_len=10,
                                 cluster_colors=None):
    """
    lineages_dict : {"exp0": [2, 5], "exp4": [3, 4], …}
                    (la clave es el nombre del vídeo / experimento,
                     los valores son los label de los linajes a trazar)
    df_clusters   : DataFrame con columnas  [exp, label, cluster]
                    cluster: 0 → amniótica, 1 → mesodérmica, 2 → pluripotente
    control_array : array o Serie (longitud = final_frame) ya suavizada
    cluster_colors: opcional {0:"#...", 1:"#...", 2:"#..."}
    """

    # ------- paleta por cluster -------------------------------------- #
    if cluster_colors is None:
        cluster_colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

    cluster_names = {0: "pluripotente", 1: "amniotica", 2: "mesodermica"}

    # ------- control & eje X ----------------------------------------- #
    frames  = np.arange(final_frame)
    control = pd.Series(control_array, index=frames)

    # ------- figura --------------------------------------------------- #
    plt.figure(figsize=(10, 6))
    legend_handles = {}

    for exp, leaves in lineages_dict.items():
        raw_data = read_lineage_file(exp, descriptor, final_frame)  # <- tu helper

        for leaf in leaves:
            # --- averiguar cluster ----------------------------------- #
            row = df_clusters[(df_clusters["exp"] == int(exp.strip("exp")))
                              & (df_clusters["label"] == leaf)]
            if row.empty:
                print(f"⚠️  Sin cluster: {exp}-{leaf}.  Saltado.")
                continue
            cluster = int(row.iloc[0]["cluster"])
            color   = cluster_colors.get(cluster, "grey")

            vec = raw_data[leaf]

            # --- normalización --------------------------------------- #
            baseline = np.nanmean(vec[:pretreat_len])
            if np.isnan(baseline) or baseline == 0:
                print(f"⚠️  {exp}-{leaf} omitido (baseline NaN/0)")
                continue
            vec_norm = (vec / baseline) / control

            # --- plot ------------------------------------------------ #
            plt.plot(frames, vec_norm, lw=1, color=color)

            # guardar “proxy” para la leyenda (una por cluster)
            if cluster not in legend_handles:
                legend_handles[cluster] = Line2D([0], [0], color=color,
                                                 lw=2, label=cluster_names[cluster])

    # ------- estética ------------------------------------------------- #
    plt.xlabel("t")
    plt.ylabel("Intensidad normalizada")
    plt.xlim(1, final_frame)
    plt.ylim(0.5, y_max)
    plt.legend(handles=list(legend_handles.values()), fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_cluster_by_condition(cluster_id,
                              cond_list,
                              descriptor,
                              control_array,
                              final_frame,
                              df_clusters,
                              pretreat_len=10,
                              cluster_names=None,
                              condition_names=None,
                              condition_colors=None,
                              outdir=None):
    """
    cluster_id      : 0 (=amniótica) · 1 (=mesodérmica) · 2 (=pluripotente)
    cond_list       : [cond1, cond2, cond3, cond4]  (cada cond es dict exp→[labels])
    control_array   : Serie/array suavizada (long = final_frame)
    df_clusters     : DataFrame [exp, label, cluster]
    """

    # ---- nombres & colores por defecto ------------------------------ #
    if cluster_names is None:
        cluster_names = {0: "pluripotente", 1: "amniotica", 2: "mesodermica"}

    if condition_names is None:
        condition_names = ["mTeSR 0–48 h",
                           "BMP 10 ng/ml 0–16 h; mTeSR 16–48 h",
                           "BMP 10 ng/ml 0–30 h; Noggin 30–48 h",
                           "BMP 10 ng/ml 0–48 h"]

    if condition_colors is None:
        condition_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    if outdir is None:
        outdir = os.path.join("data", "plots",
                              "clusters_by_condition", descriptor)
    os.makedirs(outdir, exist_ok=True)

    # ---- vector control y eje X ------------------------------------- #
    frames  = np.arange(final_frame)
    control = pd.Series(control_array, index=frames)

    # ---- figura ------------------------------------------------------ #
    plt.figure(figsize=(10, 6))
    legend_handles = []

    # recorrer las 4 condiciones
    for cond_idx, cond in enumerate(cond_list):
        color      = condition_colors[cond_idx]
        cond_label = condition_names[cond_idx]

        for exp, leaves in cond.items():
            raw = read_lineage_file(exp, descriptor, final_frame)

            for leaf in leaves:
                # ¿Está este linaje en el cluster deseado?
                row = df_clusters[
                    (df_clusters["exp"] == int(exp.strip("exp")))
                    & (df_clusters["label"] == leaf)
                    & (df_clusters["cluster"] == cluster_id)
                ]
                if row.empty:
                    continue

                vec       = raw[leaf]
                baseline  = np.nanmean(vec[:pretreat_len])
                if np.isnan(baseline) or baseline == 0:
                    continue
                vec_norm  = (vec / baseline) / control

                plt.plot(frames, vec_norm, lw=1, color=color)

        # proxy para la leyenda (una entrada por condición)
        legend_handles.append(Line2D([0], [0], color=color, lw=2,
                                     label=cond_label))

    # ---- estética & guardado ---------------------------------------- #
    plt.xlabel("t")
    plt.ylabel("Intensidad normalizada")
    plt.xlim(1, final_frame)
    plt.ylim(0.5, 5)
    plt.legend(handles=legend_handles, title="Condición", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    cond1 = {
    "exp0": [5, 6, 7, 8], #[5, 6, 7, 8, 9, 10]
    "exp2": [15, 16, 17, 20],
    "exp4": [3, 4, 5, 6]
    }

    cond2 = {
    "exp13": [1, 2, 3, 4, 5],
    "exp14": [18, 26, 17, 19, 32] #[16, 18, 22, 23, 26, 27, 28, 14, 17, 19, 20, 32]
    }

    cond3 = {
    "exp19": [9, 16, 22, 26, 29],
    "exp43": [8, 14, 15, 18, 19],
    "exp46": [4, 6, 7, 9, 13]
    } #"exp41": [8, 11, 12, 14, 17, 18, 19], "exp44": [1, 2, 3, 4]

    cond4 = {
    "exp58": [1, 2, 5, 7],
    "exp59": [5, 6, 12, 14, 16, 18],
    "exp63": [2, 3, 4, 5, 7, 8]
    }

    control_array_path = "C:/Users/pauma/Desktop/TFG/repo/analysis/control_smoothed.pkl"

    control_array = pd.read_pickle(control_array_path)

    df_clusters = pd.read_csv("C:/Users/pauma/Desktop/TFG/repo/analysis/cell_intensities_3d.csv")

    df_clusters = df_clusters[["exp", "label", "cluster"]]

    cond_list = [cond1, cond2, cond3, cond4]

    '''
    plot_cluster_by_condition(
        cluster_id    = 2,
        cond_list     = cond_list,
        descriptor    = "mean",
        control_array = control_array,
        final_frame   = 153,
        df_clusters   = df_clusters
    )
    
    '''
    plot_all_normalized_lineages_idColors(
        lineages_dict=cond4,
        descriptor="mean",
        control_array=control_array,
        final_frame=153,
        df_clusters=df_clusters,
        y_max=4
    )

    plot_all_normalized_lineages_expColors(
        lineages_dict=cond4,
        descriptor="mean",
        control_array=control_array,
        final_frame=153,
        y_max=4
    )
    