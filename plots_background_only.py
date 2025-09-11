from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from matplotlib.colors import to_rgba

import models


def plot_accuracy_on_background_only():
    """
    plot accuracy on background alone
    :return:
    """
    model_types = ["cnn", "vit", "hybrid"]

    average_top1_accuracy_list = []
    average_top5_accuracy_list = []

    for model_type in model_types:
        # bar color
        if model_type == "cnn":
            color_top_1 = "#4a5759"
            color_top_5 = "#b0c4b1"
        elif model_type == "vit":
            color_top_1 = "#415a77"
            color_top_5 = "#778da9"
        elif model_type == "hybrid":
            color_top_1 = "#b5838d"
            color_top_5 = "#e5989b"
        else:
            color_top_1 = "#415a77"
            color_top_5 = "#778da9"
        accuracy_path = Path("outputs/background/accuracy") / (model_type + "_accuracy.csv")
        df = pd.read_csv(accuracy_path)
        df.rename(columns={df.columns[0]: "model_name"}, inplace=True)
        df = df.drop(df[df["model_name"] == "average"].index)

        # compute average accuracy across models
        average_top1 = df["top_1_accuracy"].mean()
        average_top5 = df["top_5_accuracy"].mean()
        average_top1_accuracy_list.append(average_top1)
        average_top5_accuracy_list.append(average_top5)

        plt.figure(figsize=(9, 5))
        x = np.arange(len(df))
        width = 0.4

        plt.bar(x, df["top_1_accuracy"], width=width, color=color_top_1,  label="Top-1 Accuracy")
        plt.bar(x + width, df["top_5_accuracy"], width=width, color=color_top_5, label="Top-5 Accuracy")
        plt.xticks(x + width/2, df["model_name"], rotation=45, ha="right")
        plt.ylabel("Accuracy")
        plt.grid(axis="y", alpha=0.3)

        plt.legend()
        plt.title(f"{model_type.upper()} accuracy on background alone")
        plt.tight_layout()

        # save plot
        save_path = Path("plots/background/accuracy")
        save_path.mkdir(exist_ok=True, parents=True)
        figure_name = plt.gca().get_title().replace(" ", "_")

        plt.savefig(save_path / figure_name)

    # plot average accuracy
    plt.figure(figsize=(6, 6))
    x = np.arange(len(model_types))
    width = 0.2

    plt.bar(x, average_top1_accuracy_list, width=width, color="#735d78", label="Top-1 Accuracy")
    plt.bar(x + width, average_top5_accuracy_list, width=width, color="#b392ac", label="Top-5 Accuracy")
    plt.xticks(x + width / 2, model_types, ha="right")
    plt.ylabel("Average Accuracy")
    plt.grid(axis="y", alpha=0.3)

    plt.legend()
    plt.title("Average accuracy on background alone")
    plt.tight_layout()

    # save plot
    save_path = Path("plots/background/accuracy")
    save_path.mkdir(exist_ok=True, parents=True)
    figure_name = plt.gca().get_title().replace(" ", "_")

    plt.savefig(save_path / figure_name)
    plt.show()

def plot_histogram_boxplot(mini_bin_size=50, bin_size=10):
    """
    plot similarity distance distribution
    :param mini_bin_size:
    :param bin_size
    :return:
    """
    output_path = Path("outputs/background/prediction")

    color_map = ["Blues", "Reds", "Greens"]
    model_types = ["cnn", "vit", "hybrid"]
    model_lists = []
    model_count = 0
    for model_type in model_types:
        model_list = models.get_model_list(model_type)
        model_lists.append(model_list)
        model_count += len(model_list)
    num_columns = 9
    num_rows = int(np.ceil(model_count / num_columns)) + 3
    fig = plt.figure(figsize=(num_columns, num_rows))
    plt.suptitle(f"Prediction Semantic Distance Distribution per Model (bin size={mini_bin_size})", fontsize=11)

    # go through each model list of cnn, vit and hybrid
    pos = 0
    df_lists = []
    for i, model_list in enumerate(model_lists):
        # read prediction files
        df_list = []
        for model in model_list:
            csv_file = output_path / (model + ".csv")
            df = pd.read_csv(csv_file, usecols=["dataset_id", "prediction_distance"])
            df_list.append(df)
        df_lists.append(df_list)

        num_models = len(model_list)
        cmap = plt.get_cmap(color_map[i])
        mini_bins = np.arange(0, 1000 + mini_bin_size, mini_bin_size)

        # histogram for each model
        for j, model in enumerate(model_list):
            df = df_list[j]
            ranks = df["prediction_distance"]
            color = cmap((j + 1) / num_models + 0.4)

            pos += 1
            plt.subplot(num_rows, num_columns, pos)
            plt.hist(ranks, bins=mini_bins, color=color)
            plt.title(model, fontsize=9)
            plt.tick_params(labelsize=8)
            plt.grid(True, alpha=0.3)

    # average similarity distance each sample across models within model type
    colors = ["skyblue", "lightcoral", "lightgreen"]
    edge_colors = ["blue", "coral", "green"]
    bins = np.arange(0, 1000 + bin_size, bin_size)
    gs = gridspec.GridSpec(num_rows, num_columns, figure=fig)
    fig.add_subplot(gs[num_rows-3:num_rows-1, :])
    average_distance_df_list = []
    for i, df_list in enumerate(df_lists):
        combined_df = pd.concat(df_list)
        average_distance_df = combined_df.groupby("dataset_id")["prediction_distance"].mean()
        average_distance_df_list.append(average_distance_df)

        plt.hist(average_distance_df, bins=bins, label=model_types[i], color=[to_rgba(colors[i], alpha=0.2)], edgecolor=edge_colors[i])
        plt.grid(True, alpha=0.3)
        # plt.xlabel("Average Semantic Distance")
        plt.ylabel("Frequency")
    plt.title(f"Average Semantic Distance Across Model Type (bin size={bin_size})", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # box plot
    fig.add_subplot(gs[num_rows-1:, :])
    bp = plt.boxplot(average_distance_df_list, tick_labels=model_types, patch_artist=True, vert=False, showmeans=True)
    for patch, color, ecolor in zip(bp["boxes"], colors, edge_colors):
        patch.set_facecolor(to_rgba(color, alpha=0.3))
        patch.set_edgecolor(ecolor)
    plt.xlabel("Average Semantic Distance")
    plt.grid(True, alpha=0.3)

    plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
    plt.subplots_adjust(wspace=0.25, hspace=0.5)

    # save plot
    save_path = Path("plots/background/histogram_boxplot")
    save_path.mkdir(exist_ok=True, parents=True)
    figure_name = f"Semantic_Distance_and_Average_Distance_(bin_{mini_bin_size}_vs_{bin_size})"

    # plt.savefig(save_path / figure_name)
    plt.show()





if __name__=="__main__":
    # plot accuracy
    plot_accuracy_on_background_only()

    # plot similarity distance of predictions to actual classes
    plot_histogram_boxplot(50, 10)