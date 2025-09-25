import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns

import models
from pprintpp import pprint

def plot_accuracy_occ():
    """
    plot the accuracies of all models when inference on OOC dataset
    TODO: save figures
    :return:
    """
    output_path = Path('outputs/accuracy_on_ooc')
    accuracy_paths = [output_path / Path(model + '_accuracy.csv') for model in ['vit', 'cnn', 'hybrid']]

    model_names = []
    accuracy_values = []
    model_types = []
    colors = []
    for path in accuracy_paths:
        df = pd.read_csv(path)
        model_count = len(df)
        if path.name.startswith('vit'):
            model_types.extend(['Vision Transformer'] + ['_Vision Transformer'] * (model_count - 1))
            colors.extend(['#83c5be'] * model_count)
        elif path.name.startswith('cnn'):
            model_types.extend(['CNN'] + ['_CNN'] * (model_count - 1))
            colors.extend((['#e5989b'] * model_count))
        else:
            model_types.extend(['Hybrid'] + ['_Hybrid'] * (model_count - 1))
            colors.extend((['#457b9d'] * model_count))
        model_names.extend(df.iloc[:, 0].tolist())
        accuracy_values.extend([a*100 for a in df.iloc[:, 1].tolist()])

    model_names = [name
                   .replace('_patch', '/')
                   .replace('small', 's')
                   .replace('base', 'b')
                   .replace('large', 'l')
                   .replace('_224', '')
                   .replace('_384', '') for name in model_names]

    fig, ax = plt.subplots()
    bar_container = ax.bar(model_names, accuracy_values, label=model_types, color=colors)
    ax.set(title='Top-1-Accuracy on OOC dataset', xlabel='Model', ylabel='Accuracy (%)', ylim=(0, 100))
    ax.bar_label(bar_container, fmt='{:,.2f}')
    ax.set_xticklabels(model_names, rotation=30, ha='right')

    ax.legend()
    plt.show()


def bar_plots_mean_std(model_type=None):
    """
    plot bar charts the average distance of the predictions to the actual classes and the standard deviations
    :return:
    """
    # path to save graphs
    save_path = Path("plots/background/mean_std/bar_chart")
    save_path.mkdir(parents=True, exist_ok=True)

    # get data
    _, mean_std_dict, average_distance, count_per_distance = process_predictions_background_only(model_type)

    # plot for mean_std_dict: mean, std of prediction distance each image
    img_names = list(mean_std_dict.keys())
    mean_values = [round(value[0], 1) for value in mean_std_dict.values()]
    std_values = [round(value[1], 1) for value in mean_std_dict.values()]

    # sort by mean
    sorted_idx = np.argsort(mean_values)
    sorted_img_names = [img_names[i] for i in sorted_idx]
    sorted_mean_values = [mean_values[i] for i in sorted_idx]
    sorted_std_values = [std_values[i] for i in sorted_idx]

    bar_chart_mean_std(
        sorted_mean_values,
        sorted_std_values,
        average_distance,
        model_type,
        save_path
    )

    values_under_average = [value for value in sorted_mean_values if value < average_distance]
    values_above_average = [value for value in sorted_mean_values if value >= average_distance]

    bar_chart_mean_std(
        values_under_average,
        sorted_std_values[:len(values_under_average)],
        average_distance,
        model_type,
        save_path,
        note="(1.1_under overall average distance)"
    )
    bar_chart_mean_std(
        values_above_average,
        sorted_std_values[len(values_under_average):],
        average_distance,
        model_type,
        save_path,
        note="(1.2_above overall average distance)"
    )

    values_under_half_average = [value for value in sorted_mean_values if value < (average_distance / 2)]
    values_around_average = [value for value in sorted_mean_values if (average_distance / 2) <= value < (3 * average_distance / 2)]
    values_above_average_and_a_half = [value for value in sorted_mean_values if value >= (3 * average_distance / 2)]

    bar_chart_mean_std(
        values_under_half_average,
        sorted_std_values[:len(values_under_half_average)],
        average_distance,
        model_type,
        save_path,
        note="(2.1_under half overall average distance)"
    )
    bar_chart_mean_std(
        values_around_average,
        sorted_std_values[len(values_under_half_average):len(values_under_half_average)+len(values_around_average)],
        average_distance,
        model_type,
        save_path,
        note="(2.2_around overall average distance)"
    )
    bar_chart_mean_std(
        values_above_average_and_a_half,
        sorted_std_values[len(values_under_half_average)+len(values_around_average):],
        average_distance,
        model_type,
        save_path,
        note="(2.3_above overall average distance and a half)"
    )

def bar_chart_mean_std(means, std_deviations, overall_mean, model_type, save_path, note=None):
    """

    :param means:
    :param std_deviations:
    :param overall_mean:
    :param model_type:
    :param save_path:
    :param note:
    :return:
    """
    if model_type == "cnn":
        color = "#b0c4b1"
        ecolor = "#3a5a40"
    elif model_type == "vit":
        color = "#bde0fe"
        ecolor = "#457b9d"
    elif model_type == "hybrid":
        color = "#edafb8"
        ecolor = "#780000"
    else:
        color = "#cdb4db",
        ecolor = "#3e1f47"

    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.bar(
        x,
        means,
        yerr=std_deviations,
        align="center",
        color=color,
        ecolor=ecolor
    )
    ax.set_title("Average distance and standard deviation of distances from predicted classes to actual classes on background alone "
                 + (("on " + model_type.upper()) if model_type else "across models")
                 + (("\n" + note) if note else ""))
    ax.set_xlabel("Images")
    ax.set_ylabel("Average distance")

    ax.yaxis.grid(True)
    ax.axhline(y=round(overall_mean, 1), color=ecolor, alpha=0.5, linestyle="--", linewidth=2) # 023047
    ax.text(0.01, overall_mean + 5, "overall average = {}".format(round(overall_mean, 1)), color=ecolor, va="bottom", ha="left", transform=ax.get_yaxis_transform())

    # plt.subplots_adjust(left=0.124, right=0.977, bottom=0.121, top=0.922, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    figure_name = (
        "Average_distance_standard_deviation_"
        + (("on_" + model_type.upper()) if model_type else "across_models")
       + (("_" + note.replace(" ", "_")) if note else "")
       + ".png"
    )
    plt.savefig(save_path / figure_name)
    # plt.show()
    plt.close()

def fill_between_line_chart_mean_std():
    """

    :return:
    """
    save_path = Path("plots/background/mean_std/line_chart")
    save_path.mkdir(parents=True, exist_ok=True)

    model_types = ["cnn", "vit", "hybrid", None]

    results = {}
    for model_type in model_types:
        _, mean_std_dict, average_distance, _ = process_predictions_background_only(model_type)
        # plot for mean_std_dict: mean, std of prediction distance each image
        img_names = list(mean_std_dict.keys())
        mean_values = [round(value[0], 1) for value in mean_std_dict.values()]
        std_values = [round(value[1], 1) for value in mean_std_dict.values()]
        # sort by std
        sorted_idx = np.argsort(mean_values)
        sorted_img_names = [img_names[i] for i in sorted_idx]
        sorted_mean_values = [mean_values[i] for i in sorted_idx]
        sorted_std_values = [std_values[i] for i in sorted_idx]
        results[model_type] = [sorted_img_names, sorted_mean_values, sorted_std_values, average_distance]

        if model_type == "cnn":
            color = "#588157"
        elif model_type == "vit":
            color = "#457b9d"
        elif model_type == "hybrid":
            color = "#780000"
        else:
            color = "#3e1f47"
        x = np.arange(len(sorted_idx))
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.plot(
            x,
            sorted_mean_values,
            color=color
        )
        ax.fill_between(
            x,
            [mean - std for mean, std in zip(sorted_mean_values, sorted_std_values)],
            [mean + std for mean, std in zip(sorted_mean_values, sorted_std_values)],
            color=color,
            alpha=0.5
        )
        ax.set_title(
            "Average distance and standard deviation of distances on background alone "
            + (("on " + model_type.upper()) if model_type else "across models")
        )
        ax.set_xlabel("Images")
        ax.set_ylabel("Average distance")

        ax.yaxis.grid(True)
        ax.axhline(y=round(average_distance, 1), color=color, alpha=0.5, linestyle="--", linewidth=2) # 023047
        ax.text(0.01, average_distance + 5, "overall average = {}".format(round(average_distance, 1)), color=color, va="bottom",
                ha="left", transform=ax.get_yaxis_transform())


        # plt.subplots_adjust(left=0.124, right=0.977, bottom=0.121, top=0.922, wspace=0.2, hspace=0.2)
        plt.tight_layout()
        figure_name = (
            "Average_distance_standard_deviation_"
            + (("on_" + model_type.upper()) if model_type else "across_models")
            + ".png"
        )
        plt.savefig(save_path / figure_name)
        # plt.show()
        # plt.close()

    # plot all together
    fig, ax = plt.subplots(figsize=(18, 10))
    x = np.arange(len(results["cnn"][0]))
    shift_x = 0
    for model_type in results.keys():
        if model_type == "cnn":
            color = "#588157"
        elif model_type == "vit":
            color = "#457b9d"
        elif model_type == "hybrid":
            color = "#780000"
        else:
            color = "#cdb4db"
        ax.plot(
            x,
            results[model_type][1],
            color=color,
            label=model_type if model_type else "overall"
        )
        ax.axhline(y=round(results[model_type][3], 1), color=color, linestyle="--", linewidth=2)
        ax.text(0.01 + shift_x, results[model_type][3] + 50, "overall average = {}".format(round(results[model_type][3], 1)), color=color,
                va="bottom", ha="left", transform=ax.get_yaxis_transform())
        shift_x += 0.1
        # ax.fill_between(
        #     x,
        #     [mean - std for mean, std in zip(results[model_type][1], results[model_type][2])],
        #     [mean + std for mean, std in zip(results[model_type][1], results[model_type][2])],
        #     color=color,
        #     alpha=0.5
        # )
    ax.set_title("Average distance and standard deviation of distances on background alone")
    ax.legend(loc="upper left")
    ax.set_xlabel("Images")
    ax.set_ylabel("Average distance")
    ax.yaxis.grid(True)

    # plt.subplots_adjust(left=0.124, right=0.977, bottom=0.121, top=0.922, wspace=0.2, hspace=0.2)
    plt.tight_layout()
    figure_name = "Average_distance_standard_deviation.png"
    plt.savefig(save_path / figure_name)
    plt.show()
    plt.close()

def histogram_plot_ranking(model_type=None):
    save_path = Path("plots/background/ranking/histogram")
    save_path.mkdir(parents=True, exist_ok=True)

    per_image_distance_dict, _, _, (count_per_distance, distance_list) = process_predictions_background_only(model_type)
    xbins = np.arange(0, 1000, 10)
    if model_type == "cnn":
        color = "#3a5a40"
    elif model_type == "vit":
        color = "#457b9d"
    elif model_type == "hybrid":
        color = "#780000"
    else:
        color = "#3e1f47" # cdb4db
    # histogram for the density of average distances (247 images)
    histogram_chart_ranking(distance_list, xbins, color, save_path, "Density of average distances (total="
                            + str(len(distance_list))
                            + ") on " + (model_type.upper() if model_type else "all model types"))

    # histogram for the density of distances of all predictions from different models
    distance_list_of_all_predictions = []
    for value in per_image_distance_dict.values():
        distance_list_of_all_predictions.extend(value.tolist())
    histogram_chart_ranking(distance_list_of_all_predictions, xbins, color, save_path, "Density of distances (total="
                            + str(len(distance_list_of_all_predictions))
                            + ") on " + (model_type.upper() if model_type else "all model types"))



def histogram_chart_ranking(distance_list, xbins, color, save_path, title):
    fig, ax = plt.subplots(figsize=(18, 10))
    style = {"facecolor": "none", "edgecolor": color, "alpha": 0.6, "linewidth": 3, "stacked": True, "fill": False}
    ax.hist(distance_list, bins=xbins, density=True, histtype="step", **style)
    ax.plot(distance_list, 0*np.array(distance_list))
    ax.set_title(title)
    ax.set_xlabel("Distance (rank)")
    ax.set_ylabel("Density")

    plt.tight_layout()
    figure_name = ax.get_title().replace(" ", "_") + "_(range_of_" + str(xbins[1] - xbins[0]) + ").png"

    plt.savefig(save_path / figure_name)
    # plt.show()
    plt.close()


def histogram_chart_ranking_all_in_one(xbins):
    model_types = {
        "cnn": [],
        "vit": [],
        "hybrid": []
    }
    for model_type in model_types:
        _, _, _, (_, distance_list) = process_predictions_background_only(model_type)
        model_types[model_type] = distance_list
    df = pd.DataFrame(model_types)
    pprint(df)

    sns.histplot(df, binwidth=xbins, binrange=(0,1000), element="step")

    plt.tight_layout()
    # plt.savefig(save_path / figure_name)
    plt.show()
    plt.close()

def process_predictions_background_only(model_type=None):
    """
    process the predictions to calculate:
    - calculate distances from the predicted classes to the actual classes
    - compute the average distance and the standard deviation for each image
    - compute the average distance across all images
    - count the number of images whose prediction distances are at the same rank
    :param model_type: None | str -- None: for all models; str: cnn, vit, hybrid (default: None)
    :return: tuple(dict, dict, float, (dict, list)): the results of the 4 calculation mentioned above
    """
    # path to files where predictions are saved
    prediction_path = Path("outputs/background/prediction")
    all_prediction_files = prediction_path.glob("*.csv")

    # categorize prediction files based on model type
    prediction_files = []

    if model_type:
        model_names = models.get_model_list(model_type)
        for file in all_prediction_files:
            model_name = file.stem
            if model_name in model_names:
                prediction_files.append(file)
    else: # model_type == None
        prediction_files = [file for file in all_prediction_files]

    # go through all files and accumulate the ranks (distances) for each image
    per_image_distance_dict = {}
    for file in prediction_files:
        df = pd.read_csv(file)
        distances = df[["dataset_id", "prediction_distance"]]
        per_image_distance_dict = collect_distances_per_image(per_image_distance_dict, distances)

    # average distance for each image and average distance across all images
    mean_std_dict = {key: [np.mean(value), np.std(value)] for key, value in per_image_distance_dict.items()}
    average_distance = np.mean([value[0] for value in mean_std_dict.values()])

    # count number of predictions each rank
    rounded_average_distances = {key: int(np.round(value[0])) for key, value in mean_std_dict.items()}
    distance_list = list(rounded_average_distances.values())
    count_per_distance = {rank: distance_list.count(rank) for rank in set(distance_list)}

    return per_image_distance_dict, mean_std_dict, average_distance, (count_per_distance, distance_list)

def calculate_average_distance_and_standard_deviation_background_only(model_type):
    """
    TODO: maybe delete this func
    calculate average distance of prediction to tha actual class when inference is run on background alone
    :return:
    """
    prediction_path = Path("outputs/background/prediction")
    prediction_files = prediction_path.glob("*.csv")
    """
    # calculate average distance across models
    # get distances for each model type and for all models
    model_name_dict = {
        "cnn": models.model_list("cnn"),
        "vit": models.model_list("vit"),
        "hybrid": models.model_list("hybrid")
    }
    cnn_distance_list = []
    vit_distance_list = []
    hybrid_distance_list = []
    distance_list = []

    # iterate through each prediction file
    for file in prediction_files:
        df = pd.read_csv(file)
        distances = df["prediction_distance"].to_list()
        model_name = file.stem
        if model_name in model_name_dict["cnn"]:
            cnn_distance_list.extend(distances)
        elif model_name in model_name_dict["vit"]:
            vit_distance_list.extend(distances)
        elif model_name in model_name_dict["hybrid"]:
            hybrid_distance_list.extend(distances)
        else:
            print("No model name matches!")
        distance_list.extend(distances)

    # count number of predictions each rank
    cnn_distance_count = {rank: cnn_distance_list.count(rank) for rank in set(cnn_distance_list)}
    vit_distance_count = {rank: vit_distance_list.count(rank) for rank in set(vit_distance_list)}
    hybrid_distance_count = {rank: hybrid_distance_list.count(rank) for rank in set(hybrid_distance_list)}
    distance_count = {rank: distance_list.count(rank) for rank in set(distance_list)}

    # calculate average distance
    cnn_average_distance = sum(cnn_distance_list) / len(cnn_distance_list)
    vit_average_distance = sum(vit_distance_list) / len(vit_distance_list)
    hybrid_average_distance = sum(hybrid_distance_list) / len(hybrid_distance_list)
    average_distance = sum(distance_list) / len(distance_list)

    print(cnn_average_distance)
    print(vit_average_distance)
    print(hybrid_average_distance)
    print(average_distance)

    # pprint(distance_count)
    # pprint(cnn_distance_count)
    # pprint(vit_distance_count)
    # pprint(hybrid_distance_count)
    """

def collect_distances_per_image(distance_dict, distances_df):
    """
    accumulate all distances of predictions of each image from different models
    :param distance_dict: dict -- {image_name: array of distances}
    :param distances_df: pandas.DataFrame -- dataframe of predictions
    :return: dict -- updated dictionary {image_name: array of distances}
    """
    if not distance_dict:
        # initiate dict with keys
        distance_dict = {
            dataset_id: np.array([], dtype=int) for dataset_id in distances_df["dataset_id"].tolist()
        }
    # average distance of prediction on each image:
    distance_dict.update({
        dataset_id: np.append(distance_dict[dataset_id], distances_df[distances_df["dataset_id"] == dataset_id].iloc[0]["prediction_distance"])
                for dataset_id in distances_df["dataset_id"].tolist()
    })
    return distance_dict

def plot_background_only():
    # fill_between_line_chart_mean_std()
    # bar_plots_mean_std()
    # bar_plots_mean_std("cnn")
    # bar_plots_mean_std("vit")
    # bar_plots_mean_std("hybrid")
    # histogram_plot_ranking()
    # histogram_plot_ranking("cnn")
    # histogram_plot_ranking("vit")
    # histogram_plot_ranking("hybrid")
    histogram_chart_ranking_all_in_one(100)
    pass

def plot_accuracy_imagenet_vs_ooc():
    imagenet_output_path = Path("outputs/accuracy_on_imagenet")
    ooc_output_path = Path("outputs/accuracy_on_ooc")
    model_types = ["cnn", "vit", "hybrid"]

    imagenet_avg_accuracies = []
    ooc_avg_accuracies = []
    results = []
    for model_type in model_types:
        imagenet_accuracy_file = imagenet_output_path / (model_type + "_accuracy.csv")
        ooc_accuracy_file = ooc_output_path / (model_type + "_accuracy.csv")

        imagenet_df = pd.read_csv(imagenet_accuracy_file, names=["model", "top_1_accuracy", "top_5_accuracy"], header=0)
        ooc_df = pd.read_csv(ooc_accuracy_file, names=["model", "top_1_accuracy", "top_5_accuracy"], header=0)
        ooc_df = ooc_df[~ooc_df["model"].str.startswith("deit")]

        results.append({
            "dataset": "ImageNet-1K Val",
            "model_type": model_type,
            "avg_accuracy": imagenet_df["top_1_accuracy"].mean() * 100
        })
        results.append({
            "dataset": "OOC Dataset",
            "model_type": model_type,
            "avg_accuracy": ooc_df["top_1_accuracy"].mean() * 100
        })

    results_df = pd.DataFrame(results)
    plot_df = results_df.pivot(index="model_type", columns="dataset", values="avg_accuracy")
    ax = plot_df.plot(kind="bar", figsize=(8, 6), color=["#4a4e69", "#c9ada7"], rot=0)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9
        )
    plt.title("Average Accuracy on ImageNet-1K Val and OOC Dataset")
    plt.ylim(0, 100)
    plt.xlabel("Model Type")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    # save plot
    save_path = Path("plots/imagenet_vs_ooc")
    save_path.mkdir(exist_ok=True, parents=True)
    figure_name = "Accuracy_on_ImageNet_and_OOC"
    plt.savefig(save_path / figure_name)

    plt.show()


if __name__ == '__main__':
    # plot_background_only()
    plot_accuracy_imagenet_vs_ooc()
