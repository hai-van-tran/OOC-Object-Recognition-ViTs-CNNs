from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import models

def write_all_predictions_in_a_file(model_type):
    """
    store all predictions in a csv file for a model type
    :param model_type:
    :return: DataFrame -- accuracies by rank ranges
    """
    # get list of models
    model_list = models.get_model_list(model_type)

    output_root = Path("outputs/similarity_ranked")
    data_root = Path("datasets/OOC_Dataset/04_OOC_compositions/similarity_ranked_compositions")
    path_list = output_root.glob("*")

    # path to csv file to save all predictions
    save_path = Path("outputs/overall/similarity_ranked")
    save_path.mkdir(exist_ok=True, parents=True)
    csv_path = save_path / (model_type + "_prediction.csv")

    # create file or overwrite it
    csv_df = pd.DataFrame(columns=["dataset_id", "actual_class_index", "prediction_class_index", "similarity_rank"])
    csv_df.to_csv(csv_path, index=False)

    count = 0
    for i, path in enumerate(path_list):
        print(f"{i+1}. read {model_type} prediction file of {path.name}..")
        prediction_path = path / "prediction"
        metadata_path = list((data_root / path.name).glob("*metadata.csv"))[0]
        metadata_df = pd.read_csv(metadata_path, usecols=["dataset_id", "similarity_rank"])

        # get prediction files which belong to the model type
        prediction_files = [file for file in prediction_path.glob("*") if file.stem in model_list]

        # read each csv file and save in overall file
        for file in prediction_files:
            df = pd.read_csv(file, usecols=["dataset_id", "actual_class_index", "prediction_class_index"])
            merged_df = df.merge(metadata_df, on="dataset_id", how="left")
            merged_df.to_csv(csv_path, mode="a", index=False, header=False)
            count += len(merged_df)
    # print(count)

def similarity_ranked_accuracy_bar_chart(range_size=5, model_type="cnn"):
    """
    create bar chart to demonstrate accuracy on similartity ranked composition for a model type
    :param range_size: int -- range of similarity ranks,
    :param model_type:
    :return:
    """
    def get_rank_range(rank, range_size):
        """
        define what rank range the data point belongs to based on its rank
        :param rank: similarity rank
        :param range_size: size of a rank bucket
        :return: rank range
        """
        start = ((rank - 1) // range_size) * range_size +1
        end = min(start + range_size - 1, 1000)
        return  f"{start}-{end}"

    def accuracy(df):
        """
        compute accuracy on given predictions
        :param df:
        :return: accuracy
        """
        return (df["actual_class_index"] == df["prediction_class_index"]).mean()

    # bar color
    if model_type == "cnn":
        color = "#b0c4b1"
    elif model_type == "vit":
        color = "#bde0fe"
    elif model_type == "hybrid":
        color = "#edafb8"
    else:
        color = "#cdb4db"

    # read csv file
    csv_file = Path("outputs/overall/similarity_ranked") / (model_type + "_prediction.csv")
    data = pd.read_csv(csv_file)

    # define rank range which each prediction belongs to
    data["rank_range"] = data["similarity_rank"].apply(lambda rank: get_rank_range(int(rank), range_size))

    # group predictions by rank range
    grouped_data = data.groupby("rank_range")[["actual_class_index", "prediction_class_index"]].apply(accuracy).reset_index(name="accuracy")
    # sort by rank ranges:
    grouped_data["start"] = grouped_data["rank_range"].str.split("-").str[0].astype(int)
    grouped_data = grouped_data.sort_values("start").reset_index(drop=True)

    # plot single bar chart
    '''
    plt.figure(figsize=(18, 10))
    plt.bar(grouped_data["rank_range"], grouped_data["accuracy"], color=color)
    plt.title(f"{model_type.upper()} accuracy on similarity ranked composition by rank ranges (size={range_size})")
    if range_size < 10:
        x_labels = grouped_data["rank_range"].tolist()
        x = np.arange(len(x_labels))
        plt.xticks(x[::5], x_labels[::5], rotation=90)
    else:
        plt.xticks(rotation=90)
    plt.xlabel(f"Rank range (size={range_size})")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    # save plot
    save_path = Path(f"plots/similarity_ranked/range_size_{range_size}")
    save_path.mkdir(exist_ok=True, parents=True)
    figure_name = plt.gca().get_title().replace(" ", "_")
    
    # plt.savefig(save_path / figure_name)
    plt.show()
    '''
    return grouped_data
def similarity_ranked_accuracy_line_chart_all_in_one(bin_size=50, model_types=["cnn", "vit", "hybrid"]):
    """
    plot accuracy of 3 model types on one plot
    :param bin_size:
    :param model_types:
    :return:
    """
    data_list = []
    colors = []
    x_labels = []
    for model_type in model_types:
        # line color
        if model_type == "cnn":
            color = "#b0c4b1"
        elif model_type == "vit":
            color = "#bde0fe"
        elif model_type == "hybrid":
            color = "#edafb8"
        else:
            color = "#cdb4db"
        df = similarity_ranked_accuracy_bar_chart(bin_size, model_type)
        data_list.append(df)
        colors.append(color)
        x_labels = df["start"].tolist() if len(df) > len(x_labels) else x_labels

    # plot
    plt.figure(figsize=(9, 6))
    x = np.arange(bin_size, 1001, bin_size)
    total_width = 0.8
    width = total_width / len(model_types)
    for i, data in enumerate(data_list):
        # plt.bar(x + i*width , data["accuracy"], width=width, color=colors[i], label=model_types[i])
        plt.plot(x, data["accuracy"], color=colors[i], linewidth=2, label=model_types[i], marker='o')
    plt.title(f"Accuracy on similarity ranked composition for different model types by rank ranges (size={bin_size})")
    # if bin_size < 10:
    #     plt.xticks((x + total_width / 2 - width / 2)[::5], x_labels[::5], rotation=90)
    # else:
    #     plt.xticks(x + total_width/2 - width/2, x_labels,  rotation=90)

    x_pos = np.arange(0, 1001, 100)
    plt.xticks(x_pos) #, labels=[x_labels[i] for i in x_pos])

    plt.xlabel(f"Rank range (size={bin_size})")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # save plot
    save_path = Path(f"plots/similarity_ranked/range_size_{bin_size}")
    save_path.mkdir(exist_ok=True, parents=True)

    figure_name = plt.gca().get_title().replace(" ", "_")
    new_name = figure_name
    i = 0
    while (save_path / (new_name + ".png")).exists():
        i += 1
        new_name = figure_name + f" ({i})"

    plt.savefig(save_path / new_name)

    plt.show()

def get_similarity_rank_of_image(dataset_id):
    """
    get the similarity rank given a dataset_id
    :param dataset_id: str -- name of image without extension
    :return: int -- similarity rank from 1 to 1000
    """
    background_name = dataset_id[:dataset_id.index("_with")]
    data_path = next(Path("datasets/OOC_Dataset/04_OOC_compositions/similarity_ranked_compositions/" + background_name).glob("*metadata.csv"), None)
    df = pd.read_csv(data_path)
    similarity_rank = df[df["dataset_id"] == dataset_id].iloc[0]["similarity_rank"]

    return int(similarity_rank)

if __name__=="__main__":
    bin_size = 20
    similarity_ranked_accuracy_line_chart_all_in_one(bin_size)
