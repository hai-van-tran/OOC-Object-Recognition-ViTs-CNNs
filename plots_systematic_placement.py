import json
import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import models


def write_all_predictions_in_a_file(output_root, model_type):
    """
    store all predictions in a csv file for a model type
    :param output_root: directory where all prediction files are saved
    :param model_type: cnn, vit or hybrid
    :return:
    """
    # get list of models
    model_list = models.get_model_list(model_type)

    data_root = Path("datasets/OOC_Dataset/04_OOC_compositions") / output_root.name
    path_list = output_root.glob("*")

    # path to csv file to save all predictions
    save_path = Path("outputs/overall") / output_root.name
    save_path.mkdir(exist_ok=True, parents=True)
    csv_path = save_path / (model_type + "_prediction.csv")

    # create file or overwrite it
    csv_df = pd.DataFrame(columns=["dataset_id", "actual_class_index", "prediction_class_index", "row", "column"])
    csv_df.to_csv(csv_path, index=False)

    count = 0
    for path in path_list:
        # failed set
        if path.name == "ILSVRC2012_val_00000293_with_ILSVRC2012_val_00000293":
            continue
        # get prediction files which belong to the model type
        prediction_path = path / "prediction"
        prediction_files = [file for file in prediction_path.glob("*") if file.stem in model_list]

        # get metadata json file
        metadata_path = list((data_root / path.name).glob("*metadata.json"))[0]
        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        # get placements
        placement_dict = {}
        for placement in metadata["placements"]:
            dataset_id = os.path.splitext(placement["filename"])[0]
            placement_dict[dataset_id] = (placement["row"], placement["column"])

        # read each csv file and save in overall file
        for file in prediction_files:
            df = pd.read_csv(file, usecols=["dataset_id", "actual_class_index", "prediction_class_index"])

            # look up row and column where object is placed for each image
            df["row"] = df["dataset_id"].apply(lambda x: placement_dict[x][0])
            df["column"] = df["dataset_id"].apply(lambda x: placement_dict[x][1])

            # add into csv file
            df.to_csv(csv_path, mode="a", index=False, header=False)
            count += len(df)
    # print(count)

def heatmap_accuracy_on_systematic_placements(pred_path, model_type):
    """
    calculate accuracy per placement (row, column) and plot in heatmap
    :param pred_path: path to overall prediction csv file
    :param model_type:
    :return:
    """
    def get_accuracy_matrix(data_df):
        """
        compute accuracy matrix
        :param data_df:
        :return:
        """
        # calculate accuracy matrix
        data_df["correct"] = data_df["actual_class_index"] == data_df["prediction_class_index"]
        accuracy_df = data_df.groupby(["row", "column"])["correct"].mean().reset_index()
        acc_matrix = accuracy_df.pivot(index="row", columns="column", values="correct")

        return acc_matrix

    if model_type in ["cnn", "vit", "hybrid"]:
        # bar color
        if model_type == "cnn":
            cmap = "Greens"
        elif model_type == "vit":
            cmap = "Blues"
        elif model_type == "hybrid":
            cmap = "Reds"
        else:
            cmap = "Greys"

        csv_file = pred_path / (model_type + "_prediction.csv")
        data = pd.read_csv(csv_file)

        # grid size
        total_rows = data["row"].max() + 1
        total_columns = data["column"].max() + 1

        accuracy_matrix = get_accuracy_matrix(data)

        plt.figure(figsize=(total_columns, total_rows))
        sns.heatmap(accuracy_matrix, annot=True, fmt=".4f", cmap=cmap, cbar=True, square=True)
        plt.title(f"{model_type.upper()} accuracy heatmap for object placements ({total_rows}x{total_columns} grid)")
        plt.xlabel("Column")
        plt.ylabel("Row")
        # save plot
        save_path = Path(f"plots/systematic_placements/grid_{total_rows}x{total_columns}")
        save_path.mkdir(exist_ok=True, parents=True)
        figure_name = plt.gca().get_title().replace(" ", "_")
        # plt.savefig(save_path / figure_name)
        plt.show()

    elif model_type == "":
        model_types = ["cnn", "vit", "hybrid"]
        cmaps = ["Blues", "Blues", "Blues"]
        accuracy_matrices = []
        total_rows = 0
        total_columns = 0

        for m in model_types:
            csv_file = pred_path / (m + "_prediction.csv")
            data = pd.read_csv(csv_file)
            accuracy_matrices.append(get_accuracy_matrix(data))
            # grid size
            total_rows = data["row"].max() + 1
            total_columns = data["column"].max() + 1

        fig, ax = plt.subplots(1, 3, figsize=(total_columns * 3, total_rows))
        for i, matrix in enumerate(accuracy_matrices):
            sns.heatmap(matrix, ax=ax[i], annot=True, fmt=".2f", cmap=cmaps[i], cbar=False, square=True)
            ax[i].set_title(model_types[i].upper())
            plt.xlabel("Column")
            plt.ylabel("Row")

        # save plot
        save_path = Path(f"plots/systematic_placements/grid_{total_rows}x{total_columns}")
        save_path.mkdir(exist_ok=True, parents=True)
        figure_name = f"Accuracy heatmaps for object placements ({total_rows}x{total_columns} grid)".replace(" ", "_")

        plt.savefig(save_path / figure_name)
        plt.show()


if __name__=="__main__":
    # write predictions into a csv file for each model type
    # output_path = Path("outputs/systematic_placements_7x7")
    # write_all_predictions_in_a_file(output_path, "cnn")
    # write_all_predictions_in_a_file(output_path, "vit")
    # write_all_predictions_in_a_file(output_path, "hybrid")

    # create heatmap of accuracy
    prediction_path = Path("outputs/overall/systematic_placements_7x7")
    # heatmap_accuracy_on_systematic_placements(prediction_path, "cnn")
    # heatmap_accuracy_on_systematic_placements(prediction_path, "vit")
    # heatmap_accuracy_on_systematic_placements(prediction_path, "hybrid")
    # heatmap_accuracy_on_systematic_placements(prediction_path, "")