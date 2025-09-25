import argparse
import csv
import torch
import json
from pathlib import Path
import helper
from models import get_model_list, create_model
from dataset import load_dataset
from sklearn.metrics import top_k_accuracy_score
import numpy as np

def inference(data_path, output_path):
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataRoot', help='root directory to dataset', default='datasets')
    parser.add_argument('-o', '--outputRoot', help='root directory to saved outputs', default='outputs')
    parser.add_argument('-m', '--modelType', help='model type: cnn, vit, hybrid', default='cnn')
    parser.add_argument('-b', '--batchSize', help='batch size', default=100)
    parser.add_argument('-n', '--numWorkers', help='number of workers', default=96)
    parser.add_argument('-t', '--task', help='inference on which task: imagenet, background, ranked, placement',
                        default='imagenet')

    args = parser.parse_args()

    # get arguments
    data_root = Path(args.dataRoot)
    output_root = Path(args.outputRoot)
    model_type = args.modelType
    batch_size = int(args.batchSize)
    num_workers = int(args.numWorkers)
    task = args.task.lower()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """
    arg --task: Inference task on: 
        imagenet: ImageNet-1K Validation Set 
        ooc: OOC dataset
        background: backgrounds alone
        ranked: similarity ranked composition (which ranks?)
        placement: systematic placements dataset (which position?)
    In dictionary `tasks`, the list of information of corresponding dataset is stored:
        {task: [dataset_path, output_path]}
    """

    tasks = {
        "imagenet": ["ImageNet2012", "imagenet"],
        "ooc": ["ooc", "ooc"],
        "background": ["OOC_Dataset/02_backgrounds", "backgrounds"],
        "ranked": ["OOC_Dataset/04_OOC_compositions/similarity_ranked_compositions", "similarity_ranked"],
        "placement": ["OOC_Dataset/04_OOC_compositions/systematic_placements_6x6", "systematic_placements_6x6"]
    }

    # get dataset path, output path
    data_path = data_path #data_root / tasks[task][0]
    output_path = output_path #output_root / tasks[task][1]
    accuracy_folder = output_path / "accuracy"
    prediction_folder = output_path / "prediction"
    if task == "imagenet":
        accuracy_folder = output_path / "accuracy_on_imagenet"

    # create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    accuracy_folder.mkdir(parents=True, exist_ok=True)
    if task != "imagenet":
        prediction_folder.mkdir(parents=True, exist_ok=True)

    # load the list of models
    model_list = get_model_list(model_type)
    transform = None
    dataloader = None

    if model_list is not None:
        # load each models
        count = 0
        for model_name in model_list:
            count += 1

            # accuracy and prediction files
            accuracy_file_name = model_type + "_accuracy" if model_type.lower() != '' else 'accuracy'
            accuracy_json_file_path = accuracy_folder / (accuracy_file_name + ".json")
            accuracy_csv_file_path = accuracy_folder / (accuracy_file_name + ".csv")

            # prediction file and metadata file
            if task != "imagenet":
                prediction_file_path = prediction_folder / (model_name + ".csv")
            if task not in ["imagenet", "ooc"]:
                metadata_file_path = next(data_path.glob("*metadata.csv"))

            # create accuracy files or check if the model has already completed inference
            status = helper.create_accuracy_files_and_check_model_status(accuracy_json_file_path,
                                                                         accuracy_csv_file_path, model_name)
            # status == True: inference of this model is done
            if status:
                continue

            model_dict = create_model(model_name)
            model = model_dict['model']
            model.eval()
            model.to(device)

            # load dataset for the 1st time
            if transform is None:
                transform = model_dict['transform']
                dataloader = load_dataset(data_path, batch_size, num_workers, transform, task)
            # if the transform is not the same as the previous model, change transform and load dataset again
            elif str(transform.transforms) != str(model_dict['transform'].transforms):
                transform = model_dict['transform']
                dataloader = load_dataset(data_path, batch_size, num_workers, transform, task)

            all_labels = []
            all_scores = []
            for batch in dataloader:
                images = batch[0].to(device)
                labels = batch[1]

                with torch.no_grad():
                    outputs = model(images).cpu()  # [100, 1000]

                    # not for task==imagenet: add predictions into file after each batch
                    if task == "ooc":
                        image_names = batch[2]
                        helper.save_prediction_ooc_dataset(output_root, data_path, model_name, image_names, outputs)
                    elif task != "imagenet":
                        image_names = batch[2]
                        helper.save_prediction(prediction_file_path, metadata_file_path, task, image_names, labels.tolist(), outputs)

                    # save results into lists to calculate accuracy later
                    all_labels.extend(labels.tolist())
                    all_scores.extend(outputs.tolist())

            # calculate accuracy
            if all_labels: # if not empty dataset
                all_classes = np.arange(1000)  # [0 - 999]
                top_1_accuracy = top_k_accuracy_score(all_labels, all_scores, k=1, labels=all_classes)
                top_5_accuracy = top_k_accuracy_score(all_labels, all_scores, k=5, labels=all_classes)

                if count == len(model_list):
                    done = True
                else:
                    done = False
                # after each model, save the accuracy into json and csv files, if all models in list are done, also calculate the average accuracy
                helper.save_accuracy(accuracy_json_file_path, accuracy_csv_file_path, model_name, top_1_accuracy,
                                     top_5_accuracy, done)
        

if __name__ == '__main__':
    # inference on ImageNet-1K Val Set
    inference()

    # inference on OOC set
    data_path = Path("datasets/ooc")
    output_path = Path("outputs/ooc")
    inference(data_path, output_path)

    # inference on background alone
    data_path = Path("datasets/OOC_Dataset/02_backgrounds")
    output_path = Path("outputs/background")
    inference(data_path, output_path)

    # inference on similarity ranked compositions

    data_root = Path("datasets/OOC_Dataset/04_OOC_compositions/similarity_ranked_compositions")
    for folder in data_root.glob("*"):
        print(f"===== Inference on {folder.name} =====")
        data_path = folder
        output_path = Path("outputs/similarity_ranked") / folder.name
        inference(data_path, output_path)

    # inference on systematic placement compositions
    data_root = Path("datasets/OOC_Dataset/04_OOC_compositions/systematic_placements_7x7")
    for folder in data_root.glob("*"):
        data_path = folder
        output_path = Path("outputs/systematic_placements_7x7") / folder.name
        inference(data_path, output_path)

    print("\nTHE END!")
