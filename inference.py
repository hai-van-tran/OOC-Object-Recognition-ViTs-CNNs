import argparse
import csv
import torch
import json
from pathlib import Path
from create_files import save_prediction_ooc_dataset
from models import model_list, create_model
from dataset import load_dataset
from sklearn.metrics import top_k_accuracy_score
import numpy as np


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataRoot', help='root directory to dataset', default='datasets')
parser.add_argument('-o', '--outputRoot', help='root directory to saved outputs', default='outputs')
parser.add_argument('-m', '--modelType', help='model type: cnn, vit, hybrid', default='vit')
parser.add_argument('-b', '--batchSize', help='batch size', default=100)
parser.add_argument('-n', '--numWorkers', help='number of workers', default=24)
parser.add_argument('-c', '--oocDataset', help='inference on OOC dataset', default=True)
args = parser.parse_args()

# get arguments
data_root = Path(args.dataRoot)
output_root = Path(args.outputRoot)
model_type = args.modelType
batch_size = int(args.batchSize)
num_workers = int(args.numWorkers)
ooc_dataset = args.oocDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get dataset path, output path
if ooc_dataset:
    data_path = data_root / 'ooc'
    output_path = output_root / 'accuracy_on_ooc'
else:
    data_path = data_root / 'ImageNet2012'
    output_path = output_root / 'accuracy_on_imagenet'

# create output directory
output_path.mkdir(parents=True, exist_ok=True)

# load the list of models
model_list = model_list(model_type)
transform = None
dataloader = None


if model_list is not None:
    # load each models
    for model_name in model_list:
        # output files
        json_file_name = model_type + "_accuracy.json" if model_type.lower() != '' else 'accuracy.json'
        json_file_path = output_path / json_file_name
        csv_file_path = output_path / json_file_name.replace('json', 'csv')

        # create output files if they not exist
        if not json_file_path.exists():
            with open(json_file_path, 'w', encoding='utf-8') as file:
                json.dump({}, file, indent=4)
        if not csv_file_path.exists():
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(['', 'top_1_accuracy', 'top_5_accuracy'])
        # check if the model has already completed inference
        json_check = False
        csv_check = False
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if model_name in data:
                json_check = True
        with open(json_file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                if model_name in row:
                    csv_check = True
        if json_check and csv_check:
            continue

        model_dict = create_model(model_name)
        model = model_dict['model']
        model.eval()
        model.to(device)
        # load dataset for the 1st time
        if transform is None:
            transform = model_dict['transform']
            dataloader = load_dataset(data_path, batch_size, num_workers, transform, ooc_dataset)
        # if the transform is not the same as the previous model
        # change transform and load dataset again
        elif str(transform.transforms) != str(model_dict['transform'].transforms):
            transform = model_dict['transform']
            dataloader = load_dataset(data_path, batch_size, num_workers, transform, ooc_dataset)

        all_labels = []
        all_scores = []

        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1]
            with torch.no_grad():
                outputs = model(images).cpu()  # [100, 1000]

                # only for ooc dataset: add predictions into file after each batch
                if ooc_dataset:
                    image_names = batch[2]
                    save_prediction_ooc_dataset(output_root, data_path, model_name, image_names, outputs)

                # save results into lists to calculate accuracy later
                all_labels.extend(labels.tolist())
                all_scores.extend(outputs.tolist())

        # calculate accuracy
        all_classes = np.arange(1000) # [0 - 999]
        top_1_accuracy = top_k_accuracy_score(all_labels, all_scores, k=1, labels=all_classes)
        top_5_accuracy = top_k_accuracy_score(all_labels, all_scores, k=5, labels=all_classes)

        # after each model, save result into a json file
        with open(json_file_path, 'r+', encoding='utf-8') as file:
            data = json.load(file)
            if model_name not in data:
                data[model_name] = {'top_1_accuracy': top_1_accuracy, 'top_5_accuracy': top_5_accuracy}
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

        # save into csv file
        with open(csv_file_path, 'r+', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            exists = [model_name in row for row in reader]
            if True not in exists:
                writer = csv.writer(file, delimiter=',')
                writer.writerow([model_name, top_1_accuracy, top_5_accuracy])


print("\nTHE END!")
