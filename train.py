from typing import List, Tuple
from simpletransformers.classification import ClassificationModel, ClassificationArgs

import logging

import pandas as pd


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def retrieve_instances_from_dataset(
    dataset: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Retrieve sentences with insertions from dataset.
    :param dataset: dataframe with labeled data
    :return: a tuple with
    * a list of id strs
    * a list of sentence strs
    """
    # fill the empty values with empty strings
    dataset = dataset.fillna("")

    ids = []
    instances = []

    for _, row in dataset.iterrows():
        for filler_index in range(1, 6):
            ids.append(f"{row['Id']}_{filler_index}")

            sent_with_filler = row["Sentence"].replace(
                "______", row[f"Filler{filler_index}"]
            )


            result = f'{row["Previous context"] } {sent_with_filler} { row["Follow-up context"] }'
            instances.append(result)

    return ids, instances

def retrieve_labels_from_dataset_for_ranking(label_set: pd.DataFrame) -> List[float]:
    """Retrieve labels from dataset.

    :param label_set: dataframe with plausibility gold scores
    :return: list of rating floats
    """
    # the labels are already in the right order for the training instances, so we can just put them in a list
    return list(label_set["Label"])


def retrieve_labels_from_dataset_for_classification(
    label_set: pd.DataFrame,
) -> List[int]:
    """Retrieve labels from dataset.
    :param label_set: dataframe with class labels
    :return: list of int class labels 0, 1 or 2 (IMPLAUSIBLE, NEUTRAL, PLAUSIBLE)
    """
    # the labels are already in the right order for the training instances, so we can just put them in a list
    label_strs = list(label_set["Label"])
    label_ints = []

    for label_str in label_strs:
        if label_str == "IMPLAUSIBLE":
            label_ints.append(0)
        elif label_str == "NEUTRAL":
            label_ints.append(1)
        elif label_str == "PLAUSIBLE":
            label_ints.append(2)
        else:
            raise ValueError(f"Label {label_str} is not a valid plausibility class.")

    return label_ints

import argparse
import logging



logging.basicConfig(level=logging.DEBUG)


def check_format_of_training_dataset(training_dataset: pd.DataFrame) -> None:
    """Check the format of dataframe with training data.

    :param training_dataset: dataframe with training set
    """
    logging.debug("Verifying the format of training dataset")

    required_columns = [
        "Id",
        "Resolved pattern",
        "Article title",
        "Section header",
        "Previous context",
        "Sentence",
        "Follow-up context",
        "Filler1",
        "Filler2",
        "Filler3",
        "Filler4",
        "Filler5",
    ]

    if not list(training_dataset.columns) == required_columns:
        raise ValueError(
            f"File does not have the required columns: {list(training_dataset.columns)} != {required_columns}."
        )

    for id in training_dataset["Id"]:
        try:
            int(id)
        except ValueError:
            raise ValueError(f"Id {id} is not a valid integer.")

    valid_patterns = [
        "IMPLICIT REFERENCE",
        "ADDED COMPOUND",
        "METONYMIC REFERENCE",
        "FUSED HEAD",
    ]

    for pattern in training_dataset["Resolved pattern"]:
        if pattern not in valid_patterns:
            raise ValueError(
                f"Resolved pattern {pattern} is not among {valid_patterns}."
            )

    for sentence in training_dataset["Sentence"]:
        if "______" not in sentence:
            raise ValueError(
                f"Sentence {sentence} does not contain placeholder '______'."
            )

    for filler_index in range(1, 6):
        for row_index, filler in enumerate(training_dataset[f"Filler{filler_index}"]):
            if not filler:
                raise ValueError(f"One of the fillers in row {row_index} is empty.")

    logging.debug(
        "Format checking for training dataset successful. No problems detected."
    )

train_set = pd.read_csv('data/train.tsv', sep="\t", quoting=3)
dev_set = pd.read_csv('data/dev.tsv', sep="\t", quoting=3)
_, train_instances = retrieve_instances_from_dataset(train_set)
_, dev_instances = retrieve_instances_from_dataset(dev_set)
label_set = pd.read_csv('data/labels.tsv', sep="\t", header=None, names=["Id", "Label"])
dev_labels = pd.read_csv('data/dev_label.tsv', sep="\t", header=None, names=["Id", "Label"])

rank_set = pd.read_csv('data/scores.tsv', sep="\t", header=None, names=["Id", "Label"])

dev_rank_set = pd.read_csv('data/dev_scores.tsv', sep="\t", header=None, names=["Id", "Label"])

labels = retrieve_labels_from_dataset_for_classification(label_set)
labels_dev = retrieve_labels_from_dataset_for_classification(dev_labels)

ranking = retrieve_labels_from_dataset_for_ranking(rank_set)
dev_ranking = retrieve_labels_from_dataset_for_ranking(dev_rank_set)

train_df = pd.DataFrame({
    'text': train_instances,
    'labels': labels
    })
eval_df = pd.DataFrame({
    'text': dev_instances,
    'labels': labels_dev
    })

# model_args = ClassificationArgs()
# model_args.eval_batch_size = 4
# # model_args.evaluate_during_training = True
# model_args.evaluate_during_training_silent = False
# model_args.evaluate_during_training_steps = 1000
# model_args.learning_rate = 4e-5
# model_args.manual_seed = 4
# model_args.max_seq_length = 256
# model_args.multiprocessing_chunksize = 5000
# model_args.no_cache = True

# model_args.num_train_epochs = 10
# model_args.overwrite_output_dir = True
# model_args.reprocess_input_data = True
# model_args.train_batch_size = 16
# model_args.regression = True
# model_args.gradient_accumulation_steps = 2
# model_args.train_custom_parameters_only = False
# model_args.save_eval_checkpoints = False
# model_args.save_model_every_epoch = False
# # model_args.labels_list = [0,1]
# model_args.output_dir = "tuned_output"
# # model_args.wandb_project = 'SemEval2022'
# model_args.train_custom_parameters_only = False
# model_args.best_model_dir = "tuned_output/best_model"

# model = ClassificationModel('roberta', 'roberta-base', num_labels=1, use_cuda=True, args=model_args)
# model.train_model(
#     train_df
# )

# predictions, raw_outputs = model.predict(eval_df["text"].tolist())

# eval_df["scores"] = predictions

# eval_df.to_csv("data/submission.csv")

import logging
from statistics import mean

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

# import wandb
from simpletransformers.classification import ClassificationArgs, ClassificationModel


layer_parameters = {f"layer_{i}-{i + 6}": {"min": 0.0, "max": 5e-5} for i in range(0, 24, 6)}

sweep_config = {
    "name": "Task7Sweep1",
    "method": "bayes",
    "metric": {"name": "mse", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"min": 1, "max": 40},
        "params_classifier.dense.weight": {"min": 0.0, "max": 1e-3},
        "params_classifier.dense.bias": {"min": 0.0, "max": 1e-3},
        "params_classifier.out_proj.weight": {"min": 0.0, "max": 1e-3},
        "params_classifier.out_proj.bias": {"min": 0.0, "max": 1e-3},
        **layer_parameters,
    },
    "early_terminate": {"type": "hyperband", "min_iter": 6,},
}

# sweep_id = wandb.sweep(sweep_config, project="Task7-Opt2")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args = ClassificationArgs()
model_args.eval_batch_size = 32
model_args.evaluate_during_training = False
model_args.evaluate_during_training_silent = False
model_args.evaluate_during_training_steps = 1000
model_args.learning_rate = 4e-5
model_args.manual_seed = 4
model_args.max_seq_length = 256
model_args.multiprocessing_chunksize = 5000
model_args.no_cache = True
model_args.no_save = True
model_args.num_train_epochs = 10
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 2
model_args.train_custom_parameters_only = False
model_args.labels_list = [0, 1, 2]
# model_args.regression = True
# model_args.wandb_project = "Task7-Opt2"

model = ClassificationModel('electra', 'google/electra-large-discriminator', num_labels=3, use_cuda=True, args=model_args)
    
model.train_model(
        train_df,
       
)

predictions, raw_outputs = model.predict(eval_df["text"].tolist())
eval_df["scores"] = predictions

eval_df.to_csv("data/submission2.csv")

