import argparse
import json

import torch

from accelerate import Accelerator
from accelerate.utils import MegatronLMDummyScheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    MegatronBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from general_utils import create_helper_directories
from train_utils import (
    prepare_model_for_training_on_multiple_gpu,
    prepare_model_for_training_on_single_gpu,
    train_model_on_multiple_gpu,
    train_model_on_single_gpu,
)


def main(config):
    model_name = config["Model"]["modelName"]
    max_length = config["Model"]["sequenceLength"]
    padding_value = config["Model"]["paddingValue"]
    truncation_flag = config["Model"]["truncationFlag"]
    return_tensors = config["Model"]["returnTensors"]
    special_token_flag = config["Model"]["specialTokenFlag"]

    dataset_class = config["Dataset"]["datasetClass"]
    dataset_name = config["Dataset"]["datasetName"]
    label_count = config["Dataset"]["labelCount"]
    shuffle_flag = config["Dataset"]["shuffleFlag"]

    checkpoint_dir = config["Logs"]["checkpointDirectory"]
    logs_dir = config["Logs"]["logsDirectory"]
    task_name = config["Logs"]["taskName"]

    batch_size = config["Train"]["batchSize"]
    start_epoch = config["Train"]["startEpoch"]
    end_epoch = config["Train"]["endEpoch"]
    learning_rate = config["Train"]["learningRate"]

    continue_flag = config["Train"]["continueFlag"]
    continue_checkpoint_path = config["Train"]["continueCheckpointPath"]
    accelerate_checkpoint_dir = config["Train"]["accelerateCheckpointDir"]

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            print("GPU Available!")
            print("Number of GPUs present: {}!".format(num_gpus))
            device = torch.device("cuda")
            model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=label_count
            )
            multi_gpu_flag = False
        else:
            accelerator = Accelerator()
            device = accelerator.device
            accelerator.print("GPU Available!")
            accelerator.print("Number of GPUs present: {}!".format(num_gpus))
            model = MegatronBertForSequenceClassification.from_pretrained(
                "{}/{}".format("models", model_name), num_labels=label_count
            )
            multi_gpu_flag = True
    else:
        print("GPU not available using CPU!")
        device = torch.device("cpu")

    checkpoint_dir, logs_path = create_helper_directories(
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        task_name=task_name,
        flag=True,
        print_flag=multi_gpu_flag,
    )

    torch.cuda.empty_cache()

    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = load_dataset(dataset_class, dataset_name)

    def tokenize_function(data):
        return tokenizer(
            data["sentence"],
            max_length=max_length,
            padding=padding_value,
            truncation=truncation_flag,
            return_tensors=return_tensors,
            add_special_tokens=special_token_flag,
        )

    train_data = dataset["train"]

    if multi_gpu_flag:
        with accelerator.main_process_first():
            train_data = train_data.map(
                tokenize_function, batched=True, remove_columns=["idx", "sentence"]
            )
    else:
        train_data = train_data.map(
            tokenize_function, batched=True, remove_columns=["idx", "sentence"]
        )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_flag)

    num_epochs = end_epoch - start_epoch + 1
    training_steps = len(train_loader) * num_epochs
    warmup_steps = 100

    if not multi_gpu_flag:
        print("Training using a single GPU!")
        model, optimizer = prepare_model_for_training_on_single_gpu(
            model=model,
            device=device,
            learning_rate=learning_rate,
            continue_flag=continue_flag,
            continue_checkpoint_path=continue_checkpoint_path,
        )
        training_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
        train_model_on_single_gpu(
            model=model,
            device=device,
            optimizer=optimizer,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            data_loader=train_loader,
            training_scheduler=training_scheduler,
            logs_path=logs_path,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        accelerator.print("Using MegatronLM for training across multiple GPUs!")
        model, optimizer, accelerator = prepare_model_for_training_on_multiple_gpu(
            model=model,
            device=device,
            learning_rate=learning_rate,
            continue_flag=continue_flag,
            accelerator=accelerator,
            accelerate_checkpoint_dir=accelerate_checkpoint_dir,
        )
        training_scheduler = MegatronLMDummyScheduler(
            optimizer=optimizer,
            total_num_steps=training_steps,
            warmup_num_steps=warmup_steps,
        )
        model, optimizer, train_loader, training_scheduler = accelerator.prepare(
            model, optimizer, train_loader, training_scheduler
        )
        train_model_on_multiple_gpu(
            model=model,
            device=device,
            accelerator=accelerator,
            optimizer=optimizer,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            data_loader=train_loader,
            training_scheduler=training_scheduler,
            logs_path=logs_path,
            checkpoint_dir=checkpoint_dir,
        )


if __name__ == "__main__":
    print("\n--------------------\nStarting model training!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model training")
    parser.add_argument(
        "--config", "-C", type=str, help="Config file for model training", required=True
    )
    args = parser.parse_args()

    json_filename = args.config
    with open(json_filename, "rt") as json_file:
        config = json.load(json_file)

    main(config=config)
    print("\n--------------------\nModel training completed!\n--------------------\n")
