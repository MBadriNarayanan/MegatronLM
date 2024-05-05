import argparse
import json

import torch

from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    MegatronBertForSequenceClassification,
)
from evaluate_utils import (
    prepare_model_for_evaluation_on_single_gpu,
    prepare_model_for_evaluation_on_multiple_gpu,
    evaluate_model_on_single_gpu,
    evaluate_model_on_multiple_gpu,
)
from general_utils import create_helper_directories


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

    batch_size = config["Eval"]["batchSize"]
    checkpoint_path = config["Eval"]["checkpointPath"]
    accelerate_checkpoint_dir = config["Eval"]["accelerateCheckpointDir"]

    (
        val_csv_path,
        val_report_path,
        test_csv_path,
        test_report_path,
    ) = create_helper_directories(
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        task_name=task_name,
        flag=False,
    )

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

    (
        val_csv_path,
        val_report_path,
        test_csv_path,
        test_report_path,
    ) = create_helper_directories(
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        task_name=task_name,
        flag=False,
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

    val_data = dataset["validation"]
    test_data = dataset["test"]

    if multi_gpu_flag:
        with accelerator.main_process_first():
            val_data = val_data.map(
                tokenize_function, batched=True, remove_columns=["idx"]
            )
            test_data = test_data.map(
                tokenize_function, batched=True, remove_columns=["idx"]
            )
    else:
        val_data = val_data.map(tokenize_function, batched=True, remove_columns=["idx"])
        test_data = test_data.map(
            tokenize_function, batched=True, remove_columns=["idx"]
        )

    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle_flag)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_flag)

    if not multi_gpu_flag:
        print("Evaluating model using a single GPU!")
        model = prepare_model_for_evaluation_on_single_gpu(
            model=model,
            device=device,
            checkpoint_path=checkpoint_path,
            flag="validation",
        )
        evaluate_model_on_single_gpu(
            model=model,
            device=device,
            data_loader=val_loader,
            report_path=val_report_path,
            csv_path=val_csv_path,
            checkpoint_path=checkpoint_path,
            flag="validation",
        )
        model = prepare_model_for_evaluation_on_single_gpu(
            model=model, device=device, checkpoint_path=checkpoint_path, flag="testing"
        )
        evaluate_model_on_single_gpu(
            model=model,
            device=device,
            data_loader=test_loader,
            report_path=test_report_path,
            csv_path=test_csv_path,
            checkpoint_path=checkpoint_path,
            flag="testing",
        )
    else:
        accelerator.print("Using MegatronLM for evaluation across multiple GPUs!")
        model, accelerator = prepare_model_for_evaluation_on_multiple_gpu(
            model=model,
            device=device,
            accelerator=accelerator,
            accelerate_checkpoint_dir=accelerate_checkpoint_dir,
            flag="validation",
        )
        model, val_loader = accelerator.prepare(model, val_loader)
        evaluate_model_on_multiple_gpu(
            model=model,
            device=device,
            accelerator=accelerator,
            data_loader=val_loader,
            report_path=val_report_path,
            csv_path=val_csv_path,
            accelerate_checkpoint_dir=accelerate_checkpoint_dir,
            flag="validation",
        )
        model, accelerator = prepare_model_for_evaluation_on_multiple_gpu(
            model=model,
            device=device,
            accelerator=accelerator,
            accelerate_checkpoint_dir=accelerate_checkpoint_dir,
            flag="testing",
        )
        model, test_loader = accelerator.prepare(model, test_loader)
        evaluate_model_on_multiple_gpu(
            model=model,
            device=device,
            accelerator=accelerator,
            data_loader=test_loader,
            report_path=test_report_path,
            csv_path=test_csv_path,
            accelerate_checkpoint_dir=accelerate_checkpoint_dir,
            flag="testing",
        )


if __name__ == "__main__":
    print("\n--------------------\nStarting model evaluation!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model evaluation")
    parser.add_argument(
        "--config",
        "-C",
        type=str,
        help="Config file for model evaluation",
        required=True,
    )
    args = parser.parse_args()

    json_filename = args.config
    with open(json_filename, "rt") as json_file:
        config = json.load(json_file)

    main(config=config)
    print("\n--------------------\nModel evaluation completed!\n--------------------\n")
