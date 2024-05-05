import time
import torch
import torch.cuda

import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.functional import log_softmax
from tqdm import tqdm


def generate_report(
    ground_truth,
    prediction,
    loss,
    avg_batch_duration,
    duration,
    report_path,
    checkpoint_path,
):
    accuracy = accuracy_score(ground_truth, prediction)
    report = classification_report(
        ground_truth,
        prediction,
        digits=3,
        zero_division=0,
        target_names=["negative", "positive"],
    )
    matrix = confusion_matrix(ground_truth, prediction)

    with open(report_path, "w") as report_file:
        report_file.write(
            "Validation Metrics for the checkpoint: {}\n".format(checkpoint_path)
        )
        report_file.write(
            "Loss: {:.3f}, Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Duration: {:.3f} seconds\n".format(
                loss, accuracy, avg_batch_duration, duration
            )
        )
        report_file.write("Classification Report\n")
        report_file.write("{}\n".format(report))
        report_file.write("Confusion Matrix\n")
        report_file.write("{}\n".format(matrix))
        report_file.write("--------------------\n")

    del ground_truth, prediction
    del accuracy, report


def evaluate_model_on_multiple_gpu(
    model,
    device,
    accelerator,
    data_loader,
    report_path,
    csv_path,
    checkpoint_path,
    flag="validation",
):
    sentence_data = []
    prediction = []

    evaluation_duration = 0.0
    avg_batch_duration = 0.0

    if flag == "validation":
        ground_truth = []
        val_loss = 0.0

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(data_loader):
            data_dict = {
                "input_ids": torch.stack(data_batch["input_ids"], dim=1).to(device),
                "attention_mask": torch.stack(data_batch["attention_mask"], dim=1).to(
                    device
                ),
                "token_type_ids": torch.stack(data_batch["token_type_ids"], dim=1).to(
                    device
                ),
            }
            labels = data_batch["label"]
            labels = labels.view(labels.size(0), -1).to(device)

            sentence = data_batch["sentence"]
            sentence = list(sentence)

            batch_start_time = time.time()

            if flag == "validation":
                output = model(**data_dict, labels=labels)
                loss = output.loss
                logits = output.logits
                batch_loss = loss.item()

            else:
                output = model(**data_dict)
                logits = output.logits

            y_pred = log_softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().tolist()

            if flag == "validation":
                target = labels.cpu().numpy().reshape(labels.shape[0]).tolist()
                val_loss += batch_loss
                ground_truth += target

            prediction += y_pred
            sentence_data += sentence

            batch_end_time = time.time()
            avg_batch_duration += batch_end_time - batch_start_time

            del data_dict, labels
            del batch_start_time, batch_end_time
            del output, logits
            del y_pred, sentence

            if flag == "validation":
                del loss, target
                del batch_loss

    avg_batch_duration /= len(data_loader)

    end_time = time.time()
    evaluation_duration = end_time - start_time

    dataframe = pd.DataFrame()
    dataframe["Sentence"] = sentence_data
    dataframe["Prediction"] = prediction

    if flag == "validation":
        val_loss /= len(data_loader)
        generate_report(
            ground_truth=ground_truth,
            prediction=prediction,
            loss=val_loss,
            avg_batch_duration=avg_batch_duration,
            duration=evaluation_duration,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
        )
        dataframe["GroundTruth"] = ground_truth
        dataframe = dataframe[["Sentence", "GroundTruth", "Prediction"]]
        del val_loss
        del ground_truth

    dataframe.to_csv(csv_path, index=False)

    del avg_batch_duration
    del evaluation_duration
    del prediction, sentence_data
    del dataframe


def evaluate_model_on_single_gpu(
    model,
    device,
    data_loader,
    report_path,
    csv_path,
    checkpoint_path,
    flag="validation",
):
    sentence_data = []
    prediction = []

    evaluation_duration = 0.0
    avg_batch_duration = 0.0

    if flag == "validation":
        ground_truth = []
        val_loss = 0.0

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(data_loader):
            input_ids = torch.stack(data_batch["input_ids"], dim=1)
            attention_mask = torch.stack(data_batch["attention_mask"], dim=1)
            labels = data_batch["label"]
            sentence = data_batch["sentence"]

            input_ids = input_ids.view(input_ids.size(0), -1).to(device)
            attention_mask = attention_mask.view(attention_mask.size(0), -1).to(device)
            labels = labels.view(labels.size(0), -1).to(device)
            sentence = list(sentence)

            batch_start_time = time.time()

            if flag == "validation":
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                logits = output.logits
                batch_loss = loss.item()

            else:
                output = model(input_ids, attention_mask=attention_mask)
                logits = output.logits

            y_pred = log_softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().tolist()

            if flag == "validation":
                target = labels.cpu().numpy().reshape(labels.shape[0]).tolist()
                val_loss += batch_loss
                ground_truth += target

            prediction += y_pred
            sentence_data += sentence

            batch_end_time = time.time()
            avg_batch_duration += batch_end_time - batch_start_time

            del input_ids, attention_mask, labels
            del batch_start_time, batch_end_time
            del output, logits
            del y_pred, sentence

            if flag == "validation":
                del loss, target
                del batch_loss

    avg_batch_duration /= len(data_loader)

    end_time = time.time()
    evaluation_duration = end_time - start_time

    dataframe = pd.DataFrame()
    dataframe["Sentence"] = sentence_data
    dataframe["Prediction"] = prediction

    if flag == "validation":
        val_loss /= len(data_loader)
        generate_report(
            ground_truth=ground_truth,
            prediction=prediction,
            loss=val_loss,
            avg_batch_duration=avg_batch_duration,
            duration=evaluation_duration,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
        )
        dataframe["GroundTruth"] = ground_truth
        dataframe = dataframe[["Sentence", "GroundTruth", "Prediction"]]
        del val_loss
        del ground_truth

    dataframe.to_csv(csv_path, index=False)

    del avg_batch_duration
    del evaluation_duration
    del prediction, sentence_data
    del dataframe


def prepare_model_for_evaluation_on_multiple_gpu(
    model, device, accelerator, accelerate_checkpoint_dir, flag="validation"
):
    model = model.to(device)
    accelerator.print(
        "Loaded checkpoint: {} for {}!".format(accelerate_checkpoint_dir, flag)
    )
    accelerator.load_state(accelerate_checkpoint_dir)
    accelerator.print("--------------------")
    return model, accelerator


def prepare_model_for_evaluation_on_single_gpu(
    model, device, checkpoint_path, flag="validation"
):
    model = model.to(device)
    print("Loaded checkpoint: {} for {}!".format(checkpoint_path, flag))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("--------------------")
    return model
