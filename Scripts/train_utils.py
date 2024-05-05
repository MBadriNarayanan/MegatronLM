import os
import time
import torch
import torch.cuda

from sklearn.metrics import accuracy_score
from torch.nn.functional import log_softmax
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from general_utils import get_model_parameters


def prepare_model_for_training_on_multiple_gpu(
    model,
    device,
    learning_rate,
    continue_flag,
    accelerator="",
    accelerate_checkpoint_dir="",
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    if continue_flag:
        accelerator.print("Model loaded for further training!")
        accelerator.load_state(accelerate_checkpoint_dir)
    else:
        accelerator.print("Prepared model for training!")
    model.train()
    return model, optimizer, accelerator


def prepare_model_for_training_on_single_gpu(
    model,
    device,
    learning_rate,
    continue_flag,
    continue_checkpoint_path="",
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    if continue_flag:
        print("Model loaded for further training!")
        checkpoint = torch.load(continue_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Prepared model for training!")
    model.train()
    get_model_parameters(model=model)
    return model, optimizer


def train_model_on_multiple_gpu(
    model,
    device,
    accelerator,
    optimizer,
    start_epoch,
    end_epoch,
    data_loader,
    training_scheduler,
    logs_path,
    checkpoint_dir,
):
    with open(logs_path, "at") as logs_file:
        logs_file.write(
            "Logs for the checkpoint stored at: {}/\n".format(checkpoint_dir)
        )

    number_of_epochs = end_epoch - start_epoch + 1

    avg_train_loss = 0.0
    avg_train_accuracy = 0.0
    avg_train_duration = 0.0
    avg_train_batch_time = 0.0

    for epoch in tqdm(range(start_epoch, end_epoch + 1)):
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_train_duration = 0.0
        avg_train_batch_duration = 0.0

        train_epoch_start_time = time.time()

        for batch_idx, data_batch in enumerate(data_loader):
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

            batch_start_time = time.time()

            with accelerator.accumulate(model):
                output = model(**data_dict, labels=labels)
            loss = output.loss
            logits = output.logits
            batch_loss = loss.item()

            clip_grad_norm_(model.parameters(), 1.0)
            accelerator.backward(loss)
            optimizer.step()
            training_scheduler.step()
            optimizer.zero_grad()

            y_pred = log_softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()
            target = labels.cpu().numpy().reshape(labels.shape[0])
            batch_accuracy = accuracy_score(target, y_pred)

            epoch_train_loss += batch_loss
            epoch_train_accuracy += batch_accuracy

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            avg_train_batch_duration += batch_duration

            if batch_idx % 100 == 0:
                write_string = "Epoch: {}, Train Batch Idx: {}, Train Batch Loss: {:.3f}, Train Batch Accuracy: {:.3f}, Train Batch Duration: {:.3f} seconds\n".format(
                    epoch, batch_idx, batch_loss, batch_accuracy, batch_duration
                )
                with open(logs_path, "at") as logs_file:
                    logs_file.write(write_string)
                del write_string

            torch.cuda.empty_cache()

            del data_dict, labels
            del output, loss, logits
            del target, y_pred
            del batch_loss, batch_accuracy
            del batch_start_time, batch_end_time
            del batch_duration

        epoch_train_loss /= len(data_loader)
        epoch_train_accuracy /= len(data_loader)
        avg_train_batch_duration /= len(data_loader)

        train_epoch_end_time = time.time()
        epoch_train_duration = train_epoch_end_time - train_epoch_start_time

        avg_train_loss += epoch_train_loss
        avg_train_accuracy += epoch_train_accuracy
        avg_train_duration += epoch_train_duration
        avg_train_batch_time += avg_train_batch_duration

        write_string = "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
            epoch,
            epoch_train_loss,
            epoch_train_accuracy,
            epoch_train_duration,
            avg_train_batch_duration,
        )
        with open(logs_path, "at") as logs_file:
            logs_file.write(write_string)
            logs_file.write("----------------------------------------------\n")
        del write_string

        accelerator.save_state(checkpoint_dir)

        del epoch_train_loss, epoch_train_accuracy
        del epoch_train_duration, avg_train_batch_duration
        del train_epoch_start_time, train_epoch_end_time

    avg_train_loss /= number_of_epochs
    avg_train_accuracy /= number_of_epochs
    avg_train_duration /= number_of_epochs
    avg_train_batch_time /= number_of_epochs

    write_string = "Avg Train Loss: {:.3f}, Avg Train Accuracy: {:.3f}, Avg Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
        avg_train_loss,
        avg_train_accuracy,
        avg_train_duration,
        avg_train_batch_time,
    )
    with open(logs_path, "at") as logs_file:
        logs_file.write(write_string)
        logs_file.write("----------------------------------------------\n")

    del write_string
    del avg_train_loss, avg_train_accuracy
    del avg_train_duration, avg_train_batch_time


def train_model_on_single_gpu(
    model,
    device,
    optimizer,
    start_epoch,
    end_epoch,
    data_loader,
    training_scheduler,
    logs_path,
    checkpoint_dir,
):
    with open(logs_path, "at") as logs_file:
        logs_file.write(
            "Logs for the checkpoint stored at: {}/\n".format(checkpoint_dir)
        )

    number_of_epochs = end_epoch - start_epoch + 1

    avg_train_loss = 0.0
    avg_train_accuracy = 0.0
    avg_train_duration = 0.0
    avg_train_batch_time = 0.0

    for epoch in tqdm(range(start_epoch, end_epoch + 1)):
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_train_duration = 0.0
        avg_train_batch_duration = 0.0

        train_epoch_start_time = time.time()

        for batch_idx, data_batch in enumerate(data_loader):
            input_ids = torch.stack(data_batch["input_ids"], dim=1)
            attention_mask = torch.stack(data_batch["attention_mask"], dim=1)
            labels = data_batch["label"]

            input_ids = input_ids.view(input_ids.size(0), -1).to(device)
            attention_mask = attention_mask.view(attention_mask.size(0), -1).to(device)
            labels = labels.view(labels.size(0), -1).to(device)

            batch_start_time = time.time()

            optimizer.zero_grad()

            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            logits = output.logits
            batch_loss = loss.item()

            clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            training_scheduler.step()
            optimizer.zero_grad()

            y_pred = log_softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()
            target = labels.cpu().numpy().reshape(labels.shape[0])
            batch_accuracy = accuracy_score(target, y_pred)

            epoch_train_loss += batch_loss
            epoch_train_accuracy += batch_accuracy

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            avg_train_batch_duration += batch_duration

            if batch_idx % 100 == 0:
                write_string = "Epoch: {}, Train Batch Idx: {}, Train Batch Loss: {:.3f}, Train Batch Accuracy: {:.3f}, Train Batch Duration: {:.3f} seconds\n".format(
                    epoch, batch_idx, batch_loss, batch_accuracy, batch_duration
                )
                with open(logs_path, "at") as logs_file:
                    logs_file.write(write_string)
                del write_string

            torch.cuda.empty_cache()

            del input_ids, attention_mask, labels
            del output, loss, logits
            del target, y_pred
            del batch_loss, batch_accuracy
            del batch_start_time, batch_end_time
            del batch_duration

        epoch_train_loss /= len(data_loader)
        epoch_train_accuracy /= len(data_loader)
        avg_train_batch_duration /= len(data_loader)

        train_epoch_end_time = time.time()
        epoch_train_duration = train_epoch_end_time - train_epoch_start_time

        avg_train_loss += epoch_train_loss
        avg_train_accuracy += epoch_train_accuracy
        avg_train_duration += epoch_train_duration
        avg_train_batch_time += avg_train_batch_duration

        write_string = "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
            epoch,
            epoch_train_loss,
            epoch_train_accuracy,
            epoch_train_duration,
            avg_train_batch_duration,
        )
        with open(logs_path, "at") as logs_file:
            logs_file.write(write_string)
            logs_file.write("----------------------------------------------\n")
        del write_string

        ckpt_path = "{}/Epoch_{}.pt".format(checkpoint_dir, epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_train_loss,
            },
            ckpt_path,
        )

        del epoch_train_loss, epoch_train_accuracy
        del epoch_train_duration, avg_train_batch_duration
        del train_epoch_start_time, train_epoch_end_time

    avg_train_loss /= number_of_epochs
    avg_train_accuracy /= number_of_epochs
    avg_train_duration /= number_of_epochs
    avg_train_batch_time /= number_of_epochs

    write_string = "Avg Train Loss: {:.3f}, Avg Train Accuracy: {:.3f}, Avg Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
        avg_train_loss,
        avg_train_accuracy,
        avg_train_duration,
        avg_train_batch_time,
    )
    with open(logs_path, "at") as logs_file:
        logs_file.write(write_string)
        logs_file.write("----------------------------------------------\n")

    del write_string
    del avg_train_loss, avg_train_accuracy
    del avg_train_duration, avg_train_batch_time
