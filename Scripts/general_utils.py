import os
import torch
import torch.cuda
import warnings

import numpy as np

from transformers import set_seed

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
set_seed(random_seed)
warnings.filterwarnings("ignore")


def create_directory(directory, print_flag=True):
    try:
        os.mkdir(directory)
        if not print_flag:
            print("Created directory: {}!".format(directory))
    except:
        if not print_flag:
            print("Directory: {} already exists!".format(directory))


def create_helper_directories(
    checkpoint_dir, logs_dir, task_name, flag=True, print_flag=True
):
    create_directory(directory=checkpoint_dir, print_flag=print_flag)
    checkpoint_dir = os.path.join(checkpoint_dir, task_name)
    create_directory(directory=checkpoint_dir, print_flag=print_flag)

    create_directory(directory=logs_dir, print_flag=print_flag)
    logs_dir = os.path.join(logs_dir, task_name)
    create_directory(directory=logs_dir, print_flag=print_flag)

    if flag:
        logs_path = os.path.join(logs_dir, "logs.txt")
        if not print_flag:
            print("Checkpoints will be stored at: {}!".format(checkpoint_dir))
            print("Training logs will be stored at: {}!".format(logs_path))
        return checkpoint_dir, logs_path
    else:
        validation_logs_dir = os.path.join(logs_dir, "Validation")
        create_directory(directory=validation_logs_dir, print_flag=print_flag)

        test_logs_dir = os.path.join(logs_dir, "Test")
        create_directory(directory=test_logs_dir, print_flag=print_flag)

        val_csv_path = os.path.join(validation_logs_dir, "output.csv")
        test_csv_path = os.path.join(test_logs_dir, "output.csv")

        val_report_path = os.path.join(validation_logs_dir, "report.txt")
        test_report_path = os.path.join(test_logs_dir, "report.txt")

        if not print_flag:
            print(
                "Validation reports will be stored at: {}!".format(validation_logs_dir)
            )
            print("Test reports will be stored at: {}!".format(test_logs_dir))

        return val_csv_path, val_report_path, test_csv_path, test_report_path


def get_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: {}".format(params))
    print("--------------------")
