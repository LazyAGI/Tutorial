import json
import os

from modelscope.msdatasets import MsDataset


def build_data_path(file_name):
    data_root = os.path.join(os.getcwd(), "dataset")
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    save_path = os.path.join(data_root, file_name)
    return save_path


def get_dataset():
    train_path = build_data_path("train_set.json")
    eval_path = build_data_path("eval_set.json")
    ds = MsDataset.load("modelscope/gsm8k", subset_name="main")
    ds = ds.rename_column("question", "instruction").rename_column(
        "answer", "output"
    )
    with open(train_path, "w") as file:
        json.dump(ds["train"].to_list(), file, ensure_ascii=False, indent=4)
    with open(eval_path, "w") as file:
        json.dump(ds["test"].to_list(), file, ensure_ascii=False, indent=4)
    return train_path, eval_path


get_dataset()