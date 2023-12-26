import json
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchmetrics
from datetime import datetime


def read_json(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def prefix_zero(n, total_length):
    return '0' * (total_length - len(str(n))) + str(n)


def get_latest_version(model_dir, model_name):
    latest_version, latest_version_path = (-1, "")
    model_dir = Path(model_dir)
    model_provider = model_name.split("/")[0] # "klue"/roberta-large
    model_title = "-".join(model_name.split("/")[1].split()) # klue/"roberta-large"
    # print(model_title)
    for child in model_dir.iterdir():
        if child.is_dir() and child.name == model_provider:
            model_files = list(child.glob(f"{model_title}_*.ckpt"))
            # print(model_files)
            if len(model_files) > 0:
                model_versions = [(int(model_file.stem.split("_")[-9][-2:]), model_file) for model_file in model_files]
                latest_version, latest_version_path = sorted(model_versions, key=lambda x: x[0], reverse=True)[0]
                break
    return latest_version, latest_version_path


# sibling of get_latest_version
def get_version(model_dir, model_name, best=False):
    version, version_path = (-1, "")
    model_dir = Path(model_dir)
    model_provider = model_name.split("/")[0] # "klue"/roberta-large
    model_title = "-".join(model_name.split("/")[1].split()) # klue/"roberta-large"
    # print(model_title)
    for child in model_dir.iterdir():
        if child.is_dir() and child.name == model_provider:
            model_files = list(child.glob(f"{model_title}_*.ckpt"))
            # print(model_files)
            if len(model_files) > 0:
                model_versions = [(int(model_file.stem.split("_")[-9][-2:]), float(model_file.stem.split("_")[-3]), model_file) for model_file in model_files]
                if best:
                    # the best performance version
                    func = lambda x: x[1]
                else: 
                    # latest version
                    func = lambda x: x[0]
                version, version_perf, version_path = sorted(model_versions, key=func, reverse=True)[0]
                break
    return version, version_perf, version_path


def plot_models(model_names: List[str], model_results: torch.Tensor, origin_path:Path, origin_target_name: str, error_gap: float = 0.5):
    if len(model_names) != len(model_results):
        raise ValueError(f"The number of model names {len(model_names)} and model results {len(model_results)} should be the same.")

    # Setting up the figure with 2 rows and 5 columns
    ncols = 2
    nrows = (len(model_names)+1)//2 if (len(model_names)+1) % 2 == 0 else (len(model_names)+1)//2 + 1
    print(f"ncols: {ncols}, nrows: {nrows}")
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5*ncols, 5*nrows))

    origin_df = pd.read_csv(origin_path)
    origin_target_values = torch.tensor(origin_df[origin_target_name].values)

    # Flattening the axes array for easy iteration
    axes = axes.flatten()

    error_count = 0
    # Looping over each model and plotting
    for i, (model_name, model_result) in enumerate(zip(model_names, model_results)):
        # {i}_{model_name}_{model_metric}_{batch_size}
        idx, model_name, _, batch_size = model_name.split("_")
        if int(i) != int(idx):
            raise ValueError(f"The index of model name {i} and model result {idx} should be the same.")

        if origin_target_values.shape != model_result.shape:
            raise ValueError("The shape of origin target values and model result should be the same.")
        metric = torchmetrics.functional.pearson_corrcoef(origin_target_values, model_result)
        # get abosulte error
        error = torch.abs(origin_target_values - model_result)
        error_gap_mask = torch.where(error >= error_gap, 1, 0)
        error_count = error_gap_mask.sum().item()
        error_gap_color = ['red' if e.item() == 1 else 'blue' for e in error_gap_mask]

        # Scatter plot for error_df
        sns.scatterplot(x=origin_target_values, y=model_result, color=error_gap_color, alpha=0.5, ax=axes[i])
        # Adding text labels for error_df
        for error_gap_idx in error_gap_mask.nonzero().flatten():
            axes[i].text(origin_target_values[error_gap_idx], model_result[error_gap_idx] + 0.1, error_gap_idx.item(), fontsize=8)

        # Reference line y=x
        sns.lineplot(x=[0, 5], y=[0, 5], color='black', ax=axes[i])

        # Set plot limits and title
        axes[i].set_xlim(-0.1, 5.5)
        axes[i].set_ylim(-0.1, 5.5)
        axes[i].set_title(f"Name: {model_name}\nBatch size: {batch_size}\nMetric: {metric:.3f}\nError count: {error_count}")
        axes[i].set_xlabel("Origin target values")
        axes[i].set_ylabel("Model result")

        # Customizing plot appearance
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    # Adjust layout to minimize spaces between plots
    fig.suptitle(f"Test model comparison\nError threshold: {error_gap}", fontsize=16)
    plt.tight_layout()
    plot_dir = Path("./plots")
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(f"./plots/plot_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")


def format_pearson(pearson_value):
    # Scale and convert to integer
    return str(int(pearson_value * 1000))


def float_only(n):
    return str(n).split(".")[1]