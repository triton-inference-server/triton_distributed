#!/usr/bin/env python3

# To run this script you need to install:
# pip install pandas matplotlib seaborn

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def parse_tp_dp(name):
    baseline_match = re.search(r"baseline_tp(\d+)dp(\d+)", name)
    context_match = re.search(r"context_tp(\d+)dp(\d+)", name)
    generate_match = re.search(r"generate_tp(\d+)dp(\d+)", name)

    baseline_gpus = (
        int(baseline_match.group(1)) * int(baseline_match.group(2))
        if baseline_match
        else 0
    )
    context_gpus = (
        int(context_match.group(1)) * int(context_match.group(2))
        if context_match
        else 0
    )
    generate_gpus = (
        int(generate_match.group(1)) * int(generate_match.group(2))
        if generate_match
        else 0
    )

    return baseline_gpus + context_gpus + generate_gpus


def get_latest_run_dirs(base_path):
    latest_run_dirs = defaultdict(list)

    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            concurrency_dirs = [
                d for d in os.listdir(subdir_path) if d.startswith("concurrency_")
            ]
            valid_dirs = defaultdict(list)
            for d in concurrency_dirs:
                concurrency = d.split("_")[1]
                json_path = os.path.join(
                    subdir_path, d, "my_profile_export_genai_perf.json"
                )
                if os.path.exists(json_path):
                    valid_dirs[concurrency].append(d)
            for valid_dir in valid_dirs.values():
                latest_dir = max(
                    valid_dir,
                    key=lambda d: datetime.strptime(
                        d.split("_")[2] + d.split("_")[3], "%Y%m%d%H%M%S"
                    ),
                )
                concurrency = latest_dir.split("_")[1]
                latest_run_dirs[subdir].append(latest_dir)
    return latest_run_dirs


def extract_val_and_concurrency(base_path, latest_run_dirs):
    results = []
    for subdir, latest_dirs in latest_run_dirs.items():
        for latest_dir in latest_dirs:
            json_path = os.path.join(
                base_path,
                subdir,
                latest_dir,
                "my_profile_export_genai_perf.json",
            )
            with open(json_path, "r") as f:
                data = json.load(f)
                output_token_throughput = data.get("output_token_throughput").get("avg")
                output_token_throughput_per_request = data.get(
                    "output_token_throughput_per_request"
                ).get("avg")
                time_to_first_token = data.get("time_to_first_token").get("avg")
                inter_token_latency = data.get("inter_token_latency").get("avg")
                request_throughput = data.get("request_throughput").get("avg")
            concurrency = latest_dir.split("_")[1]
            num_gpus = parse_tp_dp(subdir)
            output_token_throughput_per_gpu = output_token_throughput / num_gpus
            request_throughput_per_gpu = request_throughput / num_gpus
            results.append(
                {
                    "configuration": subdir,
                    "num_gpus": num_gpus,
                    "concurrency": float(concurrency),
                    "output_token_throughput": output_token_throughput,
                    "output_token_throughput_per_request": output_token_throughput_per_request,
                    "output_token_throughput_per_gpu": output_token_throughput_per_gpu,
                    "time_to_first_token": time_to_first_token,
                    "inter_token_latency": inter_token_latency,
                    "request_throughput_per_gpu": request_throughput_per_gpu,
                }
            )
    return results


def create_graph(base_path, results):
    points = [
        {
            "label": result["configuration"],
            "order": float(result["concurrency"]),
            "x": result["output_token_throughput_per_request"],
            "y": result["output_token_throughput_per_gpu"],
        }
        for result in results
    ]
    df = pd.DataFrame(points)
    df_sorted = df.sort_values(by=["label", "order"])

    # Use seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df_sorted.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    # Add legend
    plt.legend(title="Legend")

    # Add labels and title
    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.title("Results")
    # Get the current axes
    ax = plt.gca()

    # # Set the major tick locator for both x and y axes
    # x_interval = 10  # Set your desired x-axis interval
    # y_interval = 20  # Set your desired y-axis interval
    # ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    # ax.yaxis.set_major_locator(MultipleLocator(y_interval))

    # Save the plot to a file
    plt.savefig(f"{base_path}/plot.png", dpi=300)  # Save as PNG with high resolution


def create_itl_graph(base_path, results):
    points = [
        {
            "label": result["configuration"],
            "order": float(result["concurrency"]),
            "x": result["concurrency"],
            "y": result["inter_token_latency"],
        }
        for result in results
    ]
    df = pd.DataFrame(points)
    df_sorted = df.sort_values(by=["label", "order"])

    # Use seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df_sorted.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    # Add legend
    plt.legend(title="Legend")

    # Add labels and title
    plt.xlabel("concurrency")
    plt.ylabel("inter_token_latency")
    plt.title("Results")
    # Get the current axes
    ax = plt.gca()

    ax.set_xscale("log")
    ax.set_yscale("log")
    # # Set the major tick locator for both x and y axes
    # x_interval = 10  # Set your desired x-axis interval
    # y_interval = 20  # Set your desired y-axis interval
    # ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    # ax.yaxis.set_major_locator(MultipleLocator(y_interval))

    # Save the plot to a file
    plt.savefig(
        f"{base_path}/plot_itl.png", dpi=300
    )  # Save as PNG with high resolution


def create_ttft_graph(base_path, results):
    points = [
        {
            "label": result["configuration"],
            "order": float(result["concurrency"]),
            "x": result["concurrency"],
            "y": result["time_to_first_token"],
        }
        for result in results
    ]
    df = pd.DataFrame(points)
    df_sorted = df.sort_values(by=["label", "order"])

    # Use seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df_sorted.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    # Add legend
    plt.legend(title="Legend")

    # Add labels and title
    plt.xlabel("concurrency")
    plt.ylabel("time_to_first_token")
    plt.title("Results")
    # Get the current axes
    ax = plt.gca()

    ax.set_xscale("log")
    ax.set_yscale("log")
    # # Set the major tick locator for both x and y axes
    # x_interval = 10  # Set your desired x-axis interval
    # y_interval = 20  # Set your desired y-axis interval
    # ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    # ax.yaxis.set_major_locator(MultipleLocator(y_interval))

    # Save the plot to a file
    plt.savefig(
        f"{base_path}/plot_ttft.png", dpi=300
    )  # Save as PNG with high resolution


def create_req_graph(base_path, results):
    points = [
        {
            "label": result["configuration"],
            "order": float(result["concurrency"]),
            "x": result["concurrency"],
            "y": result["request_throughput_per_gpu"],
        }
        for result in results
    ]
    df = pd.DataFrame(points)
    df_sorted = df.sort_values(by=["label", "order"])

    # Use seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df_sorted.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    # Add legend
    plt.legend(title="Legend")

    # Add labels and title
    plt.xlabel("concurrency")
    plt.ylabel("request_throughput_per_gpu")
    plt.title("Results")
    # Get the current axes
    ax = plt.gca()

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # # Set the major tick locator for both x and y axes
    # x_interval = 10  # Set your desired x-axis interval
    # y_interval = 20  # Set your desired y-axis interval
    # ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    # ax.yaxis.set_major_locator(MultipleLocator(y_interval))

    # Save the plot to a file
    plt.savefig(
        f"{base_path}/plot_req.png", dpi=300
    )  # Save as PNG with high resolution


def create_pareto_graph(base_path, results):
    data_points = [
        {
            "label": result["configuration"].split("_")[0].replace("context", "disagg"),
            "configuration": result["configuration"],
            "concurrency": float(result["concurrency"]),
            "output_token_throughput_per_request": result[
                "output_token_throughput_per_request"
            ],
            "output_token_throughput_per_gpu": result[
                "output_token_throughput_per_gpu"
            ],
            "time_to_first_token": result["time_to_first_token"],
            "inter_token_latency": result["inter_token_latency"],
            "is_pareto_efficient": False,
        }
        for result in results
    ]
    # Load data into a pandas DataFrame
    df = pd.DataFrame(data_points)

    # Function to find Pareto-efficient points
    def pareto_efficient(ids, points):
        points = np.array(points)
        pareto_points = []
        for i, (point_id, point) in enumerate(zip(ids, points)):
            dominated = False
            for j, other_point in enumerate(points):
                if i != j and all(other_point >= point):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(point)
                df.at[point_id, "is_pareto_efficient"] = True
        return np.array(pareto_points)

    # Plot Pareto frontier for each label
    plt.figure(figsize=(10, 6))
    labels = df["label"].unique()

    for label in labels:
        group = df[df["label"] == label]

        # Plot the points
        plt.scatter(
            group["output_token_throughput_per_request"],
            group["output_token_throughput_per_gpu"],
            label=f"Label {label}",
        )

        # Find and plot Pareto-efficient points
        pareto_points = pareto_efficient(
            group.index,
            group[
                [
                    "output_token_throughput_per_request",
                    "output_token_throughput_per_gpu",
                ]
            ].values,
        )

        # Sort Pareto points by x-axis for plotting
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
        plt.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            linestyle="--",
            label=f"Pareto Frontier {label}",
        )

    df.to_csv(f"{base_path}/results.csv")

    # Add labels and legend
    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.title("Llama3.1 70B vLLM FP8, EOS H100, ISL cached / uncached / OSL 0/3000/150")
    plt.legend()
    plt.grid(True)
    # Get the current axes
    ax = plt.gca()

    # # Set the major tick locator for both x and y axes
    # x_interval = 5  # Set your desired x-axis interval
    # y_interval = 5  # Set your desired y-axis interval
    # ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    # ax.yaxis.set_major_locator(MultipleLocator(y_interval))
    plt.savefig(
        f"{base_path}/pareto_plot.png", dpi=300
    )  # Save as PNG with high resolution


def main(base_path):
    # Usage
    latest_run_dirs = get_latest_run_dirs(base_path)
    extracted_values = extract_val_and_concurrency(base_path, latest_run_dirs)
    print(extracted_values)
    create_graph(base_path, extracted_values)
    create_pareto_graph(base_path, extracted_values)
    create_itl_graph(base_path, extracted_values)
    create_ttft_graph(base_path, extracted_values)
    create_req_graph(base_path, extracted_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GAP results")
    parser.add_argument(
        "base_path", type=str, help="Base path to the results directory"
    )
    args = parser.parse_args()
    main(args.base_path)
