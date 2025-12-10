import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def plot_discount_factors(discount_factors, pruning_percentages):
    x = np.array(discount_factors)
    y = np.array(pruning_percentages)
    plt.xlabel("Discount Factor")
    plt.ylabel("Pruning Percentage")
    plt.title("Pruning Percentage Experiment")
    plt.xticks(x)
    plt.plot(x, y)
    plt.show()


def compute_cosine_similarity(text1, text2, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(
        [text1, text2], padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state

    print(cosine_similarity(embeddings[0]), embeddings[1])

    similarity = cosine_similarity(embeddings[0], embeddings[1])[0][0]
    return similarity


def plot_cummulative_reward(csv_file_name, x, y):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y])
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.show()


def plot_recalls(csv_file_name, x, y_1, y_2=None, y_3=None):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="ExNonZeroDiscount", marker="o", linestyle="-")
    if y_2 != None:
        plt.plot(data[x], data[y_2], label="Greedy K", marker="s", linestyle="--")
    if y_3 != None:
        plt.plot(data[x], data[y_3], label="Deep Agent", marker="^", linestyle=":")
    plt.xlabel("Enironment size (width, length)")
    plt.ylabel("Recall Percentage")
    plt.legend()
    plt.show()


def plot_recalls_qr(csv_file_name, x, y_1):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="ExNonZeroDiscount", marker="o", linestyle="-")
    # plt.title("Recall percentage")
    plt.xlabel("State Space size")
    plt.ylabel("Recall Percentage")
    plt.legend()
    plt.show()


def plot_recalls_qr_closeness_feature(csv_file_name, x, y_1, y_2):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="Recall - Closeness", marker="o", linestyle="-")
    plt.plot(data[x], data[y_2], label="Recall - Feature", marker="s", linestyle="--")
    # plt.title("Recall percentage")
    plt.xlabel("State Space size")
    plt.ylabel("Recall Percentage")
    plt.legend()
    plt.show()


def plot_cumulative_reward_env_size(csv_file_name, x, y_1, y_2, y_3):
    data = pd.read_csv(csv_file_name)
    plt.plot(
        data[x], data[y_1], label="Training Combined Policy", marker="o", linestyle="-"
    )
    plt.plot(data[x], data[y_2], label="ExNonZeroDiscount", marker="s", linestyle="--")
    plt.plot(data[x], data[y_3], label="Greedy K", marker="^", linestyle=":")
    plt.title("Cumulative Reward based on environment size")
    plt.xlabel("Enironment size (width, length)")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.show()


def plot_speedup(csv_file_name, x, y_1, y_2, y_3):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_3], label="Speedup", marker="o", linestyle="-")
    plt.xlabel("Enironment size (width, length)")
    plt.ylabel("Speedup")
    plt.show()


def plot_speedup_qr(csv_file_name, x, y_1, y_2, y_3):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_3], label="Speedup", marker="o", linestyle="-")
    plt.xlabel("State Space size")
    plt.ylabel("Speedup")
    plt.show()


def plot_recall_deep(csv_file_name, x, y_1):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="Recall (Deep Agent)", marker="o", linestyle="-")
    plt.xlabel("Environment Size")
    plt.ylabel("Recall for Deep Agent (%)")
    plt.show()


def plot_recall_qr_heuristic(csv_file_name, x, y_1, k):
    data = pd.read_csv(csv_file_name)
    plt.plot(
        data[x],
        data[y_1],
        label="Recall - Query Refinement (Greedy-K)",
        marker="o",
        linestyle="-",
    )
    plt.xlabel("State Space size")
    plt.ylabel(f"Recall for Greedy K={k}")
    plt.show()


def plot_cumulative_reward_qr(csv_file_name, x, y_1, y_2, y_3):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="ExNonZeroDiscount", marker="o", linestyle="-")
    plt.plot(
        data[x], data[y_2], label="Training From Scratch", marker="s", linestyle="--"
    )
    plt.plot(data[x], data[y_3], label="Training Greedy-K", marker="^", linestyle=":")
    plt.xlabel("State space size")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()
    plt.show()


def plot_speedup_action_size(csv_file_name, x, y_1, y_2, y_3, y_4, y_5):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_4], label="ExNonZeroDiscount", marker="o", linestyle="-")
    plt.plot(data[x], data[y_5], label="Greedy-K", marker="s", linestyle="--")
    plt.xlabel("Action size")
    plt.ylabel("Speedup (Gridworld)")
    plt.legend()
    plt.show()


# def plot_times(csv_file_name, x, y_1, y_2, y_3):
#     data = pd.read_csv(csv_file_name)
#     plt.plot(data[x], data[y_1], label="ExNonZeroDiscount", marker='o', linestyle='-')
#     plt.plot(data[x], data[y_2], label="Training From Scratch", marker='s', linestyle='--')
#     plt.plot(data[x], data[y_3], label="Training Greedy-K", marker='^', linestyle=':')
#     plt.xlabel("Number of synthetic rewards")
#     plt.ylabel("Time taken for the algorithms")
#     plt.legend()
#     plt.show()


def plot_times(csv_file_name, x, y_1, y_2):
    data = pd.read_csv(csv_file_name)
    plt.plot(data[x], data[y_1], label="ExNonZeroDiscount", marker="o", linestyle="-")
    plt.plot(
        data[x], data[y_2], label="Training From Scratch", marker="s", linestyle="--"
    )
    plt.xlabel("Number of synthetic rewards")
    plt.ylabel("Time taken for the algorithms")
    plt.legend()
    plt.show()
