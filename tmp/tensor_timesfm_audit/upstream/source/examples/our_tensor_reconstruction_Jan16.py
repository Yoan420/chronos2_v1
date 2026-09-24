#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Option B: CP-MLP reconstruction-only (RAW), first 80% time, epoch-based training,
and APPEND results to a CSV file.

Reproducibility:
  - seed for random/numpy/torch
  - deterministic cudnn option
  - DataLoader shuffle uses torch.Generator
  - worker_init_fn seeds each worker
"""

import argparse
import csv
import os
import time
import datetime
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_global_seed(seed: int, deterministic_cudnn: bool = True) -> torch.Generator:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(int(seed))
    return data_loader_generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def r2_from_sse_sumy_sumy2(
    sum_squared_error: float, sum_y: float, sum_y2: float, observation_count: int
) -> float:
    if observation_count <= 0:
        return float("nan")
    total_sum_squares = sum_y2 - (sum_y * sum_y) / float(observation_count)
    if total_sum_squares <= 0.0:
        return float("nan")
    return 1.0 - (sum_squared_error / total_sum_squares)


def center_over_time(time_embedding_matrix: torch.Tensor) -> torch.Tensor:
    return time_embedding_matrix - time_embedding_matrix.mean(dim=0, keepdim=True)


def off_diagonal_l2_covariance_penalty(
    time_embedding_matrix: torch.Tensor,
) -> torch.Tensor:
    centered_matrix = center_over_time(time_embedding_matrix)  # [T_train, R]
    training_time_count = int(centered_matrix.shape[0])
    covariance_matrix = (centered_matrix.T @ centered_matrix) / max(
        training_time_count, 1
    )  # [R, R]
    diagonal_matrix = torch.diag(torch.diag(covariance_matrix))  # [R, R]
    off_diagonal_matrix = covariance_matrix - diagonal_matrix  # [R, R]
    return (off_diagonal_matrix.pow(2)).mean()


class ObservedTriplesDataset(Dataset):
    def __init__(self, tensor_3d_cpu: torch.Tensor):
        cpu_tensor = tensor_3d_cpu.detach().cpu()
        observed_mask = ~torch.isnan(cpu_tensor)
        self.triple_indices = observed_mask.nonzero(as_tuple=False).long()  # [M,3]
        self.observed_values = cpu_tensor[observed_mask].float()  # [M]

    def __len__(self) -> int:
        return int(self.observed_values.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.triple_indices[index], self.observed_values[index]


class TensorCPMLPReconstructor(nn.Module):
    def __init__(
        self,
        training_time_count: int,
        firm_count: int,
        feature_count: int,
        rank_embedding_size: int,
        mlp_hidden_size: int,
        dropout_probability: float,
    ):
        super().__init__()
        self.time_embeddings = nn.Embedding(training_time_count, rank_embedding_size)
        self.firm_embeddings = nn.Embedding(firm_count, rank_embedding_size)
        self.feature_embeddings = nn.Embedding(feature_count, rank_embedding_size)

        self.predictor = nn.Sequential(
            nn.Linear(3 * rank_embedding_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(mlp_hidden_size, 1),
        )

    def forward(
        self,
        time_index_batch: torch.Tensor,
        firm_index_batch: torch.Tensor,
        feature_index_batch: torch.Tensor,
    ) -> torch.Tensor:
        time_vector_batch = self.time_embeddings(time_index_batch)  # [B,R]
        firm_vector_batch = self.firm_embeddings(firm_index_batch)  # [B,R]
        feature_vector_batch = self.feature_embeddings(feature_index_batch)  # [B,R]
        concatenated_inputs = torch.cat(
            [time_vector_batch, firm_vector_batch, feature_vector_batch], dim=1
        )  # [B,3R]
        return self.predictor(concatenated_inputs).squeeze(1)  # [B]


@torch.no_grad()
def evaluate_reconstruction_r2_on_dataloader(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float, int]:
    model.eval()
    sum_squared_error = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    observation_count = 0

    for triple_indices_cpu, observed_values_cpu in tqdm(
        dataloader, desc="Eval", leave=False
    ):
        triple_indices = triple_indices_cpu.to(device, non_blocking=True)
        observed_values = observed_values_cpu.to(device, non_blocking=True)

        time_index_batch = triple_indices[:, 0]
        firm_index_batch = triple_indices[:, 1]
        feature_index_batch = triple_indices[:, 2]

        predicted_values = model(
            time_index_batch, firm_index_batch, feature_index_batch
        )

        diff = predicted_values - observed_values
        sum_squared_error += float(torch.sum(diff * diff).item())
        sum_y += float(torch.sum(observed_values).item())
        sum_y2 += float(torch.sum(observed_values * observed_values).item())
        observation_count += int(observed_values.numel())

    r2_value = r2_from_sse_sumy_sumy2(
        sum_squared_error, sum_y, sum_y2, observation_count
    )
    mse_value = (
        sum_squared_error / float(observation_count)
        if observation_count > 0
        else float("nan")
    )
    return float(r2_value), float(mse_value), int(observation_count)


def append_row_to_csv(csv_path: str, header: list[str], row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_npy_path", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument("-r", "--rank_embedding_size", type=int, default=30)
    parser.add_argument("--mlp_hidden_size", type=int, default=1024)
    parser.add_argument("--dropout_probability", type=float, default=0.1)

    parser.add_argument("-e", "--reconstruction_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--lambda_ortho_time", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic_cudnn", type=int, default=1)

    parser.add_argument(
        "--output_csv", type=str, default="./results/cp_mlp_optionB_results.csv"
    )
    parser.add_argument("--run_tag", type=str, default="")

    args = parser.parse_args()

    data_loader_generator = set_global_seed(
        int(args.seed), deterministic_cudnn=bool(int(args.deterministic_cudnn))
    )

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    raw_tensor_numpy = np.load(args.tensor_npy_path).astype(np.float32)
    raw_tensor_cpu = torch.from_numpy(raw_tensor_numpy).float()

    total_time_count, firm_count, feature_count = raw_tensor_cpu.shape
    training_time_count = int(float(args.train_ratio) * float(total_time_count))
    if training_time_count <= 0:
        raise ValueError("training_time_count <= 0. Check train_ratio and T.")

    raw_tensor_train_cpu = raw_tensor_cpu[:training_time_count]
    train_shape = tuple(raw_tensor_train_cpu.shape)
    print(f"[INFO] raw_tensor_train shape = {train_shape}")

    train_dataset = ObservedTriplesDataset(raw_tensor_train_cpu)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        generator=data_loader_generator,
        worker_init_fn=seed_worker if int(args.num_workers) > 0 else None,
    )

    eval_dataloader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        worker_init_fn=seed_worker if int(args.num_workers) > 0 else None,
    )

    model = TensorCPMLPReconstructor(
        training_time_count=training_time_count,
        firm_count=firm_count,
        feature_count=feature_count,
        rank_embedding_size=int(args.rank_embedding_size),
        mlp_hidden_size=int(args.mlp_hidden_size),
        dropout_probability=float(args.dropout_probability),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    mse_loss = nn.MSELoss()

    start_wall_time = time.time()

    for epoch_index in range(1, int(args.reconstruction_epochs) + 1):
        model.train()
        epoch_sum_squared_error = 0.0
        epoch_observation_count = 0

        progress_bar = tqdm(
            train_dataloader, desc=f"Train Epoch {epoch_index}", leave=False
        )
        for triple_indices_cpu, observed_values_cpu in progress_bar:
            triple_indices = triple_indices_cpu.to(device, non_blocking=True)
            observed_values = observed_values_cpu.to(device, non_blocking=True)

            time_index_batch = triple_indices[:, 0]
            firm_index_batch = triple_indices[:, 1]
            feature_index_batch = triple_indices[:, 2]

            optimizer.zero_grad(set_to_none=True)
            predicted_values = model(
                time_index_batch, firm_index_batch, feature_index_batch
            )

            loss_reconstruction = mse_loss(predicted_values, observed_values)

            if float(args.lambda_ortho_time) > 0.0:
                time_embedding_matrix = model.time_embeddings.weight
                ortho_penalty_value = off_diagonal_l2_covariance_penalty(
                    time_embedding_matrix
                )
                total_loss = (
                    loss_reconstruction
                    + float(args.lambda_ortho_time) * ortho_penalty_value
                )
            else:
                total_loss = loss_reconstruction

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            diff = predicted_values - observed_values
            epoch_sum_squared_error += float(torch.sum(diff * diff).item())
            epoch_observation_count += int(observed_values.numel())

            epoch_mse = epoch_sum_squared_error / max(epoch_observation_count, 1)
            progress_bar.set_postfix({"epoch_mse_raw": f"{epoch_mse:.4f}"})

        print(f"[INFO] epoch {epoch_index}: approx_train_mse_raw = {epoch_mse:.6f}")

    r2_raw, mse_raw, observation_count = evaluate_reconstruction_r2_on_dataloader(
        model, eval_dataloader, device
    )
    elapsed_seconds = time.time() - start_wall_time

    print(
        "\n===== Option B: In-sample Reconstruction on RAW first-80%-time slice ====="
    )
    print(f"R2_raw  = {r2_raw:.6f}")
    print(f"MSE_raw = {mse_raw:.6f}")
    print(f"n_obs   = {observation_count}")
    print(f"elapsed_sec = {elapsed_seconds:.3f}")

    header = [
        "timestamp_utc",
        "method",
        "run_tag",
        "tensor_path",
        "train_ratio",
        "train_T",
        "train_shape",
        "n_obs",
        "rank_embedding_size",
        "mlp_hidden_size",
        "dropout_probability",
        "reconstruction_epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "num_workers",
        "lambda_ortho_time",
        "seed",
        "deterministic_cudnn",
        "r2_raw",
        "mse_raw",
        "elapsed_sec",
    ]
    row = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "method": "CP-MLP(concat->MLP)",
        "run_tag": args.run_tag,
        "tensor_path": args.tensor_npy_path,
        "train_ratio": float(args.train_ratio),
        "train_T": int(training_time_count),
        "train_shape": str(train_shape),
        "n_obs": int(observation_count),
        "rank_embedding_size": int(args.rank_embedding_size),
        "mlp_hidden_size": int(args.mlp_hidden_size),
        "dropout_probability": float(args.dropout_probability),
        "reconstruction_epochs": int(args.reconstruction_epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "lambda_ortho_time": float(args.lambda_ortho_time),
        "seed": int(args.seed),
        "deterministic_cudnn": int(args.deterministic_cudnn),
        "r2_raw": float(r2_raw),
        "mse_raw": float(mse_raw),
        "elapsed_sec": float(elapsed_seconds),
    }
    append_row_to_csv(args.output_csv, header, row)
    print(f"[INFO] Appended 1 row to CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

"""
1. dropout = 0.0
2. no normalization. 
3. distribution of prediction. 
4. timesfm prediction. 
5. reconstruction distrubution
6. generate a very simple synthetic tensor with known structure, and see if CP-MLP can reconstruct it.
6.1 then add cross-section; add noise.
7. TimesFM -> check time factor prediction. 
"""
