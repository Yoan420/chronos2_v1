"""Small CPU diagnostic of the published Rideshare indexing/gradient contract.

This deliberately does not import upstream scripts, install TimesFM, load weights,
or reproduce its benchmark. The tiny model below mirrors only the reviewed
embedding/concatenation, detached forecast and frozen-time training operations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    base = Path(__file__).resolve().parent
    source = base / 'upstream/source/examples/rideshare_3d_tensor/Rideshare_TimesFM_debug_Sept22_3losses_ortho_multi_step.py'
    T, C, H, R = 541, 210, 168, 4
    test_start = T - H
    train = torch.arange(C + 1)
    validation = torch.arange(C + 1, test_start)
    test_context = torch.arange(test_start - C, test_start)

    temporal = nn.Embedding(T, R)
    trip = nn.Embedding(2, R)
    feature = nn.Embedding(3, R)
    decoder = nn.Sequential(nn.Linear(3 * R, 16), nn.ReLU(), nn.Linear(16, 1))
    params = list(temporal.parameters()) + list(trip.parameters()) + list(feature.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=1e-5)

    def decode(z):
        n = len(z)
        return decoder(torch.cat([z, trip(torch.zeros(n, dtype=torch.long)), feature(torch.zeros(n, dtype=torch.long))], dim=1)).squeeze(-1)

    # All reconstruction labels and the temporal decorrelation term use 0..C.
    optimizer.zero_grad(set_to_none=True)
    pred = decode(temporal(train))
    centered = temporal(torch.arange(1, C + 1))
    centered = centered - centered.mean(dim=0)
    cov = centered.T @ centered / (len(centered) - 1)
    offdiag = cov - torch.diag_embed(cov.diagonal())
    loss = ((pred - torch.sin(train.float() / 24)) ** 2).mean() + 0.1 * offdiag.square().mean()
    loss.backward()
    grad = temporal.weight.grad.detach().clone()
    gradient_rows = torch.where(grad.abs().sum(dim=1) > 0)[0]
    assert grad[validation].abs().max().item() == 0
    optimizer.step()  # weight decay may still change rows lacking data gradients.

    # A mock forecaster substitutes for TimesFM. The published contract detaches
    # its predicted factors before forecast supervision and freezes the table.
    mock_validation_factors = temporal(torch.arange(1, C + 1)).mean(dim=0).repeat(len(validation), 1).detach()
    temporal.weight.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    forecast_loss = ((decode(mock_validation_factors) - torch.cos(validation.float() / 24)) ** 2).mean()
    forecast_loss.backward()
    assert temporal.weight.grad is None
    assert any(p.grad is not None for p in decoder.parameters())
    optimizer.step()
    temporal.weight.requires_grad_(True)

    unconditioned = test_context[test_context > C]
    assert len(unconditioned) == 162 and len(test_context) == 210
    result = {
        'kind': 'structural_gradient_diagnostic_NOT_forecast_benchmark',
        'upstream_commit': '130fd4787f268cbdd0d46ac5b840290b7c3440dc',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_lines': {'class': [758, 851], 'detached_helper': [625, 661], 'split': [1945, 1961], 'test_context': [2010, 2014], 'reconstruction': [2065, 2090], 'forecast_freeze': [2120, 2149], 'test_call': [2215, 2226]},
        'shape_and_parameters': {'T': T, 'C': C, 'H': H, 'diagnostic_rank': R, 'diagnostic_hidden': 16},
        'train_rows': [0, C], 'validation_rows': [C + 1, test_start - 1],
        'test_rows': [test_start, T - 1], 'test_context_rows': [test_start - C, test_start - 1],
        'test_context_size': len(test_context),
        'test_context_rows_without_observation_gradient': len(unconditioned),
        'test_context_fraction_without_observation_gradient': len(unconditioned) / len(test_context),
        'reconstruction_nonzero_time_gradient_row_count': len(gradient_rows),
        'validation_time_gradient_max_after_reconstruction': grad[validation].abs().max().item(),
        'time_gradient_after_forecast_supervision': None,
        'decoder_receives_forecast_supervision': True,
        'scope_limit': 'Index/gradient mechanism verified on a small reimplementation. TimesFM and published metrics not run. Rows may undergo optimizer weight decay; no claim they remain bitwise equal to initialization.',
        'torch_version': torch.__version__, 'device': 'cpu',
        'seconds': time.perf_counter() - start,
    }
    (base / 'rideshare_context_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
