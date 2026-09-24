#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

PYFILE="${PYFILE:-${SCRIPT_DIR}/EPS_TimesFM_debug_best.py}"

TENSOR_NPY="${TENSOR_NPY:-${REPO_ROOT}/fundamentals_analyst_forecast_EPS_regression.npy}"

TICKER_NPY="${TICKER_NPY:-${REPO_ROOT}/fundamentals_analyst_forecast_EPS_regression_ticker.npy}"

XGBOOST_CSV="${XGBOOST_CSV:-${REPO_ROOT}/fundamentals_analyst_forecast_EPS_regression_ready.csv}"

MEANING_PREFIX="${MEANING_PREFIX:-eps_tfm_sweep_June8_2026_no_validation_10_training}"
OUT_BASE="${OUT_BASE:-${SCRIPT_DIR}/${MEANING_PREFIX}}"

export WANDB_PROJECT="${WANDB_PROJECT:-eps-timesfm-tensor-frobenius}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-timesfm-eps-rank739-single-run}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

L_ORTHO="${L_ORTHO:-0.0418271077924523}"
L_FROB="${L_FROB:-1.74951994324895e-05}"

OUT_ROOT="${OUT_BASE}/rank739_lortho_${L_ORTHO}_lfrob_${L_FROB}"
mkdir -p "${OUT_ROOT}"

WANDB_NAME_BASE="${WANDB_NAME:-timesfm-eps-rank739_lortho_${L_ORTHO}_lfrob_${L_FROB}_single_local}"
SEEDS="${SEEDS:-20260519 20260520 20260521 20260522 20260523 20260524 20260525 20260526 20260527 20260528}"
SUMMARY_CSV="${OUT_ROOT}/seed_forecasted_r2_summary.csv"
STATS_CSV="${OUT_ROOT}/seed_forecasted_r2_stats.csv"

echo "============================================================"
echo "10-seed TimesFM EPS job"
echo "Rank_CP: 739"
echo "L_ORTHO: ${L_ORTHO}"
echo "L_FROB: ${L_FROB}"
echo "OUT_ROOT: ${OUT_ROOT}"
echo "WANDB_NAME_BASE: ${WANDB_NAME_BASE}"
echo "SEEDS: ${SEEDS}"
echo "============================================================"

printf "seed,forecasted_r2,results_csv\n" > "${SUMMARY_CSV}"
echo
for SEED in ${SEEDS}; do
  SEED_OUT_ROOT="${OUT_ROOT}/seed_${SEED}"
  mkdir -p "${SEED_OUT_ROOT}"
  export WANDB_NAME="${WANDB_NAME_BASE}_seed_${SEED}"
  echo "------------------------------------------------------------"
  echo "Seed: ${SEED}"
  echo "SEED_OUT_ROOT: ${SEED_OUT_ROOT}"
  echo "WANDB_NAME: ${WANDB_NAME}"
  echo "------------------------------------------------------------"
  "${PYTHON_BIN}" -u "${PYFILE}" \
  --use_timesfm true \
  --trainable_weights false \
  --epochs_per_window 10 \
  --start_split_ratio 0.1 \
  --minimum_context_length 8 \
  --batch_size 131072 \
  --forecast_batch_size 4096 \
  --reconstruction_batch_size 4096 \
  --number_of_forecast_minibatches 4 \
  --number_of_reconstruction_minibatches 4 \
  --learning_rate 0.001 \
  --weight_decay 0.00001 \
  --num_workers 8 \
  --Rank_CP 739 \
  --mlp_hidden 1024 \
  --dropout_p 0.1 \
  --use_two_stage_forecast true \
  --selected_feature_indices all_except_consensus \
  --driver_feature_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --reset_each_window false \
  --recompute_norm_each_window false \
  --timesfm_repo_id google/timesfm-2.0-500m-pytorch \
  --timesfm_horizon_len 1 \
  --timesfm_context_cap 2048 \
  --timesfm_per_core_batch_size 32 \
  --timesfm_freq_category 2 \
  --cuda 0 \
  --output_dir "${SEED_OUT_ROOT}" \
  --target_feature_index 13 \
  --consensus_feature_index 12 \
  --tensor_npy_path "${TENSOR_NPY}" \
  --ticker_sidecar_npy_path "${TICKER_NPY}" \
  --xgboost_eval_csv_path "${XGBOOST_CSV}" \
  --filter_oos_to_xgboost_eligible false \
  --w_recon 0.03532053274272246 \
  --w_predict 0 \
  --w_forecast_drivers 0.22583411080689977 \
  --w_regression 1 \
  --lambda_ortho_time "${L_ORTHO}" \
  --ortho_penalty_norm L2 \
  --lambda_frobenius_time "${L_FROB}" \
  --make_fiber_forecast_plot false \
  --plot_firm_index 0 \
  --plot_feature_index 0 \
  --plot_all_firms true \
  --plot_png_filename "" \
  --timesfm_baseline_use_gpu true \
  --timesfm_baseline_forecast_all_features true \
  --timesfm_baseline_decode_batch_size 1024 \
  --seed "${SEED}"
  RESULT_CSV="$(find "${SEED_OUT_ROOT}" -type f -name "results_*.csv" -printf "%T@ %p\n" | sort -nr | sed -n "1s/^[^ ]* //p")"
  if [[ -z "${RESULT_CSV}" ]]; then
    echo "ERROR: no results_*.csv found under ${SEED_OUT_ROOT}" >&2
    exit 1
  fi
  FORECASTED_R2="$(awk -F, "NR == 1 { for (i = 1; i <= NF; i++) if (\$i == \"Forecasted_R2\") col = i; next } NR == 2 && col { print \$col; exit }" "${RESULT_CSV}")"
  if [[ -z "${FORECASTED_R2}" ]]; then
    echo "ERROR: Forecasted_R2 not found in ${RESULT_CSV}" >&2
    exit 1
  fi
  printf "%s,%s,%s\n" "${SEED}" "${FORECASTED_R2}" "${RESULT_CSV}" >> "${SUMMARY_CSV}"
  echo "Seed ${SEED} Forecasted_R2=${FORECASTED_R2}"
done
echo "============================================================"
echo "Forecasted_R2 summary across seeds"
awk -F, "BEGIN { print \"n,mean_forecasted_r2,std_forecasted_r2\" } NR > 1 { n++; sum += \$2; sumsq += (\$2 * \$2) } END { if (n == 0) exit 1; mean = sum / n; var = (n > 1) ? (sumsq - (sum * sum / n)) / (n - 1) : 0; if (var < 0 && var > -1e-12) var = 0; printf \"%d,%.10f,%.10f\\n\", n, mean, sqrt(var) }" "${SUMMARY_CSV}" | tee "${STATS_CSV}"
echo "Per-seed summary: ${SUMMARY_CSV}"
echo "Stats summary: ${STATS_CSV}"
echo "============================================================"

echo "Done: seeds=${SEEDS}, Rank_CP=739, L_ORTHO=${L_ORTHO}, L_FROB=${L_FROB}"

# baseline: half data-training, then do prediction. vs. half-data-training, test-time-training. then do prediction. 
# test-time-training is "reconstruction". 
# replicate: "fine-tuning"; no need to training; maximize expected results. 

# cross-sectional or more data --> R^2

# base: cold data. 
# then: data gradually increase/come-in. 
# cold-start: hold firm position, then fresh start, monitor over TTT results. 
# 1) fresh new-start 2) discrepency between data: more/less 3) data come late, still useful: data arrive-random.

# June 2: Still on going.