#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: bash scripts/run_grpo.sh <smoke|pilot|formal> <format|correctness> <seed>"
    echo
    echo "Required environment variables:"
    echo "  MATH_GRPO_MODEL_PATH   Pinned local Qwen checkpoint"
    echo "  MATH_GRPO_TRAIN_FILE   Prepared train.parquet"
    echo "  MATH_GRPO_DEV_FILE     Prepared dev.parquet (never test.parquet)"
    echo "  MATH_GRPO_OUTPUT_ROOT  Checkpoint and validation output root"
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

mode=$1
condition=$2
seed=$3

if [[ ! ${seed} =~ ^[0-9]+$ ]]; then
    echo "Seed must be a non-negative integer." >&2
    exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
reward_path="${project_root}/src/math_grpo/rewards.py"
git_commit=$(git -C "${project_root}" rev-parse HEAD)

required_variables=(
    MATH_GRPO_MODEL_PATH
    MATH_GRPO_TRAIN_FILE
    MATH_GRPO_DEV_FILE
    MATH_GRPO_OUTPUT_ROOT
)
for variable_name in "${required_variables[@]}"; do
    if [[ -z ${!variable_name:-} ]]; then
        echo "Missing required environment variable: ${variable_name}" >&2
        exit 2
    fi
done

for required_file in \
    "${MATH_GRPO_TRAIN_FILE}" \
    "${MATH_GRPO_DEV_FILE}" \
    "${reward_path}"; do
    if [[ ! -f ${required_file} ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 2
    fi
done

if [[ ! -d ${MATH_GRPO_MODEL_PATH} ]]; then
    echo "Model directory not found: ${MATH_GRPO_MODEL_PATH}" >&2
    exit 2
fi

if [[ $(basename "${MATH_GRPO_DEV_FILE}") == "test.parquet" ]]; then
    echo "Refusing to use test.parquet as trainer validation data." >&2
    exit 2
fi

case ${condition} in
    format)
        reward_function=compute_score
        ;;
    correctness)
        reward_function=compute_score_correctness_only
        ;;
    *)
        echo "Condition must be 'format' or 'correctness'." >&2
        exit 2
        ;;
esac

case ${mode} in
    smoke)
        train_batch_size=2
        rollout_n=2
        ppo_mini_batch_size=4
        max_tokens_per_gpu=4096
        total_training_steps=2
        total_epochs=1
        save_freq=-1
        test_freq=-1
        val_before_train=False
        logger='["console"]'
        ;;
    pilot)
        train_batch_size=16
        rollout_n=4
        ppo_mini_batch_size=32
        max_tokens_per_gpu=8192
        total_training_steps=20
        total_epochs=1
        save_freq=10
        test_freq=5
        val_before_train=True
        logger='["console","swanlab"]'
        ;;
    formal)
        train_batch_size=32
        rollout_n=8
        ppo_mini_batch_size=64
        max_tokens_per_gpu=12288
        total_training_steps=null
        total_epochs=2
        save_freq=20
        test_freq=10
        val_before_train=True
        logger='["console","swanlab"]'
        if [[ ${condition} == format && ! ${seed} =~ ^(17|42|2026)$ ]]; then
            echo "Formal format seed must be 17, 42, or 2026." >&2
            exit 2
        fi
        if [[ ${condition} == correctness && ${seed} != 42 ]]; then
            echo "Formal correctness-only seed must be 42." >&2
            exit 2
        fi
        if [[ -n $(git -C "${project_root}" status --porcelain) ]]; then
            echo "Formal runs require a clean Git worktree." >&2
            exit 2
        fi
        ;;
    *)
        echo "Mode must be 'smoke', 'pilot', or 'formal'." >&2
        exit 2
        ;;
esac

run_name="qwen25-05b-${condition}-${mode}-s${seed}"
if [[ ${mode} == formal ]]; then
    run_name="qwen25-05b-${condition}-s${seed}"
fi

run_dir="${MATH_GRPO_OUTPUT_ROOT}/${run_name}"
if [[ -e ${run_dir} && ${MATH_GRPO_ALLOW_EXISTING_RUN_DIR:-0} != 1 ]]; then
    echo "Run directory already exists: ${run_dir}" >&2
    echo "Set MATH_GRPO_ALLOW_EXISTING_RUN_DIR=1 only for intentional debugging reuse." >&2
    exit 2
fi
mkdir -p "${run_dir}"

# Keep SwanLab's local records on the data disk. The caller may override this
# path, for example when using a persistent shared logging directory.
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${MATH_GRPO_OUTPUT_ROOT}/swanlog}"
mkdir -p "${SWANLAB_LOG_DIR}"

export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Launching ${run_name}"
echo "Prompts per step: ${train_batch_size}"
echo "Rollouts per prompt: ${rollout_n}"
echo "Trajectories per step: $((train_batch_size * rollout_n))"
echo "Candidate parameters remain subject to RTX 5090 pilot validation."

python -m verl.trainer.main_ppo \
    +project_git_commit=${git_commit} \
    +project_condition=${condition} \
    +project_seed=${seed} \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    "data.train_files=${MATH_GRPO_TRAIN_FILE}" \
    "data.val_files=${MATH_GRPO_DEV_FILE}" \
    data.seed=${seed} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    "actor_rollout_ref.model.path=${MATH_GRPO_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.data_loader_seed=${seed} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.seed=${seed} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.seed=${seed} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_tokens_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_tokens_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.seed=${seed} \
    "reward.custom_reward_function.path=${reward_path}" \
    reward.custom_reward_function.name=${reward_function} \
    trainer.critic_warmup=0 \
    "trainer.logger=${logger}" \
    trainer.project_name=math-verl-grpo \
    trainer.experiment_name=${run_name} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=${total_epochs} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=${val_before_train} \
    trainer.resume_mode=disable \
    trainer.log_val_generations=16 \
    "trainer.default_local_dir=${run_dir}/checkpoints" \
    "trainer.validation_data_dir=${run_dir}/validation" \
    ray_kwargs.ray_init.runtime_env.py_executable=null
