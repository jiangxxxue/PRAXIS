#!/bin/bash
# 知识冲突测试脚本
# 测试LLM先在框架A上SFT，再在框架B上SFT后，回到框架A的性能表现
#
# 使用方法:
#   bash scripts/run_knowledge_conflict_test.sh

set -e

#########需要手动修改的配置###########
### 1. 基础模型路径
### 2. 第一个训练的框架（FIRST_FRAMEWORK）
### 3. 第二个训练的框架（SECOND_FRAMEWORK）
### 4. 最终测试的框架（TEST_FRAMEWORK，通常与第一个框架相同）
### 5. GPU配置
### 6. 训练参数

# ========================================
# 配置变量
# ========================================

# 基础模型路径
BASE_MODEL_PATH="/workspace/data/models/Qwen2.5-Coder-7B-Instruct"

# 框架配置
FIRST_FRAMEWORK="${FIRST_FRAMEWORK:-verl}"           # 第一个训练的框架
SECOND_FRAMEWORK="${SECOND_FRAMEWORK:-open-r1}"      # 第二个训练的框架
TEST_FRAMEWORK="${TEST_FRAMEWORK:-verl}"             # 测试框架（通常与第一个相同）

# 项目根目录
PROJECT_DIR="/PRAXIS/praxis/domain_code_generation"
SCRIPT_DIR="${PROJECT_DIR}/scripts"

# 模型输出路径
FIRST_SFT_MODEL="${BASE_MODEL_PATH%/}-${FIRST_FRAMEWORK}-sft"
SECOND_SFT_MODEL="${BASE_MODEL_PATH%/}-${FIRST_FRAMEWORK}-${SECOND_FRAMEWORK}-sft"

# 推理服务器配置
SERVER_PORT=8000
SERVER_HOST="0.0.0.0"

# GPU配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 训练参数
MAX_SEQ_LENGTH=2048
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=5e-6
NUM_EPOCHS=2
WARMUP_RATIO=0.03

# 生成参数
NUM_COMPLETIONS=1
MAX_TOKENS=2048
TEMPERATURE=0.0
TOP_P=1.0

# ========================================
# 颜色输出
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ========================================
# 辅助函数
# ========================================

print_header() {
    echo ""
    echo "============================================================"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo "============================================================"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BOLD}${BLUE}### $1${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 停止推理服务器
stop_inference_server() {
    local pid_file="${SCRIPT_DIR}/logs/inference_server.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "停止推理服务器 (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 2
            rm -f "$pid_file"
            print_success "推理服务器已停止"
        fi
    fi
}

# ========================================
# 主流程开始
# ========================================

print_header "🧪 知识冲突测试实验"

echo "实验配置:"
echo "  基础模型: ${BASE_MODEL_PATH}"
echo "  第一次训练框架: ${FIRST_FRAMEWORK}"
echo "  第二次训练框架: ${SECOND_FRAMEWORK}"
echo "  测试框架: ${TEST_FRAMEWORK}"
echo "  第一次SFT输出: ${FIRST_SFT_MODEL}"
echo "  第二次SFT输出: ${SECOND_SFT_MODEL}"
echo ""
echo "实验流程:"
echo "  1️⃣  准备第一个框架 (${FIRST_FRAMEWORK}) 的训练数据"
echo "  2️⃣  第一次SFT训练 (${BASE_MODEL_PATH} → ${FIRST_FRAMEWORK})"
echo "  3️⃣  准备第二个框架 (${SECOND_FRAMEWORK}) 的训练数据"
echo "  4️⃣  第二次SFT训练 (${FIRST_FRAMEWORK}-SFT → ${SECOND_FRAMEWORK})"
echo "  5️⃣  在测试框架 (${TEST_FRAMEWORK}) 上准备评测数据"
echo "  6️⃣  启动推理服务器 (使用最终模型)"
echo "  7️⃣  批量代码生成"
echo "  8️⃣  批量执行评估"
echo "  9️⃣  聚合评估指标"
echo ""

read -p "按回车键继续，或 Ctrl+C 取消..."

# ========================================
# 阶段1：准备第一个框架的训练数据
# ========================================

print_header "阶段 1/9: 准备 ${FIRST_FRAMEWORK} 框架的训练数据"

print_step "1.1 解析算法核心方法"
cd "${SCRIPT_DIR}"
FRAMEWORK="${FIRST_FRAMEWORK}" bash run_parse_algorithm_methods.sh
print_success "解析完成"

print_step "1.2 构建提示词"
FRAMEWORK="${FIRST_FRAMEWORK}" bash run_prompts_construction.sh
print_success "提示词构建完成"

print_step "1.3 构建训练数据集"
cd "${SCRIPT_DIR}/sft"

# 检查并设置正确的路径
FIRST_REPO_NAME="${FIRST_FRAMEWORK}-main"
FIRST_SOURCE_DIR="${PROJECT_DIR}/${FIRST_FRAMEWORK}/knowledge_corpus/${FIRST_REPO_NAME}"
FIRST_OUTPUT_DIR="${SCRIPT_DIR}/data/${FIRST_FRAMEWORK}"
FIRST_TRAINING_DATA="${FIRST_OUTPUT_DIR}/${FIRST_FRAMEWORK}_training_dataset.jsonl"

if [ ! -d "$FIRST_SOURCE_DIR" ]; then
    print_error "源目录不存在: ${FIRST_SOURCE_DIR}"
    exit 1
fi

mkdir -p "$FIRST_OUTPUT_DIR"

python3 finetune_dataset_builder.py \
    --source-dir "$FIRST_SOURCE_DIR" \
    --output-file "$FIRST_TRAINING_DATA" \
    --format jsonl \
    --max-file-size 1048576

print_success "训练数据集构建完成: ${FIRST_TRAINING_DATA}"

# ========================================
# 阶段2：第一次SFT训练
# ========================================

print_header "阶段 2/9: 第一次 SFT 训练 (${FIRST_FRAMEWORK})"

print_step "2.1 开始训练"
echo "输入模型: ${BASE_MODEL_PATH}"
echo "训练数据: ${FIRST_TRAINING_DATA}"
echo "输出模型: ${FIRST_SFT_MODEL}"
echo ""

mkdir -p "${FIRST_SFT_MODEL}"

cd "${SCRIPT_DIR}/sft"

python3 finetuning.py \
    --model_name_or_path "${BASE_MODEL_PATH}" \
    --dataset_path "${FIRST_TRAINING_DATA}" \
    --output_dir "${FIRST_SFT_MODEL}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --val_split_ratio 0.1 \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type cosine \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 3 \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --use_wandb false \
    --fp16 false \
    --bf16 true \
    --tf32 true \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --remove_unused_columns false \
    --logging_first_step true \
    --report_to none \
    --keep_file_types "python,shell,yaml,markdown" \
    --stride_fraction 0.125 \
    --add_file_path_header "false"

print_success "第一次训练完成: ${FIRST_SFT_MODEL}"

# ========================================
# 阶段3：准备第二个框架的训练数据
# ========================================

print_header "阶段 3/9: 准备 ${SECOND_FRAMEWORK} 框架的训练数据"

print_step "3.1 解析算法核心方法"
cd "${SCRIPT_DIR}"
FRAMEWORK="${SECOND_FRAMEWORK}" bash run_parse_algorithm_methods.sh
print_success "解析完成"

print_step "3.2 构建提示词"
FRAMEWORK="${SECOND_FRAMEWORK}" bash run_prompts_construction.sh
print_success "提示词构建完成"

print_step "3.3 构建训练数据集"
cd "${SCRIPT_DIR}/sft"

# 检查并设置正确的路径
SECOND_REPO_NAME="${SECOND_FRAMEWORK}-main"
SECOND_SOURCE_DIR="${PROJECT_DIR}/${SECOND_FRAMEWORK}/knowledge_corpus/${SECOND_REPO_NAME}"
SECOND_OUTPUT_DIR="${SCRIPT_DIR}/data/${SECOND_FRAMEWORK}"
SECOND_TRAINING_DATA="${SECOND_OUTPUT_DIR}/${SECOND_FRAMEWORK}_training_dataset.jsonl"

if [ ! -d "$SECOND_SOURCE_DIR" ]; then
    print_error "源目录不存在: ${SECOND_SOURCE_DIR}"
    exit 1
fi

mkdir -p "$SECOND_OUTPUT_DIR"

python3 finetune_dataset_builder.py \
    --source-dir "$SECOND_SOURCE_DIR" \
    --output-file "$SECOND_TRAINING_DATA" \
    --format jsonl \
    --max-file-size 1048576

print_success "训练数据集构建完成: ${SECOND_TRAINING_DATA}"

# ========================================
# 阶段4：第二次SFT训练
# ========================================

print_header "阶段 4/9: 第二次 SFT 训练 (${SECOND_FRAMEWORK})"

print_step "4.1 开始训练（基于第一次训练的模型）"
echo "输入模型: ${FIRST_SFT_MODEL}"
echo "训练数据: ${SECOND_TRAINING_DATA}"
echo "输出模型: ${SECOND_SFT_MODEL}"
echo ""

mkdir -p "${SECOND_SFT_MODEL}"

cd "${SCRIPT_DIR}/sft"

python3 finetuning.py \
    --model_name_or_path "${FIRST_SFT_MODEL}" \
    --dataset_path "${SECOND_TRAINING_DATA}" \
    --output_dir "${SECOND_SFT_MODEL}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --val_split_ratio 0.1 \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type cosine \
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_grad_norm 1.0 \
    --optim adamw_torch \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 3 \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --use_wandb false \
    --fp16 false \
    --bf16 true \
    --tf32 true \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --remove_unused_columns false \
    --logging_first_step true \
    --report_to none \
    --keep_file_types "python,shell,yaml,markdown" \
    --stride_fraction 0.125 \
    --add_file_path_header "false"

print_success "第二次训练完成: ${SECOND_SFT_MODEL}"

# ========================================
# 阶段5：准备测试框架的评测数据
# ========================================

print_header "阶段 5/9: 准备 ${TEST_FRAMEWORK} 框架的评测数据"

print_step "5.1 解析算法核心方法"
cd "${SCRIPT_DIR}"
FRAMEWORK="${TEST_FRAMEWORK}" bash run_parse_algorithm_methods.sh
print_success "解析完成"

print_step "5.2 构建提示词"
FRAMEWORK="${TEST_FRAMEWORK}" bash run_prompts_construction.sh
print_success "提示词构建完成"

# ========================================
# 阶段6：启动推理服务器
# ========================================

print_header "阶段 6/9: 启动推理服务器"

# 先停止可能存在的服务器
stop_inference_server

print_step "6.1 启动推理服务器（使用最终训练的模型）"
echo "模型路径: ${SECOND_SFT_MODEL}"
echo "服务器端口: ${SERVER_PORT}"
echo ""

cd "${SCRIPT_DIR}/inference"

# 设置环境变量并启动服务器
MODEL_PATH="${SECOND_SFT_MODEL}" \
SERVER_PORT="${SERVER_PORT}" \
SERVER_HOST="${SERVER_HOST}" \
bash start_inference_server.sh

print_success "推理服务器已启动"

# ========================================
# 阶段7：批量代码生成
# ========================================

print_header "阶段 7/9: 批量代码生成"

print_step "7.1 使用推理服务器生成代码"

# 设置模型名称（用于输出目录）
MODEL_NAME="$(basename ${SECOND_SFT_MODEL})"

cd "${SCRIPT_DIR}/inference"

FRAMEWORK="${TEST_FRAMEWORK}" \
MODEL_NAME="${MODEL_NAME}" \
SERVER_URL="http://localhost:${SERVER_PORT}" \
NUM_COMPLETIONS="${NUM_COMPLETIONS}" \
MAX_TOKENS="${MAX_TOKENS}" \
TEMPERATURE="${TEMPERATURE}" \
TOP_P="${TOP_P}" \
bash run_batch_code_generation_with_server.sh

print_success "代码生成完成"

# ========================================
# 阶段8：批量执行评估
# ========================================

print_header "阶段 8/9: 批量执行代码评估"

print_step "8.1 运行评估脚本"

cd "${SCRIPT_DIR}"

FRAMEWORK="${TEST_FRAMEWORK}" \
MODEL_NAME="${MODEL_NAME}" \
bash run_batch_execution_evaluation_pure.sh

print_success "评估完成"

# ========================================
# 阶段9：聚合评估指标
# ========================================

print_header "阶段 9/9: 聚合评估指标"

print_step "9.1 聚合所有测试实例的指标"

# 获取测试实例列表
TEST_DATA_DIR="${SCRIPT_DIR}/data/${TEST_FRAMEWORK}/${MODEL_NAME}"
TEST_EXAMPLES_LIST=()
while IFS= read -r file; do
    filename=$(basename "$file")
    if [[ $filename =~ algorithm_methods_data_(.+)_output\.jsonl ]]; then
        test_example="${BASH_REMATCH[1]}"
        TEST_EXAMPLES_LIST+=("$test_example")
    fi
done < <(find "$TEST_DATA_DIR" -name "algorithm_methods_data_*_output.jsonl" -type f | sort)

if [ ${#TEST_EXAMPLES_LIST[@]} -eq 0 ]; then
    print_error "未找到任何测试实例"
    exit 1
fi

TEST_EXAMPLES_STR="${TEST_EXAMPLES_LIST[*]}"
echo "测试实例: ${TEST_EXAMPLES_STR}"
echo ""

cd "${SCRIPT_DIR}"

python3 aggregate_metrics.py \
    --model_dir "${TEST_DATA_DIR}" \
    --test_examples ${TEST_EXAMPLES_STR} \
    --framework "${TEST_FRAMEWORK}"

print_success "指标聚合完成"

# ========================================
# 实验完成
# ========================================

print_header "🎉 知识冲突测试实验完成！"

echo "实验总结:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "训练流程:"
echo "  1. 基础模型: ${BASE_MODEL_PATH}"
echo "  2. 第一次训练 (${FIRST_FRAMEWORK}): ${FIRST_SFT_MODEL}"
echo "  3. 第二次训练 (${SECOND_FRAMEWORK}): ${SECOND_SFT_MODEL}"
echo ""
echo "评测信息:"
echo "  测试框架: ${TEST_FRAMEWORK}"
echo "  测试实例: ${TEST_EXAMPLES_STR}"
echo "  结果目录: ${TEST_DATA_DIR}"
echo ""
echo "模型路径:"
echo "  第一次SFT: ${FIRST_SFT_MODEL}"
echo "  第二次SFT: ${SECOND_SFT_MODEL}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 停止推理服务器
print_step "清理：停止推理服务器"
stop_inference_server

print_success "实验完成！"

exit 0
