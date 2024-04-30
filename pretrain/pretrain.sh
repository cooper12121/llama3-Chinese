
# # connection config
# # #---------------------------------------------------------------------------------
NET_TYPE="low"
export NCCL_IB_TIMEOUT=24
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi



# DEBUG=false
# # env config
# #---------------------------------------------------------------------------------
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000
NUM_GPUS=$NODE_NUM


WORKSPACE=""
export PYTHONPATH=${WORKSPACE}


MODEL_PATH=""





# DATA_PATH="/apdcephfs_qy3/share_301372554/share_info/hencygao/pycode/Yi-pretrain/data/WuDao/all_8g.json"
# DATA_PATH="/apdcephfs_qy3/share_301372554/share_info/qianggao/dataset/wudao"
DATA_PATH=""

EVAL_PATH="None"

#评测时的输出结果
EVAL_OUTPUT_PATH="None"  


MODEL_OUTPUT_DIR=""


MODEL_TYPE="llama3"
#Yi_8*6b是冷启、Yi_router是热启动
TASK="llama3_pretrain_wudao"
# data config

FORMAT_MODE=wudao

MAX_RESPONSE=1





# training setups
#---------------------------------------------------------------------------------
# BATCH_SIZE=
MICRO_BATCH_SIZE=16
# NUM_GPUS=8
echo $NUM_GPUS
echo $MICRO_BATCH_SIZE
# GRADIENT_ACCUMULATION_STEP=$((BATCH_SIZE / NUM_GPUS / MICRO_BATCH_SIZE))
GRADIENT_ACCUMULATION_STEP=4


MAX_LENGTH=1024


PADDING_SIDE="right"
TRUNCATION_SIDE="left"
POOLING_TYPE="last"

EPOCH=1
LEARNING_RATE=2e-4




# deepspeed setups
#---------------------------------------------------------------------------------
# DS_ZERO=3
# if [[ $DS_ZERO = 2 ]]; then
#     DEEPSPEED=${WORKSPACE}/configs/default_zero2_config.json
# else
#     DEEPSPEED=${WORKSPACE}/configs/default_offload_opt_param.json
# fi

TMP_DIR=${WORKSPACE}/tmp
mkdir -p $TMP_DIR



echo $NODE_IP_LIST > ${TMP_DIR}/env.txt
 
# generate hostfile and pssh.hosts
sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts

DEEPSPEED=${WORKSPACE}/config/configs/ds_config_zero3.json


# output config
#----------------------------------------------------------------------------------
CURRENT_TIME=$(date +'%m-%d-%Y_%H:%M:%S')
EXPERIMENT_NAME=${CURRENT_TIME}_new_begin_${TASK}_${MODEL_NAME}_${DATA_NAME}_bs_${BATCH_SIZE}_maxlen_${MAX_LENGTH}_pad_${PADDING_SIDE}_lr_${LEARNING_RATE}_format_${FORMAT_MODE}

OUTPUT_DIR=${MODEL_OUTPUT_DIR}/${EXPERIMENT_NAME}
LOGS_PATH=${OUTPUT_DIR}/logs

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH


echo "begin experiment ${EXPERIMENT_NAME}"
                    
export CMD="deepspeed   --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} pretrain.py \
    --model_name_or_path $MODEL_PATH \
    --train_files  
     ../dataset/wudao/part-202101281a.json \
    ../dataset/wudao/part-202101281b.json \
    ../wudao/part-2021009337.json \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --eval_output_path $EVAL_OUTPUT_PATH \
    --output_dir $OUTPUT_DIR\
    --do_train True \
    --do_eval False \
    --num_train_epochs 1 \
    --max_steps -1 \
    --model_max_length $MAX_LENGTH \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEP} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.2 \
    --save_total_limit 15 \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --streaming False \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed "configs/ds_config_zero3.json" \
    --bf16 True \
    --use_lora False"

#接着训练就不用预热学习率了

echo $CMD
eval ${CMD} 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt
set +x
