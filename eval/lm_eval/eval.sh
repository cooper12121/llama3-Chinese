


lm_eval --model hf \
    --model_args pretrained=/apdcephfs_qy3/share_301372554/share_info/qianggao/llama3/output/04-22-2024_13:11:15_llama3_pretrain_wudao_24gpu___bs__maxlen_1024_pad_right_lr_1e-4_format_wudao \
    --tasks ceval-valid \
    --batch_size 8 \
    --output_path /apdcephfs_qy3/share_301372554/share_info/qianggao/llama3/eval/lm_eval \
    --write_out \
    --show_config \


# accelerate launch -m lm_eval --model hf \
#     --tasks lambada_openai,arc_easy \
#     --batch_size 16

# lm_eval --model hf \
#     --tasks lambada_openai,arc_easy \
#     --model_args parallelize=True \
#     --batch_size 16


# lm_eval --model hf \
#     --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
#     --tasks lambada_openai,hellaswag \
#     --device cuda:0 \
#     --batch_size auto:4