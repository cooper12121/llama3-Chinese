
import copy
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import logging
import os,sys

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig,DataCollatorForLanguageModeling,default_data_collator
from datasets import load_dataset
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Optional,List

# from torch.utils.data import IterableDataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper

# from metrics import build_compute_metrics_fn

# from model.modeling_mixtral import MixtralForCausalLM

from datasets import load_dataset

sys.path.append("/apdcephfs_qy3/share_301372554/share_info/qianggao/")
from modeling_file.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100


pretrain_features = datasets.Features({
    'id': datasets.Value('int32'),
    'uniqueKey': datasets.Value('string'),
    'titleUkey': datasets.Value('string'),
    'dataType': datasets.Value('string'),
    'title': datasets.Value('string'),
    'content': datasets.Value('string')
})

def _make_r_io_base(f, mode: str):
    import io
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jsonline_load(f, mode="r"):
    """Load a .jsonl file into a dictionary."""
    f = _make_r_io_base(f, mode)
    json_objects_list = [json.loads(line) for line in f]
    return json_objects_list

def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

        

def print_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

@dataclass
class ModelArguments:
    lora_trainable : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default="embed_tokens,lm_head")
    use_lora : Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-moe-16b")
    attn_implementation : Optional[str] = field(default="flash_attention_2")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

@dataclass
class DataArguments:
    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "The input training data file (a text file)."}),
    data_path: str = field(default=None, metadata={"help": "Path to the training data."}),
    eval_path: str = field(default=None, metadata={"help": "Path to the eval data."}),
    eval_output_path: str = field(default=None, metadata={"help": "Path to the eval data."}),
    streaming: bool = field(default=False, metadata={"help": "Path to the eval data."}),


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int= field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    ),
    
    do_train: Optional[bool] = field(default=True),
    do_eval: Optional[bool] = field(default=True),
    # save_safetensors: Optional[bool] = field(
    #         default=False,
    #         metadata={
    #             "help": "Use safetensors saving and loading for state dicts instead of default torch.load and torch.save."
    #         },
    # )#没用，还是存了好多份很大，

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            # return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        len(tokenized.input_ids) for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    #pretrain只需要label[1:],不包括begin符号即可
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        # label[0]=IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def preprocess_pretrain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""

    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            "<|begin_of_text|>"+text+"<|end_of_text|>",#添加bos和eos
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in sources['content']
    ]
    input_ids  = labels= [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    # labels = copy.deepcopy(input_ids)

    #pretrain只需要label[1:],不包括begin符号即可, 不用置0，shift logit直接会忽略label[0]
    # for label in labels:
    #     label[0] = IGNORE_INDEX#只有开头的起始符号不用算loss
    # print(input_ids[0],labels[0])
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


    
def build_model(model_args, training_args, checkpoint_dir):
    if not model_args.use_lora: assert model_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if training_args.bf16 else torch.float16)

    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # if torch.cuda.is_bf16_supported():
    #         print("support")
    #         torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    #         # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # else:
    #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    from llama3.modeling_llama import LlamaForCausalLM
    # from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
    # model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=model_args.bits == 4,
        #     load_in_8bit=model_args.bits == 8,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=compute_dtype,
        #     bnb_4bit_use_double_quant=model_args.double_quant,
        #     bnb_4bit_quant_type=model_args.quant_type,
        # ) if model_args.use_lora else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        use_cache=False,
    )

    if compute_dtype == torch.float16 and model_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('='*80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('='*80)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32
    # Tokenizer
    
    if model_args.use_lora and model_args.bits < 16:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if model_args.use_lora:
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        else:
            logger.info(f'Init LoRA modules...')
            target_modules = model_args.lora_trainable.split(',')
            modules_to_save = model_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            lora_rank = model_args.lora_rank
            lora_dropout = model_args.lora_dropout
            lora_alpha = model_args.lora_alpha
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save)
            

            model = get_peft_model(model, peft_config)

    try:
        model.print_trainable_parameters()
    except Exception:
        print_trainable_parameters(model)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    
    return model

def compute_metrics_(prediction):
    logits = prediction.predictions
    score = prediction.label_ids
    # print(f">>> logits {logits}")
    # print(f">>> score {score}")

    pred_compare = (logits.unsqueeze(1) > logits.unsqueeze(2)) * 1.
    label_compare = (score.unsqueeze(1) > score.unsqueeze(2)) * 1.
    correct = torch.triu(pred_compare == label_compare, diagonal=1).sum()
    accuracy = correct / (pred_compare.shape[1] * (pred_compare.shape[1] -1) / 2.)
    return {"Preference Acc": accuracy}

from itertools import chain
block_size=1024
def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))       
        logger.info("group texts input examples length%d after_group size%d"%(len(examples['input_ids']),len(result["input_ids"])))
        result["labels"] = result["input_ids"].copy()
        return result

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank == 0:
        logger.info('='*100)
        logger.info(training_args)
    
    # from llama.tokenization_llama import LlamaTokenizer
    # from modeling_file.qwen2_moe.configuration_qwen2_moe i

    tokenizer = transformers.AutoTokenizer.from_pretrained(
    # tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        cache_dir="/apdcephfs_qy3/share_301372554/share_info/qianggao/dataset/wudao/cache"
    )

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)


    if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
         tokenizer.pad_token_id = tokenizer.eos_token_id

        

    if training_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(model_args.model_name_or_path))
    
   
    file_pattern = "*.json"
    raw_train_datasets = load_dataset(
        'json',
        # data_dir=data_args.data_path,
        data_files=data_args.train_files,
        # data_files=data_args.data_path,
        features=pretrain_features,
        split="train",
        streaming=data_args.streaming,
        cache_dir="/apdcephfs_qy3/share_301372554/share_info/qianggao/dataset/wudao/cache",
        # cache_dir=None
        )
    # raw_eval_datasets = load_dataset(
    #     'json',
    #     data_files=data_args.eval_data_path,
    #     split="train",
    #     cache_dir=training_args.cache_dir
    # )
    if data_args.streaming:
        raw_train_datasets = raw_train_datasets.shuffle(seed=1024, buffer_size=1000000)

    if training_args.local_rank > 0: 
        torch.distributed.barrier()
    
    if not data_args.streaming:
        train_dataset = raw_train_datasets.map(
            preprocess_pretrain,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True, # not args.overwrite_cache
            desc="Running Encoding",
            fn_kwargs={ "tokenizer": tokenizer }
        
        )
    else:
        train_dataset = raw_train_datasets.map(
            preprocess_pretrain,
            batched=True,
            batch_size=3000,
            remove_columns=raw_train_datasets.column_names,
            fn_kwargs={ "tokenizer": tokenizer }
        )

    
    lm_datasets = train_dataset.map(
                group_texts,
                batched=True,
                batch_size = 60000,
            )

    # eval_dataset = raw_eval_datasets.map(
    #     train_tokenize_function,
    #     batched=True,
    #     batch_size=3000,
    #     num_proc=32,
    #     remove_columns=raw_train_datasets.column_names,
    #     load_from_cache_file=True, # not args.overwrite_cache
    #     desc="Running Encoding",
    #     fn_kwargs={ "tokenizer": tokenizer }
    # )
    
    


    if training_args.local_rank == 0:
        torch.distributed.barrier()
    
    # print_rank_0(f"Training dataset samples:{len(lm_datasets)}")
    # for index in random.sample(range(len(train_dataset)), 3):
    #     print_rank_0(f"Sample {index} of the training set: {train_dataset[index]['input_ids']}, {train_dataset[index]['labels']}.")
    #     print_rank_0(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # data_module = dict(train_dataset=lm_datasets, eval_dataset=None, data_collator=data_collator)
    # data_module = dict(train_dataset=lm_datasets, eval_dataset=None, data_collator=default_data_collator)

    resume_from_checkpoint_dir = get_last_checkpoint(training_args.output_dir)
    
    print_rank_0(f"\n\n****resulme from :{resume_from_checkpoint_dir}****\n\n")
    model = build_model(model_args, training_args, resume_from_checkpoint_dir)

    if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # print("*\n"*60,type(training_args.max_steps),type(training_args.num_train_epochs))
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=compute_metrics_,**data_module)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=compute_metrics_,
                    train_dataset=lm_datasets,data_collator=default_data_collator)
    if model_args.use_lora:
        trainer.add_callback(SavePeftModelCallback)

    
    if training_args.do_train:
        logger.info("*** Training ***")
        trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
        trainer.save_state()
        # if not model_args.use_lora:
        #     safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        
        trainer.save_model(output_dir=training_args.output_dir)
        # torch.save(trainer.model,training_args.output_dir)
        tokenizer.save_pretrained(save_directory=training_args.output_dir)
    
        # Evaluation
    # if training_args.do_eval:
    #     model.eval()
    #     logger.info("*** Evaluate ***")

    #     compute_metrics = (
    #         build_compute_metrics_fn(
    #             tokenizer = tokenizer,
    #             output_dir = training_args.output_dir,
    #             cache_dir = training_args.cache_dir,
    #             extract_template = "default",
    #             split = "eval",
    #         )
    #     )
    #     trainer.compute_metrics = compute_metrics
    #     # trainer.model = trainer.model.from_pretrained(resume_from_checkpoint_dir) #不训练直接评测才需要加载模型
    #     # 也可以用predict查看具体的生成内容
    #     # metrics = trainer.evaluate()


        
    #     output = trainer.predict(test_dataset=eval_dataset)
    #     metrics = output.metrics


    #     pred_str = tokenizer.batch_decode(output.predictions)
    #     result = []
    #     for i in range(len(raw_eval_datasets)):
    #         result.append(
    #             {
    #                 "content":raw_eval_datasets[i]['content'],
    #                 "summary":raw_eval_datasets[i]['summary'],
    #                 "pred":pred_str[i]['pred'],
    #             }
    #         )
    #     with open(f"/apdcephfs_qy3/share_301372554/share_info/qianggao/Yi/eval_router_warmboot/n/{trainer.state.epoch}_result.json",'w',encoding='utf-8')as f:
    #         json.dump(result,f,ensure_ascii=False)

    #     """ import math
    #     try:
    #         perplexity = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         perplexity = float("inf")
    #     metrics["perplexity"] = perplexity """

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    train()
