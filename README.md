# llama3-Chinese
* 对llama3进行中文全参预训练，区别于其他使用lora预训练的项目。预计一星期后发布。
* 基于wudao200G中文语料进行,后续再扩展到其他语料
  
[**🇨🇳中文**](./README.md) | [**🌐English**](./README_EN.md) 

<p align="center">
    <br>
    <img src="./figures/llama3.jpg" width="800"/>
    <br>
</p>
<!-- <p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/cooper12121/llama3-Chinese.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/cooper12121/llama3-Chinese">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/cooper12121/llama3-Chinese">
    <a href="https://app.codacy.com/gh/cooper12121/llama3-Chinese/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/142d688425494644b5b156068f55370d"/></a>
</p> -->

本项目基于Meta发布的[llama3-8B-base模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B)进行开发。本项目利用大规模中文无标注数据[WuDaoCorporaText](https://data.baai.ac.cn/details/WuDaoCorporaText)进行了中文增量全参预训练，得到了**llama3-chinese-8B-base**基础模型，后续会进一步通过指令精调，得到了**llama3-chinese-chat**指令模型。



#### 本项目主要内容

- 🚀 开源中文llama3基础模型，该模型在[llama3-8B-base模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B)的基础上进行了中文增量全参预训练（放开所有参数，并不只是针对embedding和l-head层）
- 🚀 开源pt，sft脚本，用户可根据需要进一步训练或微调模型
- 🚀 后续会继续针对llama3-8B-chat版本进行中文指令微调

----

## 新闻

**[2024/04/26] 发布20G（300万条）语料训练的checkpoint及基准测试结果，200G训练还未完成，持续更新中**

**[2024/04/19] 🚀 创建仓库，正式开始预训练**


## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |

| [⏬模型下载](#模型下载)        | llama3-chinese大模型下载地址    |
| [💯模型效果](#模型效果) | 介绍了模型在部分任务上的效果    |
| [📝训练与精调](#训练与精调) | 介绍了如何训练和精调中文llama3-chinese大模型 |
| [❓常见问题](#常见问题) | 一些常见问题的回复 |


## 模型下载

### 模型选择指引

以下是本项目的模型对比以及建议使用场景。**如需聊天交互，请选择Instruct版。**

| 对比项                | llama3-chinese-8B-base                                     | llama3-chinese-8B-chat                                  |
| :-------------------- | :----------------------------------------------------: | :----------------------------------------------------------: |
| 模型类型 | **基座模型** | **指令/Chat模型（类ChatGPT）** |
| 模型大小 |    8B                          |            8B |
| 训练类型     | Causal-LM (CLM)           | 指令精调                                                     |
| 训练方式 | 全参数                         | 全参数 |
| 基于什么模型训练 | meta/llama3-8B-base | meta/llama3-8B-chat |
| 训练语料 | WuDaoCorporaText | 指令数据 |
| 词表大小 | 原版词表，127999 | 原版词表， 127999 |



### 下载地址

| 模型名称                  |   类型   |                    规格                    |                    完整版 GB）                    |
| :------------------------ | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | 
| llama3-chinese-8B-base | 基座模型 | 8B | [[🤗HF]](https://huggingface.co/gao-NLP/llama3-chinese-8B-base) |
| llama3-chinese-8B-chat | 指令模型 | 8B |[[🤗HF]](https://huggingface.co/gao-NLP/llama3-chinese-8B-chat) | 




## 模型效果

为了评测相关模型的效果，本项目分别进行了生成效果评测和客观效果评测（NLU类），从不同角度对大模型进行评估。推荐用户在自己关注的任务上进行测试，选择适配相关任务的模型。


### 客观效果评测

#### C-Eval

[C-Eval](https://cevalbenchmark.com)是一个全面的中文基础模型评估套件，其中验证集和测试集分别包含1.3K和12.3K个选择题，涵盖52个学科。

| Models             | 类型 | Valid (0-shot) | Valid (5-shot) | Test (0-shot) | Test (5-shot) |
| ------------------------ | :------------: | :------------: | :-----------: | :-----------: | :-----------: |
| **meta/llama3-8B-base** | 基座 | 48.01 | 50.15 |  |  |
| **llama3-chinese-8B-base**  | 基座 |  |  | |  |



#### CMMLU



#### MMLU



#### LongBench


### 量化效果评测



## 训练与精调

### 预训练


### 指令精调



## 常见问题
**1. triu_tril_cuda_template" not implemented for 'BFloat16**

  这是torch版本的问题，在torch 2.1.0之后的版本已经修复
，对于torch 2.1.0之前的版本，目前有三种解决方案
* 方法1：在modeling_llama.py line 1095  
  将```causal_mask = torch.triu(causal_mask, diagonal=1)```  
  修改为：
  ```
  causal_mask = causal_mask.to(torch.float32)#
  causal_mask = torch.triu(causal_mask, diagonal=1)
  causal_mask = causal_mask.to('cuda', dtype=torch.bfloat16)#
  ```
* 方法2：在modeling_llama.py line 1094行前添加：  
  ```self.register_buffer("triu0",torch.ones(sequence_length, target_length).to("cuda").triu())```  
  将line 1095 ```causal_mask = torch.triu(causal_mask, diagonal=1)```  
  修改为：```causal_mask=causal_mask*self.triu0```
* 方法3：在加载模型前的代码中添加
  ```torch.set_default_tensor_type(torch.cuda.HalfTensor)```
  但这种方式可能引起cuda内核的pin_memory错误，可行与否与具体的环境有关



