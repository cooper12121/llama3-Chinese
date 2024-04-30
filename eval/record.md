## llama3的评测代码
    1. 使用开源的评测库进行评测 lm-evaluation-harness: lm_eval folder
    2. 自定义代码评测，可以参考chinese-mixtral的评测代码 
    3. baichuan 也有代码

    # 加载本地数据要在对应的task下的config中更改路劲，然后修改api.task中的download函数