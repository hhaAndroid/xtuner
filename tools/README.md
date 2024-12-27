# DPO

## 版本

```bash
git clone https://github.com/hhaAndroid/xtuner.git -b dpo
```
如果想启动节省显存版本，需要在命令行指定 `--use-liger` 参数。并且安装如下特定版本库：

```bash
pip install git+https://github.com/hhaAndroid/Liger-Kernel.git@hha
````

如果之前有安装，现在不想覆盖原先的，可以通过指定 pythonpath 实现。在运行前执行


```shell
git clone https://github.com/hhaAndroid/Liger-Kernel.git -b hha
export PYTHONPATH=Liger-Kernel/src/:$PYTHONPATH
```

## 环境安装

```shell
pip install -r tools/requirements.txt
```

- torch 版本推荐用 torch 2.5, 也可以使用 torch 2.4
- flash attention 请自行安装


## 数据准备

只支持 jsonl 格式，每一条格式为：

```json
{
    "chosen_conversations": [
      {
        "from": "human",
        "value": "human instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      },
      {
        "from": "human",
        "value": "human instruction"
      },
      {
          "from": "gpt",
          "value": "chosen answer (required)"
      }],
   "rejected_conversations": [
      {
        "from": "human",
        "value": "human instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }]
}
```

其中 chosen_conversations 和 rejected_conversations 可以是单轮对话也可以是多轮对话，并且对话轮数可以不一样，prompt 和 response 也可以不一样。

## 训练

```shell
cd tools
bash shell/llm_dpo_demo.sh
```
