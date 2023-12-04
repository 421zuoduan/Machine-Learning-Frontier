from typing import List, Tuple
from utils import *
from configs.config import *
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
    IA3Config,
    PromptTuningInit,
)
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch
import evaluate
import peft
import transformers
import datasets
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    db_cfg = ADNI_fMRI_config  # 在主函数里换数据集只改这一个参数
    db = database(db_cfg.name, db_cfg.name)
    random_state = 0
    # setup_seed(db_cfg.seed)
    if 'fMRI' in db_cfg.name:
        db.mean_it()
    data, label, labelmap = db.discrete(db_cfg.slice_num)
    dataset = data_to_dict(data, label, random_state)
    keys = featuers_without_label(dataset)
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=keys,
    )

    # 用于 DataCollator 和 Trainer 的 tokenizer
    tokenizer = get_tokenizer(model_name_or_path=model_name_or_path)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest")

    # count_labels(tokenized_datasets['test']['labels'])
    # # 重命名标签
    # rename_labels([[1, 2, 3], [0], [4]], tokenized_datasets)
    # # 计数
    # count_labels(tokenized_datasets['test']['labels'])

    # -------------------train-----------------------
    # model_name_or_path = "roberta-large"
    peft_config = PromptEncoderConfig(
        task_type="SEQ_CLS",
        num_virtual_tokens=20,
        encoder_hidden_size=128)

    peft_config1 = PromptTuningConfig(
        task_type="SEQ_CLS",
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="Classify if the patience is under Alzheimer's Disease, Mild Cognitive Impairment or progressive, or Normal Control by the clinical data:",
        num_virtual_tokens=16,
        tokenizer_name_or_path=model_name_or_path)

    peft_config2 = IA3Config(
        task_type="SEQ_CLS", target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"])

    # ---------------------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, return_dict=True, num_labels=5)
    model = get_peft_model(model, peft_config1)
    model.print_trainable_parameters()
    # "trainable params: 1351938 || all params: 355662082 || trainable%: 0.38011867680626127"

    training_args = TrainingArguments(
        output_dir=model_name + "-" + tuning_method + "-" + db_cfg.name,  # 训练结果输出路径
        learning_rate=1e-2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy='no',
        save_strategy='no',
        save_steps=10,  # 每多少个训练轮次保存一次模型
        load_best_model_at_end=True,  # 是否保留最好模型
        logging_dir="logs",
        logging_strategy="epoch",
    )

    eval_losses = []
    losses = []

    class MyCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.log_history:
                if "loss" in state.log_history[-1]:
                    losses.append(state.log_history[-1]['loss'])
                    print(f"-----------losses before{state.epoch} :{losses}")
                elif "eval_loss" in state.log_history[-1]:
                    eval_losses.append(state.log_history[-1]['eval_loss'])
                    print(
                        f"-----------eval_losses before{state.epoch} :{eval_losses}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=get_tokenizer(model_name_or_path=model_name_or_path),
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # 评估模型的函数
        callbacks=[MyCallback()]
    )

    trainer.train()  # 训练

    # # 绘制损失值和准确率曲线
    # x = np.arange(num_epochs) + 1

    # plt.figure(figsize=(10, 5))
    # plt.plot(x, accuracies, "r.-")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title(db_cfg.name + " with " + model_name + " Accuracy")

    # 创建一个图形窗口和子图
    fig, ax = plt.subplots()

    # # 绘制eval_losses的折线图
    # ax.plot(eval_losses, label='Eval Loss')

    # 绘制losses的折线图
    ax.plot(losses, label='Loss')

    # 添加标题和图例
    ax.set_title(db_cfg.name + " with " + model_name + " Loss")
    ax.legend()
    plt.show()

    # print(trainer.evaluate(tokenized_datasets["validation"]))
    print(trainer.evaluate(tokenized_datasets["test"]))  # 测试
