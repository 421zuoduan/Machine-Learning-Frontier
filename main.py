import numpy as np
import torch
import evaluate
import peft
import transformers
import datasets
import matplotlib.pyplot as plt


from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from configs.config import *
from utils import *
from typing import List, Tuple
from Tokenized import Tokenized

if __name__ == '__main__':

    db_cfg = ADNI_config  # 在主函数里换数据集只改这一个参数
    db = database(db_cfg.name, db_cfg.name)
    for random_state in db_cfg.random_state:
        setup_seed(random_state)
        TokenizeObject = Tokenized(
            db_cfg.model_name_or_path, db, db_cfg, random_state)
        tokenized_datasets = TokenizeObject.get_tokenized_dataset()
        data_collator = DataCollatorWithPadding(
            tokenizer=TokenizeObject.tokenizer, padding="longest")
        
# ---------------------------------model definition---------------------------------
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)

        model = AutoModelForSequenceClassification.from_pretrained(
            db_cfg.model_name_or_path, return_dict=True, num_labels=5)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

# --------------------------------train--------------------------------------------
        training_args = TrainingArguments(
            output_dir=db_cfg.model_name + "-" +
            db_cfg.tuning_method + "-" + db_cfg.name,  # 训练结果输出路径
            learning_rate=1e-3,
            per_device_train_batch_size=db_cfg.batch_size,
            per_device_eval_batch_size=db_cfg.batch_size,
            num_train_epochs=db_cfg.num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_steps=10,  # 每多少个训练轮次保存一次模型
            load_best_model_at_end=True,  # 是否保留最好模型
            logging_dir="logs",
            logging_steps=100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=TokenizeObject.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,  # 评估模型的函数
        )

        trainer.train()  # 训练

        # 绘制损失值和准确率曲线
        plot_acc(db_cfg, db_cfg.model_name, db_cfg.num_epochs, accuracies)

        print(trainer.evaluate(tokenized_datasets["test"]))  # 测试
