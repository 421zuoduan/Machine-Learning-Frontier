from typing import List, Tuple
from utils import *
from configs.config import *
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
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

    db_cfg = ADNI_config  # 在主函数里换数据集只改这一个参数
    db = database(db_cfg.name, db_cfg.name)
    random_state = 3
    # setup_seed(db_cfg.seed)
    data, label, labelmap = db.discrete(db_cfg.slice_num)
    dataset = data_to_dict(data, label, random_state)
    keys = featuers_without_label(dataset)
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=keys,
    )

    # # 删去所有标签为1的元素
    # for i in ['train', 'test', 'validation']:
    #     for j in [1]:
    #         print(len(tokenized_datasets[i]))
    #         tokenized_datasets[i] = tokenized_datasets[i].filter(
    #             lambda example: example['label'] != [j])
    #         print(len(tokenized_datasets[i]))

    # 用于 DataCollator 和 Trainer 的 tokenizer
    tokenizer = get_tokenizer(model_name_or_path=model_name_or_path)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest")

    count_labels(tokenized_datasets['test']['labels'])
    # 重命名标签
    rename_labels([[1, 2, 3], [0], [4]], tokenized_datasets)
    # 计数
    count_labels(tokenized_datasets['test']['labels'])
    # -------------------train-----------------------
    # model_name_or_path = "roberta-large"
    peft_config = PromptEncoderConfig(
        task_type="SEQ_CLS",
        num_virtual_tokens=20,
        encoder_hidden_size=128)

    peft_config1 = PromptTuningConfig(
        task_type="SEQ_CLS",
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text="Classify if the patience is ill or not:",
        num_virtual_tokens=16,
        tokenizer_name_or_path=model_name_or_path)

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
        tokenizer=get_tokenizer(model_name_or_path=model_name_or_path),
        data_collator=data_collator,
        compute_metrics=compute_metrics,  # 评估模型的函数
    )

    trainer.train()  # 训练

    # 绘制损失值和准确率曲线
    x = np.arange(num_epochs) + 1

    plt.figure(figsize=(10, 5))
    plt.plot(x, accuracies, "r.-")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(db_cfg.name + " with " + model_name + " Accuracy")

    plt.show()

    # print(trainer.evaluate(tokenized_datasets["validation"]))
    print(trainer.evaluate(tokenized_datasets["test"]))  # 测试
# print(trainer.evaluate(tokenized_datasets["validation"]))
# print(trainer.evaluate(tokenized_datasets["test"]))
# # notebook_login()
# # model.push_to_hub("your-name/roberta-large-peft-p-tuning", use_auth_token=True)

# # -------------------inference-----------------------
# peft_model_id = "smangrul/roberta-large-peft-p-tuning"
# config = PeftConfig.from_pretrained(peft_model_id)
# inference_model = AutoModelForSequenceClassification.from_pretrained(
#     config.base_model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# model = PeftModel.from_pretrained(inference_model, peft_model_id)

# # classes = ["not equivalent", "equivalent"]

# # sentence1 = "Coast redwood trees are the tallest trees on the planet and can grow over 300 feet tall."
# # sentence2 = "The coast redwood trees, which can attain a height of over 300 feet, are the tallest trees on earth."

# # inputs = tokenizer(sentence1, sentence2, truncation=True,
# #                 padding="longest", return_tensors="pt")

# # with torch.no_grad():
# #     outputs = model(**inputs).logits
# #     print(outputs)

# # paraphrased_text = torch.softmax(outputs, dim=1).tolist()[0]
# # for i in range(len(classes)):
# #     print(f"{classes[i]}: {int(round(paraphrased_text[i] * 100))}%")
