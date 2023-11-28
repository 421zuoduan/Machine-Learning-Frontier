import numpy as np
import torch
import evaluate
import peft
import transformers
import datasets

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


config = datasets.DownloadConfig(resume_download=True, max_retries=100)


model_name_or_path = "roberta-large"
task = "mrpc"
num_epochs = 20
lr = 1e-3
batch_size = 32

dataset = load_dataset(
    "glue", task, cache_dir="./hf_cache", download_config=config)
# print(dataset["train"][0])

metric = evaluate.load("glue", task) # 加载 GLUE 任务的评估脚本


def compute_metrics(eval_pred): # 评估性能
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")): # 根据模型的名字判断填充的方向， 这里的padding是指在序列数据的两端添加特定的元素，使序列达到指定的长度
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, padding_side=padding_side) # 根据给定的 model_name_or_path创建一个 tokenizer 对象
if getattr(tokenizer, "pad_token_id") is None: # 检查 tokenizer 是否有 pad_token_id 属性
    tokenizer.pad_token_id = tokenizer.eos_token_id # 若没有则赋值为 eos_token_id 属性的值


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(
        examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    return outputs


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

peft_config = PromptEncoderConfig(
    task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# "trainable params: 1351938 || all params: 355662082 || trainable%: 0.38011867680626127"

training_args = TrainingArguments(
    output_dir="your-name/roberta-large-peft-p-tuning",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# notebook_login()
# model.push_to_hub("your-name/roberta-large-peft-p-tuning", use_auth_token=True)

peft_model_id = "smangrul/roberta-large-peft-p-tuning"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(
    config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, peft_model_id)

classes = ["not equivalent", "equivalent"]

sentence1 = "Coast redwood trees are the tallest trees on the planet and can grow over 300 feet tall."
sentence2 = "The coast redwood trees, which can attain a height of over 300 feet, are the tallest trees on earth."

inputs = tokenizer(sentence1, sentence2, truncation=True,
                   padding="longest", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits
    print(outputs)


paraphrased_text = torch.softmax(outputs, dim=1).tolist()[0]
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrased_text[i] * 100))}%")