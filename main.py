from utils import data_to_dict
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

from configs.config import *
from utils import *
from typing import List, Tuple

if __name__ == '__main__':

    db=database('ADNI', 'ADNI')
    db_cfg = ADNI_config
    random_state = 0
    # setup_seed(db_cfg.seed)
    data, label, labelmap = db.discrete(db_cfg.slice_num)
    dataset = data_to_dict(data, label, random_state)
    keys = featuers_without_label(dataset)
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=keys,
    )
    tokenizer=get_tokenizer(model_name_or_path=model_name_or_path) # 用于 DataCollator 和 Trainer 的 tokenizer
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    # print(tokenized_datasets['train'])
    # -------------------train-----------------------
    # model_name_or_path = "roberta-large"
    peft_config = PromptEncoderConfig(
    task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    
    # ---------------------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, return_dict=True, num_labels=5)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # "trainable params: 1351938 || all params: 355662082 || trainable%: 0.38011867680626127"

    training_args = TrainingArguments(
        output_dir="roberta-large-peft-p-tuning",
        learning_rate=1e-3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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
        # compute_metrics=compute_metrics,
    )

    trainer.train() # 训练

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