# from utils import data_to_dict
# import numpy as np
# import torch
# import evaluate
# import peft
# import transformers
# import datasets


# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from peft import PeftModel, PeftConfig
# from datasets import load_dataset
# from peft import (
#     get_peft_config,
#     get_peft_model,
#     get_peft_model_state_dict,
#     set_peft_model_state_dict,
#     PeftType,
#     PromptEncoderConfig,
# )
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     DataCollatorWithPadding,
#     TrainingArguments,
#     Trainer,
# )

# from configs.config import *
# from utils import *
# from typing import List, Tuple
# from Tokenized import Tokenized

# # --------------------------inference-----------------------

# peft_model_id = "roberta-large-peft-p-tuning/checkpoint-360"
# config = PeftConfig.from_pretrained(peft_model_id)
# inference_model = AutoModelForSequenceClassification.from_pretrained(
#     config.base_model_name_or_path,num_labels=5)
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# model = PeftModel.from_pretrained(inference_model, peft_model_id)

# db_cfg = ADNI_config  # 在主函数里换数据集只改这一个参数
# db = database(db_cfg.name, db_cfg.name)

# _, classes, _ = db.discrete(db_cfg.slice_num)

# random_state = 0
# TokenizeObject = Tokenized(
#     model_name_or_path, db, db_cfg, random_state)
# tokenized_datasets = TokenizeObject.get_tokenized_dataset()
# # with torch.no_grad():
# #     for test_data in tokenized_datasets['test']:
# #         test_data['input_ids']=torch.tensor(test_data['input_ids'])
# #         test_data['attention_mask']=torch.tensor(test_data['attention_mask'])
# #         outputs = model(**test_data).logits
# #         print(outputs)
# #     # outputs = model(**).logits
# #     # print(outputs)


# prob = torch.softmax(outputs, dim=1).tolist()[0]
# for i in range(len(classes)):
#     print(f"{classes[i]}: {prob[i] * 100}%")
