import torch
from transformers import AutoModel, PreTrainedModel
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Tuple
from colorlog import ColoredFormatter

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


############################################################################
############################### 数据编码与解码 ####################################
def encode(args, data, tokenizer):
    shape = data.shape
    if (len(shape) == 2):
        encode_tensor_list = []
        for i in range(shape[0]):
            encode_item_list = []
            for j in range(shape[1]):
                encode_str = args.level_token_format.format(data[i, j])
                encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                encode_item_list.append(encode_item)
            # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
            encode_tensor_list.append(encode_item_list)
        return torch.tensor(encode_tensor_list, dtype=torch.long)
    elif (len(shape) == 3):
        encode_tensor_tensor_list = []
        for i in range(shape[0]):
            encode_tensor_list = []
            for j in range(shape[1]):
                encode_item_list = []
                for k in range(shape[2]):
                    encode_str = args.level_token_format.format(data[i, j, k])
                    encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                    encode_item_list.append(encode_item)
                # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
                encode_tensor_list.append(encode_item_list)
            encode_tensor_tensor_list.append(encode_tensor_list)
        return torch.tensor(encode_tensor_tensor_list, dtype=torch.long)
    else:
        raise NotImplementedError("len(shape)!=2 or len(shape)!=3 Not Implement!")





############################################################################
############################### 获取LLM ####################################
def getLLM(path) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    获取预训练语言模型和分词器。

    Returns:
        Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        包含加载的语言模型和分词器的元组。
    """
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    return model, tokenizer


############################################################################
############################# 日志配置 #####################################
def logConfig(path_format: str, task_format, add_terminal: False, *data):
    formatter = ColoredFormatter(
        "%(white)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'white',
            'WARNING': 'white',
            'ERROR': 'white',
            'CRITICAL': 'white,bg_red',
        },
        secondary_log_colors={},
        style='%'
    )
    task_name = task_format.format(*data)
    logger = logging.getLogger(f'{task_name}_logger')
    logger.setLevel(logging.DEBUG)

    log_filename = os.path.join(path_format, f'{task_name}.log')
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    if (add_terminal):
        # 创建一个用于在控制台输出的处理程序
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


############################################################################
############################# prompt operation #############################

def prompt_reflect(prompt, device):
    max_vals, _ = torch.max(prompt, dim=1, keepdim=True)
    min_vals, _ = torch.min(prompt, dim=1, keepdim=True)
    prompt_slice = max_vals - min_vals
    prompt_slice[prompt_slice == 0] = 1e-18
    prompt = (prompt - min_vals) / prompt_slice
    return prompt


def mask_slice_reflect(mask_slice, prompt_length, device):
    mask, slice = mask_slice[:, 0], mask_slice[:, 1]
    mask, slice = torch.abs_(mask) * prompt_length, torch.abs_(slice) * (prompt_length + 1)
    mask, slice = mask.long().to(device), slice.long().to(device)
    mask, slice = mask % (prompt_length), slice % (prompt_length + 1)
    return mask.long(), slice.long()


def concate_prompt_data(prompt, data, mask, slice, tokenizer: PreTrainedTokenizer, device):
    batch_size = data.shape[0]
    data_length = data.shape[-1]
    masktokenid = tokenizer.mask_token_id
    clstokenid = tokenizer.cls_token_id
    septokenid = tokenizer.sep_token_id

    mask_tensor = torch.tensor([masktokenid], dtype=torch.long, device=device)
    cls_tensor = torch.tensor([clstokenid], dtype=torch.long, device=device)
    sep_tensor = torch.tensor([septokenid], dtype=torch.long, device=device)

    prompt = prompt.view(-1)
    mask_pos = mask.item()
    slice_pos = slice.item()
    prompt = torch.cat(
        (cls_tensor, prompt[:mask_pos], mask_tensor, prompt[mask_pos:], sep_tensor)
    )

    before_prompt = prompt[:slice_pos + 1]
    after_prompt = prompt[slice_pos + 1:]

    before_prompt = torch.stack([before_prompt] * batch_size, dim=0)
    after_prompt = torch.stack([after_prompt] * batch_size, dim=0)

    mask = mask + data_length + 1 if mask_pos >= slice_pos else mask + 1

    concat_data = torch.cat(
        (before_prompt, data, after_prompt), dim=1
    )
    return concat_data, mask


