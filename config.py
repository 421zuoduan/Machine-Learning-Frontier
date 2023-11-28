import os
import argparse
import datetime

def str2bool(v):
    #print(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def args:

    start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--start_time', type = str, default = str(start_time), help = 'start time of a certain experiment')
    parser.add_argument('--save_path', type = str, default = './results', help = 'saving path that is a folder')
    parser.add_argument('--save_by_epoch', type = int, default = 5, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--rusume', type = str2bool, default = True, help = 'True for resuming the training')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--gpu_ids', type = str, default = '0', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = str2bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 1000, help = 'number of epochs of training')
    parser.add_argument('--train_batch_size', type = int, default = 15, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0002, help = 'Adam: learning rate for G / D')
    parser.add_argument('--b1', type = float, default = 0.9, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 20, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = './datasets', help = 'datasets baseroot')
    parser.add_argument('--LLM_path', type = str, default = './LLM', help = 'LLM path that is a folder')
    # Logs and checkpoint parameters
    parser.add_argument('--log_path', type = str, default = './logs', help = 'LOG path that is a folder')
    parser.add_Argument('--checkpoint_path', type = str, default = './checkpoints', help = 'CHECKPOINT path that is a folder')
    # Validation parameters
    parser.add_argument('--val_freq', type = int, default = 5, help = 'frequency of validation, 0 for no validation while training')
    parser.add_argument('--val_figures', type = str, default = './results, help = 'validation figures path that is a folder')
    parser.add_argument('--val_batch_size', type = int, default = 1, help = 'size of the batches')
    # Prompt parameters
    parser.add_argument('gru_hidden_state', type = int, default = 64, help = 'gru_hidden_state')
    parser.add_argument('gru_layer', type = int, default = 1, help = 'gru_layer')
    parser.add_argument('prompt_seq_length', type = int, default = 64, help = 'prompt_seq_length')
    parser.add_argument('mask_str', type = str, default = '[MASK]', help = 'mask_str')
    parser.add_argument('slice_num', type = int, default = 100, help = 'slice_num')
    parser.add_argument('random_state', type = int, default = (0, 1, 2, 3, 4), help = 'random_state')
    parser.add_argument('mask_hidden_features', type = int, default = (512, 256), help = 'mask_hidden_features')
    parser.add_argument('cut_hidden_features', type = int, default = (64, 128, 256), help = 'cut_hidden_features')
    # other parameters
    parser.add_argument('--add_terminal', type = str2bool, default = True, help = 'add_terminal')
    parser.add_argument('--patience', type = int, default = 10, help = 'patience')
    parser.add_argument('--seed', type = int, default = 24, help = 'seed')
    parser.add_argument('--dropout', type = float, default = 0.05, help = 'dropout')
    parser.add_argument('--device', type = torch.device, default = torch.device('cuda'), help = 'device')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'batch_size')
    parser.add_argument('--level_token_format', type = str, default = '{:0>2}', help = 'level_token_format')
    parser.add_argument('--label_token_format', type = str, default = '{}', help = 'label_token_format')
    parser.add_argument('--taskformat', type = str, default = '{}_{}', help = 'taskformat')
    parser.add_argument('--slice_num', type = int, default = 100, help = 'slice_num')

    opt = parser.parse_args()
    print(opt)

    return opt