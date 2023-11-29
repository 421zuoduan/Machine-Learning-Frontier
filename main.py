import torch.nn
import numpy as np
import random

from utils import set_seed, getLLM, logConfig
from config import *
from trainer import trainer
from model_generation import model_generate
from dataloader import load_dataset


def main(db, args):

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取LLM模型
    LLM_model, tokenizer = getLLM(args.LLM_path)
    LLM_model.to(device)
    LLM_model.eval()
    
    # 数据离散化
    data, label, labelmap = db.discrete(args.slice_num)
    
    # 定义随机状态
    for random_state in args.random_state:

        # 设置种子
        set_seed(random_state)

        # 定义日志文件
        logger = logConfig(args.log_path, args.taskformat, args.add_terminal, args.name, random_state)

        # 获取数据集和prompt数量
        traindata, validdata, train_loader, valid_loader= load_dataset(args, data, label, tokenizer, random_state)

        # 用以生成prompt_model和mask_model
        traindata_dimention = len(traindata.shape)
        traindata_len = traindata.shape[-2] if traindata_dimention == 3 else 1

        # 生成最终的模型
        model = model_generate(args, LLM_model, traindata_dimention, traindata_len, labelmap, device)

        # 训练设置（优化器，损失函数）
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.NAdam(
            model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay)

        # 得到插入Mask的prompt
        beforePrompt, afterPrompt, mask = model.prompt_model.returnPrompt(tokenizer)

        # 写入日志文件
        logger.info("Init Prompt!")
        logger.info(f'beforePrompt:{beforePrompt}')
        logger.info(f'afterPrompt:{afterPrompt}')
        logger.info(f'mask:{mask}')

        # 训练所需变量定义
        best_valid_loss = float('inf')
        not_change = 0
        prompt_iter = 0

        # 开始训练
        for epoch in range(1, args.num_epochs + 1):
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            valid_epoch_loss = 0.0
            valid_correct_predictions = 0
            valid_total_samples = 0
            correct_predictions, epoch_loss, total_samples = trainer(train_loader, model, tokenizer, 
                                                                     correct_predictions, epoch_loss, total_samples, optimizer, criterion, device, train=True)
            
            model.eval()
            valid_correct_predictions, valid_epoch_loss, valid_total_samples = trainer(valid_loader, model, tokenizer, 
                                                                                       valid_correct_predictions, valid_epoch_loss, valid_total_samples, optimizer, criterion, device, train=True)
            
            model.train()

            # 计算epoch平均损失和准确率
            epoch_loss /= len(train_loader)
            accuracy = correct_predictions / total_samples
            valid_epoch_loss /= len(valid_loader)
            valid_accuracy = valid_correct_predictions / valid_total_samples

            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                # 保存模型
                torch.save(model.prompt_model,
                           f'checkpoint/promptModel/{args.name}_{random_state}_{prompt_iter}_romptModel.pt')
                torch.save(model.mask_model, f'checkpoint/maskModel/{args.name}_{random_state}_{prompt_iter}_maskModel.pt')
                logger.info(f'Epoch [{epoch}/{args.num_epochs}], Valid Loss: {valid_epoch_loss:.4f}!')
            else:
                not_change += 1
                if (not_change == args.patience):
                    not_change = 0
                    model.prompt_model.reParameterize()

                    beforePrompt, afterPrompt, mask = model.prompt_model.returnPrompt(tokenizer)
                    logger.info(f"Prompt Iter {prompt_iter}Early Stop! Change Prompt To Find A Better One!")
                    logger.info(f'beforePrompt:{beforePrompt}')
                    logger.info(f'afterPrompt:{afterPrompt}')
                    logger.info(f'mask:{mask}')
                    best_valid_loss = float('inf')
                    prompt_iter += 1
            logger.info(f'Loss Not Changed Num:{not_change}, Prompt Iter:{prompt_iter}')
            logger.info(
                f'Epoch [{epoch}/{args.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}%')





if __name__ == '__main__':
    '''
    choice of datasets: ADNI, ADNI_fMRI, PPMI, OCD_fMRI, FTD_fMRI
    '''

    args = args()
    main(ADNI, args)
    # train(ADNI_fMRI, ADNI_fMRI_config)
