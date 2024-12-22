# download dataset：git clone https://oauth2:Cc9Lmwwid1KFiMv5uViT@www.modelscope.cn/datasets/xiaoyaowudi/Minipile.git

from modules.config import *

from modules.model import RWKV_v4
import argparse
import os
import sys
import torch
from utils import BinIdxDataset
import random
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

best_checkpoint_callback = ModelCheckpoint(
    save_top_k = 2,
    monitor="training_loss",
    mode="min",
    filename="RWKV-best-{epoch:02d}-{training_loss:.2f}",
)
reg_checkpoint_callback = ModelCheckpoint(
    every_n_epochs = 100,
    filename="RWKV-reg-{epoch:02d}-{training_loss:.2f}",
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'A simple Implementation of RWKV v4')
    # parser.add_argument('--init', action = 'store_true', help = 'Initialize weights (prior than --weights option)')
    parser.add_argument('-w', '--weights', type = str, help = 'Specify the path of the weights file')
    # parser.add_argument('-o', '--output_weights', default = 'rwkv_v4.pth', type = str, help = 'Specify the path of the output weights file')
    parser.add_argument('-d', '--dataset_idx', default='minipile_train.idx', type = str, help = 'Path to the dataset idx file.')

    args = parser.parse_args()

    # print(args.weights, type(args.weights))

    if args.weights != None:
        if not os.path.isfile(args.weights):
            print(f'[FATAL] Weights file {args.weights} doesn\'t exists. Immediately exit with code 1.')
            sys.exit(1)
    
    dataset = BinIdxDataset(args.dataset_idx, context_len, miniepoch_size)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(20240728)

    cuda_device = torch.device('cuda')

    model = RWKV_v4(init_weights = True, vocab_size = vocab_size, total_layers = total_layers, emb_size = emb_size,
                    time_mixing_weight_std = time_mixing_weight_std, time_mixing_hidden_size = time_mixing_hidden_size,
                    channel_mixing_weight_std = channel_mixing_weight_std, channel_mixing_hidden_size = channel_mixing_hidden_size, embedding_init_value = 1e-4,
                    adam_betas = adam_betas, learning_rate = learning_rate)
    
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = 8)
    
    trainer = L.Trainer(devices = gpu_count, max_epochs = epoch_count, log_every_n_steps = 1, callbacks = [best_checkpoint_callback, reg_checkpoint_callback])
    if args.weights!=None:
        trainer.fit(model = model, train_dataloaders = dataset_loader, ckpt_path = args.weights)
    else:
        trainer.fit(model = model, train_dataloaders= dataset_loader)
    
    # if args.init:

    #     set_seed(20240728)
        
    #     model = RWKV_v4(init_weights = True, vocab_size = vocab_size, total_layers = total_layers, emb_size = emb_size,
    #                     time_mixing_weight_std = time_mixing_weight_std, time_mixing_hidden_size = time_mixing_hidden_size,
    #                     channel_mixing_weight_std = channel_mixing_weight_std, channel_mixing_hidden_size = channel_mixing_hidden_size, embedding_init_value = 1e-4)
    #     # torch.save(model.state_dict(), args.output_weights)
    #     # script_model = torch.jit.script(model)
    #     # torch.jit.save(script_model, args.output_weights)
    
    # else:
    #     if args.weights == None:
    #         print(f'[FATAL] You need to specify a weight file. Immediately exit with code 2.')
    #         sys.exit(2)
    #     model = RWKV_v4(init_weights = False, vocab_size = vocab_size, total_layers = total_layers, emb_size = emb_size,
    #                     time_mixing_weight_std = time_mixing_weight_std, time_mixing_hidden_size = time_mixing_hidden_size,
    #                     channel_mixing_weight_std = channel_mixing_weight_std, channel_mixing_hidden_size = channel_mixing_hidden_size, embedding_init_value = 1e-4)
    #     cuda_device = torch.device('cuda')
    #     model.load_state_dict(torch.load(args.weights, map_location = cuda_device))

    #     optim = torch.optim.Adam(params = model.parameters, betas = adam_betas)

    #     torch.backends.cudnn.allow_tf32 = True
    #     torch.backends.cuda.matmul.allow_tf32 = True


    # model = RWKV_v4(init_weights = True, vocab_size = vocab_size, total_layers = total_layers, emb_size = emb_size,
    #             time_mixing_weight_std = 2, time_mixing_hidden_size = emb_size,
    #             channel_mixing_weight_std = 2, channel_mixing_hidden_size = 4 * emb_size)
