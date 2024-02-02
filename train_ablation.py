import os
import sys
import time
import argparse
import logging
import numpy as np
import random

import torch
from torch import nn

# libraries for visualizing tensor data
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

from utils.data import datasets
from utils.model import models
from utils.evaluate import Evaluator
from utils.loss import myloss
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

# here imma have to lower epoch to get my segmented images, it was set to 80 before

def main(seed=2022, epoches=100): #500
    parser = argparse.ArgumentParser(description='ablation')
    # dataset option
    parser.add_argument('--model_name', type=str, default='mosts', choices=['mosts'], help='model name')
    parser.add_argument('--data_loader', type=str, default='ablation_data_loader', choices=['ablation_data_loader'], help='data_loader name')
    parser.add_argument('--valid_group', type=int, default=3, help='set the valid group index (default: 0)')
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='N', help='input batch size for testing (default: 16)')
    parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers for data loader (default: 16)')
    parser.add_argument('--loss_name', type=str, default='combo', choices=['weighted_bce', 'dice', 'batch_dice', 'focal','combo','combo_batch', 'combo_mix'], help='set the loss function')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    
    # this argument is so that we run the script once to export the model to onnx
    parser.add_argument('--export_onnx', action='store_true', help='Export model as ONNX')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    rng_ = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # Setup data generator
    mydataset_embedding = datasets[args.data_loader]
    data_train = mydataset_embedding(split='train', random_gen = rng_, num_candidates = 5, transform = None, transform_ref = None, valid_group=args.valid_group)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.train_batch_size, num_workers = args.num_workers, pin_memory=True, shuffle=True, prefetch_factor=2, worker_init_fn=worker_init_fn)
    data_val = mydataset_embedding(split='test', random_gen = rng_, num_candidates = 5, transform = None, transform_ref = None, valid_group=args.valid_group)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=args.test_batch_size, num_workers = args.num_workers, pin_memory=True, shuffle=False, prefetch_factor=2)
    
    evaluator = Evaluator(num_class=data_val.split_point+1) # ignore background class

    dir_name = 'log/' + str(args.data_loader) + '_' + str(args.model_name) + '_valid_group_' + str(args.valid_group)
    # ablation_data_loader_mosts_valid_group_3

    if not os.path.exists(dir_name):
        # os.mkdir(dir_name)
        # make directory dir_name, don't overwrite if it exists
        os.makedirs(dir_name, exist_ok=True)
    
    now_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    logging.basicConfig(level=logging.INFO,
                        filename=dir_name + '/output_' + now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('data_loader: %s, model_name: %s, loss_name: %s, batch_size: %s', args.data_loader, args.model_name, args.loss_name, args.train_batch_size)
    logging.info('train with: %s', data_train.train)
    logging.info('test with: %s', data_val.test)

    # Complie model
    model = models[args.model_name]()
    
    # Load pretrained models for training with another dataset
    #   - path for pretrained set (DTD training set)
    
    pre_trained_model_path = 'C:\\Users\\AUVSL\\Documents\\Jay\\MOSTS\\log\\ablation_data_loader_mosts_valid_group_3\\epoch_2023_10_19_04_30_27_texture.pth' #the one we wanna use for UC merced set next
    model.load_state_dict(torch.load(pre_trained_model_path))
    
    # Exporting the model as an ONNX file:
    if args.export_onnx:
        # Settin the model to evaluation mode for export
        model.eval()

        # This is the directory the ONNX files r saved in
        onnx_directory = 'C:\\Users\\AUVSL\\Documents\\Jay\\MOSTS\\ONNX_exports'
        onnx_file_path = os.path.join(onnx_directory, 'model.onnx')
        # Create the directory if it doesn't exist
        if not os.path.exists(onnx_directory):
            os.makedirs(onnx_directory)

        # Temporary tensor inputs for 'image' and 'patch' to match model inputs seen in ablation_data_loader.py
        dummy_image = torch.randn(1, 3, 256, 256)
        dummy_patch = torch.randn(1, 3, 256, 256)

        # Actually exporting the model in ONNX format
        torch.onnx.export(model, (dummy_image, dummy_patch), onnx_file_path, verbose=True, input_names=['image', 'patch'], output_names=['output'])
        # kill the script after exporting
        return 
    
    
    # CUDA init
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    # Setup loss function & optimizer, scheduler
    criterion = myloss[args.loss_name]()
    optim_para = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(optim_para, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(optim_para,lr=args.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # Init loss & IoU
    IoU_final = 0
    epoch_final = 0
    losses = 0
    iteration = 0

    # Start training
    for epoch in range(epoches):
        train_loss = 0
        logging.info('epoch:' + str(epoch))
        start = time.time()
        np.random.seed(epoch)
        random.seed(epoch)
        data_train.curriculum = (epoch+1)/epoches
        data_train.random_gen = np.random.default_rng(epoch)

        for i, data in enumerate(loader_train):
            query, label, reference =  data[0], data[1], data[2]

            iteration += 1
            if torch.cuda.is_available():
                query = query.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                reference = reference.cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(query, reference)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses += loss.item()

            if iteration % 20 == 0:
                run_time = time.time() - start
                start = time.time()
                losses = losses / 20
                logging.info('iter:' + str(iteration) + " time:" + str(run_time) + " train loss = {:02.5f}".format(losses))
                losses = 0
        model_path = dir_name + '/epoch_{epoches}_texture.pth'.format(epoches=now_time)
        
        print("Training progress: ",data_train.curriculum*100,"%")

        # Model evaluation after one epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0
            evaluator.reset()
            np.random.seed(seed+1)
            random.seed(seed+1)
            data_val.curriculum = 1
            data_val.random_gen = np.random.default_rng(seed+1)

            for i, data in enumerate(loader_val):
                query, label, reference, image_class = data[0], data[1], data[2], data[3]

                if torch.cuda.is_available():
                    query = query.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    reference = reference.cuda(non_blocking=True)

                scores = model(query, reference)
                val_loss += criterion(scores, label)
                seg = torch.clone(scores[:, 0, :, :].detach())
                seg[seg >= 0.5] = 1
                seg[seg < 0.5] = 0
 
                # Add batch sample into evaluator

                # converts predicted segment tensor into numpy array
                pred = seg.long().data.cpu().numpy()

                # makes directory result, if it already exists then we don't overwrite
                os.makedirs('results', exist_ok=True)

                # looping through data
                for idx in range(query.shape[0]): 
                    # query, reference, & ground truth labels do not change so we can just use first iteration
                    if epoch == 1:
                        # query image
                        os.makedirs('results/query', exist_ok=True)
                        query_img = query[idx].cpu().numpy().transpose(1, 2, 0) 
                        plt.imsave(f'results/query/query_img_{epoch}_{idx}.png', query_img)

                        # reference image
                        os.makedirs('results/reference',exist_ok=True)
                        ref_img = reference[idx].cpu().numpy().transpose(1, 2, 0) 
                        plt.imsave(f'results/reference/ref_img_{epoch}_{idx}.png', ref_img)

                        # ground truth label
                        os.makedirs('results/truth_label',exist_ok=True)
                        label_img = label[idx].cpu().numpy()
                        plt.imsave(f'results/truth_label/label_img_{epoch}_{idx}.png', label_img, cmap='plasma')

                    # predicted segmentation (already np array)
                    os.makedirs('results/predicted_segment',exist_ok=True)
                    pred_img = pred[idx]
                    plt.imsave(f'results/predicted_segment/pred_img_{epoch}_{idx}.png', pred_img, cmap='viridis')
                
                label = label.cpu().numpy()
                evaluator.add_batch(label, pred, image_class)

            mIoU, mIoU_d = evaluator.Mean_Intersection_over_Union()
            FBIoU = evaluator.FBIoU()

            logging.info("{:10s} {:.3f}".format('IoU_mean', mIoU))
            logging.info("{:10s} {}".format('IoU_mean_detail', mIoU_d))
            logging.info("{:10s} {:.3f}".format('FBIoU', FBIoU))
            if mIoU > IoU_final:
                epoch_final = epoch
                IoU_final = mIoU
                torch.save(model.state_dict(), model_path)
            logging.info('best_epoch:' + str(epoch_final))
            logging.info("{:10s} {:.3f}".format('best_IoU', IoU_final))
        
        model.train()
        scheduler.step()
        logging.info(f"LR: {optimizer.param_groups[0]['lr']}")

    logging.info(epoch_final)
    logging.info(IoU_final)

if __name__ == '__main__':
    main()