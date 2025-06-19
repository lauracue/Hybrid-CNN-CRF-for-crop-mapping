#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Elena Cué La Rosa

"""

import argparse
import os
import time
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from src.logger import create_logger
import scipy.io
from src.utils import (
    read_tiff,
    check_folder,
    add_padding
)
from src.dataloader import DatasetFromCoord
from src.network import DeepLabv3_plus

logger = getLogger()


parser = argparse.ArgumentParser(description="Evaluate model")

#########################
#### data parameters ####
#########################
parser.add_argument('--data_dir', default='./stack_images',
                        help="Path containing the raster and labels in numpy format")
parser.add_argument("--mask_path", type=str, default='C:/PostDoc/Thesis_project/Tesis_All_files/Campo_Verde/New_Masks/TrainTestMasks/TrainTestMask_50_50_Dec.tif', 
                        help="path to dataset mask")
parser.add_argument('--trans_path', default='trans_stack_orig_label.mat', 
                    help="Directory with the trasnmat")
parser.add_argument("--lab_start_end", type=int, default=[1,9],
                    help="list of start and end labels")
parser.add_argument("--size_crops", type=int, default=128, nargs="+",
                    help="crops resolutions")
parser.add_argument("--ovrl", type=int, default=0.0, nargs="+",
                    help="overlap")

#########################
#### exp parameters ###
#########################
parser.add_argument("--forward_passes", default=1, type=float,help="forward_passes for MC droput")
parser.add_argument("--imps_val", default=-5, type=float,help="value for imp trans")
parser.add_argument("--pos_val", default=0.0, type=int,help="val for pos trans")
parser.add_argument("--w_crf", default=1.0, type=float, help="weight for crf loss")
parser.add_argument("--w_cross", default=1.0, type=float, help="weight for cross loss")
parser.add_argument("--tr_trans", type=bool, default=False, help="True for trainsition matrix")
parser.add_argument("--global_tr", default=False, help="True to train global matrix")
parser.add_argument("--warmup_cross", default=False, help="warmup epochs for crossentropy loss")
parser.add_argument("--stop_grad", default=False, type=bool, help="set impossible trasn to 0")
parser.add_argument("--softmax", default=False, type=bool, help="True to apply softmax at the end")
parser.add_argument("--sigmoid", default=False, type=bool, help="True to apply sigmoid at the end")
parser.add_argument("--tanh", default=False, type=bool, help="True to apply tanh at the end")

#########################
#### pretrained model ###
#########################
parser.add_argument("--pretrained", default="checkpoint.pth.tar", type=str, 
                    help="path to pretrained weights")

#########################
#### dist parameters ###
#########################
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### model parameters ###
#########################
parser.add_argument('--model', default='resnet',type=str,
                    help='(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
parser.add_argument('--model_depth',default=10,type=int,
                    help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
parser.add_argument('--conv1_t_size',default=5,type=int,
                    help='Kernel size in t dim of conv1.')
parser.add_argument('--conv1_t_stride', default=1, type=int,
                    help='Stride in t dim of conv1.')
parser.add_argument('--no_max_pool',action='store_true',
                    help='If true, the max pooling after conv1 is removed.')
parser.add_argument('--resnet_shortcut',default='B',type=str,
                    help='Shortcut type of resnet (A | B)')
parser.add_argument("--batch_size", default=8, type=int,
                    help="batch size ")

parser.add_argument("--workers", default=0, type=int,
                    help="number of data loading workers")
parser.add_argument("--root_path", type=str, default="./exp/v1_cross",
                    help="experiment root path")
parser.add_argument("--dump_path", type=str, default="",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=[31,10,5,20,8], help="seed")


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def main(model_iter, args_test):
    global args, transmat, nb_classes
    args = args_test
    # fix_random_seeds(args.seed[model_iter])
    check_folder(args.dump_path)
    
    args.pretrained = os.path.join(args.dump_path,args.pretrained)

    mask_img = read_tiff(args.mask_path)

    ######## load data #####
    
    image_sar = np.load(os.path.join(args.data_dir,'raster_16_test.npy'))
    depth,t,row,col = image_sar.shape
    num_channels = depth
    inp_seq = t   
    
    # get coordinates
    labels = np.load(os.path.join(args.data_dir,'labels_pad_test.npy'))
    length = labels.shape[0]
    
    # get coordinates
    coords_test = np.load(os.path.join(args.data_dir,'coords_test.npy'))
    
    # get add padding parametres with mask image
    _, stride, step_row, step_col, overlap = add_padding(mask_img, args.size_crops, args.ovrl)
    
    # convert original classes to ordered classes
    nb_classes = len(np.unique(labels))-1
    max_class = np.max(labels)
    labels = labels-1
    labels[labels==255] = max_class
    classes = np.unique(labels)
    lbl_tmp = labels.copy()
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    for j in range(len(classes)):
        labels[lbl_tmp == classes[j]] = labels2new_labels[classes[j]]
        
    depth,t,row,col = image_sar.shape
        
    pred_prob = np.zeros(shape = (len(mask_img[mask_img==2]),length), dtype='float16')
    pred_class = np.zeros(shape = (len(mask_img[mask_img==2]),length), dtype='uint8')
    pred_vit = np.zeros(shape = (len(mask_img[mask_img==2]),length), dtype='uint8')
    lab_vector = np.zeros(shape = (len(mask_img[mask_img==2]),length), dtype='uint8')

    # create a logger
    logger = create_logger(os.path.join(args.dump_path, "test.log"),rank=args.rank)
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    #build data
    test_dataset = DatasetFromCoord(
        image_sar,
        labels,
        coords_test,
        args.size_crops
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Load the transition matrix
    mat = scipy.io.loadmat(args.trans_path)
    mat = mat["trans_stack"]
    transmat = [mat[0][j] for j in range(mat.shape[1])]
    transmat = np.array(transmat).astype('float')
    transmat = np.moveaxis(transmat,0,2)
    transmat = transmat[:,:,args.lab_start_end[0]-1:args.lab_start_end[1]-1]
    transmat[transmat==0] = args.imps_val
    transmat[transmat==1] = args.pos_val
            
    logger.info("Building data done with {} images loaded.".format(len(test_loader)))

    # build model
    model = DeepLabv3_plus(args.model_depth, 
                 num_channels=num_channels,
                 inp_seq = inp_seq,
                 n_classes=nb_classes,
                 length = length,
                 t_kernel_size = args.conv1_t_size,
                 transmat = transmat,
                 w_cross = args.w_cross,
                 w_crf = args.w_crf,
                 batch_size = args.batch_size,
                 psize = args.size_crops,
                 tr_trans = args.tr_trans,
                 global_tr = args.global_tr,
                 stop_grad = args.stop_grad,
                 softmax = args.softmax,
                 sigmoid = args.sigmoid,
                 tanh = args.tanh)
    
    transmat = torch.Tensor(transmat).float().cuda()

    # model to gpu
    model = model.cuda()
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:0")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    cudnn.benchmark = True

    validate_network(test_loader, model, coords_test, pred_prob, pred_class, pred_vit, 
                                    lab_vector, stride, step_row, step_col, overlap, logger)

    logger.info("============ Inference finished ============")

def validate_network(dataloader, model, coords, pred_prob, pred_class, pred_vit, 
                     lab_vector, stride, step_row, step_col, overlap, logger):
    # set model to evaluation mode
    softmax = nn.Softmax(dim=2).cuda()
    for it in range(args.forward_passes):        
        model.eval()
        # enable_dropout(model)    % uncomment to apply MC dropout
        j=0
        logger.info("============ Forward pass number {} ============".format(it))
        for i, inputs in enumerate(dataloader):      
            # compute model loss and output
            if i < 40:
                start_time = time.time()
                
            out_batch, seq_vit, tags = model(inputs[0].cuda(non_blocking=True), 
                                       indic=True, tags=inputs[1].cuda(non_blocking=True),
                                       only_tag=True, overlap=overlap)  
            out_batch = softmax(out_batch)
            
            if i < 40:
                # Record time for this iteration
                iter_time = time.time() - start_time
                avg_time = iter_time * len(dataloader)
                
                logger.info(f"Iteration {it+1} took {iter_time:.2f} seconds.")
                logger.info(f"Estimated total prediction time: {avg_time/3600} hours.")
                
            
            out_batch = out_batch.data.cpu().numpy()

            pred_prob[j:j+out_batch.shape[0]] = np.amax(out_batch,axis=2)
            pred_class[j:j+out_batch.shape[0]] = np.argmax(out_batch,axis=2)
            pred_vit[j:j+out_batch.shape[0]] = seq_vit
            lab_vector[j:j+out_batch.shape[0]] = tags.data.cpu().numpy()
            
            j+=out_batch.shape[0]       
    
        np.save(os.path.join(args.dump_path, 'pred_prob_{}_{}'.format(str(args.ovrl),it)), pred_prob)
        np.save(os.path.join(args.dump_path, 'pred_class_{}_{}'.format(str(args.ovrl),it)), pred_class)
        np.save(os.path.join(args.dump_path, 'pred_vit_{}_{}'.format(str(args.ovrl),it)), pred_vit)
        np.save(os.path.join(args.dump_path, 'class_{}_{}'.format(str(args.ovrl),it)), lab_vector)

        pred_prob = np.zeros(pred_prob.shape, dtype='float16')
        pred_class = np.zeros(pred_class.shape, dtype='uint8')
        pred_vit = np.zeros(pred_vit.shape, dtype='uint8')
        lab_vector = np.zeros(lab_vector.shape, dtype='uint8')

if __name__ == "__main__":
    for mo in range(5):  # iterate over the 5 experiments
        args_test = parser.parse_args()
        args_test.dump_path = os.path.join(args_test.root_path,'model_{}'.format(mo))
        main(mo,args_test)
