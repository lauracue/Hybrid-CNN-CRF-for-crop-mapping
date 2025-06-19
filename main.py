"""
@author: Laura Elena Cué La Rosa

"""

import argparse
import math
import os
from logging import getLogger

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import scipy.io
from src.metrics import metrics

from src.utils import (
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    check_folder,
    plot_figures
)
from src.dataloder import DatasetFromCoord
from src.network import DeepLabv3_plus

logger = getLogger('cnn_crf')

parser = argparse.ArgumentParser(description="Implementation of 3DCNN-CRF")

#########################
#### data parameters ####
#########################
parser.add_argument('--data_dir', default='./stack_images',
                        help="Path containing the raster and labels in numpy format")
parser.add_argument('--trans_path', default='trans_stack_orig_label.mat', 
                    help="Directory with the trasnmat")
parser.add_argument("--lab_start_end", type=int, default=[1,9],
                    help="list of start and end labels")
parser.add_argument("--size_crops", type=int, default=128, nargs="+",
                    help="crops resolutions")
parser.add_argument("--samples", default=10000, type=int,
                    help="samples per epoch")


#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=16, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=0.01, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0.00001, help="final learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=1, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### exp parameters ###
#########################
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
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
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


#########################
#### other parameters ###
#########################
parser.add_argument("--workers", default=0, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=1,
                    help="Save the model periodically")
parser.add_argument("--root_path", type=str, default="./exp/v1_cross",
                    help="experiment root path")
parser.add_argument("--dump_path", type=str, default="",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=[31,10,5,20,8], help="seeds")


def main(model_iter, args_training):
    global args, transmat, nb_classes, fig_dir
    args = args_training
    check_folder(args.dump_path)
    fig_dir = os.path.join(args.dump_path,'figures')
    check_folder(fig_dir)
    fix_random_seeds(args.seed[model_iter])

    ######## load data #####
    
    image_sar = np.load(os.path.join(args.data_dir,'raster_16_train.npy'))
    depth,t,row,col = image_sar.shape
    num_channels = depth
    inp_seq = t   
    
    # get coordinates
    labels = np.load(os.path.join(args.data_dir,'labels_pad_train.npy'))
    length = labels.shape[0]
    
    # get coordinates
    coords_tr = np.load(os.path.join(args.data_dir,'coords_train.npy'))
        
    # convert original classes to ordered classes
    nb_classes = len(np.unique(labels))-1
    max_class = np.max(labels)
    labels = labels-1
    labels[labels==255] = max_class
    classes = np.unique(labels)
    lbl_tmp = labels.copy()
    labels2new_labels = dict((c, i) for i, c in enumerate(classes))

    for j in range(len(classes)):
        labels[lbl_tmp == classes[j]] = labels2new_labels[classes[j]]
    
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build train data
    train_dataset = DatasetFromCoord(
        image_sar,
        labels,
        coords_tr,
        args.size_crops,
        args.samples,
        augm=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    
    # Load transition matrix
    mat = scipy.io.loadmat(args.trans_path)
    mat = mat["trans_stack"]
    transmat = [mat[0][j] for j in range(mat.shape[1])]
    transmat = np.array(transmat).astype('float')
    transmat = np.moveaxis(transmat,0,2)
    transmat = transmat[:,:,args.lab_start_end[0]-1:args.lab_start_end[1]-1]
    transmat[transmat==0] = args.imps_val
    transmat[transmat==1] = args.pos_val
    
    if args.tr_trans:
        transmat = None
            
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = DeepLabv3_plus(10, 
                 num_channels=num_channels,
                 inp_seq = inp_seq,
                 n_classes=nb_classes,
                 length = length,
                 t_kernel_size = 5,
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
    
    if not args.tr_trans:
        transmat = torch.Tensor(transmat).float().cuda(non_blocking=True)
        
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd
    )
    
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    
    logger.info("Building optimizer done.")


    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore["epoch"]


    cudnn.benchmark = True


    for epoch in range(start_epoch, args.epochs):
        np.random.shuffle(train_loader.dataset.coord)
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # train the network
        scores_tr, metrics_mean_vit = train(train_loader, model, optimizer, 
                              epoch, lr_schedule)

        
        training_stats.update(scores_tr)
        

        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            logger.info("============ Saving best models at epoch %i ... ============" % epoch)
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )


def train(train_loader, model, optimizer, epoch, lr_schedule):

    model.train()
    
    summ = []
    summ_vit = []
    
    loss_avg = AverageMeter()

    for it, inputs in enumerate(train_loader):      

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        # ============ multi-res forward passes and loss ... ============
        # compute model loss and output
        input_batch = inputs[0].cuda(non_blocking=True)
        lab = inputs[1].cuda(non_blocking=True)
        
        if it % 50 == 0:
            out_batch, seq_vit = model(input_batch, True)
        else:
            out_batch = model(input_batch)
            loss = model.loss(out_batch, lab, epoch, args.warmup_cross)
        
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()
        
        # performs updates using calculated gradients
        optimizer.step()
        
        # update the average loss
        loss_avg.update(loss.item())

        # Evaluate summaries only once in a while
        if it % 50 == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            out_batch = out_batch.data.cpu().numpy()
            B, L, C, H, W =  out_batch.shape
            pred_net = np.argmax(out_batch, axis=2)
            if it == 0:
                seq_vit1 = np.reshape(seq_vit, (B, H, W, L))
                seq_vit1 = np.moveaxis(seq_vit1, -1, 1)
                plot_figures(inputs[0].cpu().numpy(), inputs[1],
                              pred_net, seq_vit1, out_batch, 
                              fig_dir, epoch, nb_classes+1, 'train')
                
                seq_vit1 = []
    
            pred_net = np.reshape(pred_net, (B, L, H*W))
            pred_net = np.moveaxis(pred_net, 2, 1)
            pred_net = np.reshape(pred_net, (B*H*W, L))             
            lab = inputs[1].data.cpu().numpy()
            
            lab = np.reshape(lab, (B, L, H*W))
            lab = np.moveaxis(lab, 2, 1)
            lab = np.reshape(lab, (B*H*W, L))
            
            pred_net = pred_net[lab[:,0]<nb_classes]
            seq_vit =  seq_vit[lab[:,0]<nb_classes]
            lab = lab[lab[:,0]<nb_classes]

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](pred_net, lab) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            
            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](seq_vit, lab) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ_vit.append(summary_batch)

            
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    loss=loss_avg,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
                
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ], axis=0) for metric in summ[0]}
    logger.info("- Train metrics: ")
    for k, v in metrics_mean.items():
        try:
            logger.info("{}: {:05.3f}".format(k, v))
        except:
            logger.info("{}: ".format(k) + " ".join("{:05.3f},".format(it) for it in v))
    metrics_mean_vit = {metric: np.mean([x[metric] for x in summ_vit], axis=0) for metric in summ_vit[0]}
    
    logger.info("- Train metrics viterbi: ")
    for k, v in metrics_mean_vit.items():
        try:
            logger.info("{}: {:05.3f}".format(k, v))
        except:
            logger.info("{}: ".format(k) + " ".join("{:05.3f},".format(it) for it in v))
    metrics_mean_vit = 0
            
    return (epoch, loss_avg.avg), metrics_mean_vit


if __name__ == "__main__":
    for mo in range(5):  # run exp 5 times with 5 seeds
        args_tr = parser.parse_args()
        args_tr.dump_path = os.path.join(args_tr.root_path,'model_{}'.format(mo))
        main(mo,args_tr)
