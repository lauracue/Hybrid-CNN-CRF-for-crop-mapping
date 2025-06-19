
from logging import getLogger
import pickle
import os
import numpy as np
import torch
from .logger import create_logger, PD_Stats
import pandas as pd
from osgeo import gdal
import glob
import errno
import random
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

"""
Some funtion were sourced from https://github.com/facebookresearch/swav
"""


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()

colormap_list = np.array([[255/255.0, 146/255.0, 36/255.0],
		               [255/255.0, 255/255.0, 0/255.0],
		               [164/255.0, 164/255.0, 164/255.0],
		               [255/255.0, 62/255.0, 62/255.0],
		               [0/255.0, 0/255.0, 0/255.0],
		               [170/255.0, 89/255.0, 255/255.0],
		               [0/255.0, 166/255.0, 83/255.0],
		               [40/255.0, 255/255.0, 40/255.0],
		               [187/255.0, 122/255.0, 83/255.0],
		               [255/255.0, 110/255.0, 203/255.0],
		               [45/255.0, 150/255.0, 255/255.0],
                       [255/255.0, 255/255.0, 255/255.0]])

def plot_figures(img_sar, cl, preds, preds_vit, probs, model_dir, epoch, nb_classes, set_name):
    
    cmap = ListedColormap(colormap_list)

    img_sar = img_sar[0,0,:cl.shape[1],:,:]    
    cl = cl[0,:,:,:]
    preds = preds[0,:,:,:]
    preds_vit = preds_vit[0,:,:,:]
    probs = np.amax(probs[0,:,:,:,:],axis=1)

    nrows = 5
    ncols = preds.shape[0]
    imgs = [img_sar,cl,preds,probs,preds_vit]
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,10))
    
    cont = 0
    cont_img = 0

    for ax in axes.flat:
        ax.set_axis_off()
        if cont==0:
            im = ax.imshow(imgs[cont][cont_img,:,:], cmap = 'gray')
        elif cont == 1 or cont == 2 or cont == 4:
            im = ax.imshow(imgs[cont][cont_img,:,:], cmap=cmap,vmin=0, vmax=nb_classes)
        else:
            im = ax.imshow(imgs[cont][cont_img,:,:], cmap='OrRd', interpolation='nearest')
        
        cont_img+=1
        if cont_img==ncols:
            cont+=1
            cont_img=0

    
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    
    # set the colorbar ticks and tick labels
    cbar.set_ticks(np.arange(0, 1, nb_classes))
    
    plt.axis('off')
    plt.savefig(os.path.join(model_dir, set_name + str(epoch) + '.png'), dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()


def check_folder(folder_dir):
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path, map_location="cuda:0")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(1, keepdim=True).view(-1)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def accuracy_tr(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred[pred > 1] = 2
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        
        return correct_k.mul_(100.0 / batch_size)
    
def f1(y_pred, y_true, depth, epsilon=1e-7):
    with torch.no_grad():
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = torch.nn.functional.one_hot(y_true, depth).to(torch.float32)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
    
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
    
        f1 = 2*(precision*recall) / (precision + recall + epsilon)
        f1 = f1.clamp(min=epsilon, max=1-epsilon)
        return f1
    
def listFiles(top_dir='.', exten='.tif'):
    list_dir = os.listdir(top_dir)
    list_dir.sort(key=lambda f: int(filter(str.isdigit, f.split('_')[0])))

    filesPathList = list()
    for dirpath in list_dir:
        files_tif = glob.glob(os.path.join(top_dir,dirpath) + '/*.tif')
        filesPathList.extend(files_tif)

    return filesPathList

def read_tiff(tiff_file):
    print(tiff_file)
    data = gdal.Open(tiff_file).ReadAsArray()
    return data

def load_norm(path, mask=[0], mask_indx = 0):
    img_paths = sorted(glob.glob(path + '/*.tif'))
    image = [np.expand_dims(read_tiff(img).astype('float32'), -1) for img in img_paths]
    image = np.concatenate(image, axis=-1)
    image = db2intensities(image)
    print("Image shape: ", image.shape, " Min value: ", image.min(), " Max value: ", image.max())
    image = filter_outliers(image, mask=mask, mask_indx = mask_indx)
    print("Filter Outliers, Min value: ", image.min(), " Max value: ", image.max())
    image = normalize(image)
    image = np.moveaxis(image,2,0)
    print("Normalize, Min value: ", image.min(), " Max value: ", image.max())
    return image

def db2intensities(img):
    img = 10**(img/10.0)
    return img

def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0], mask_indx = 0):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask==mask_indx, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
    return img

def normalize(img):
    '''image shape: [row, cols, channels]'''
    # img = 2*(img -img.min(axis=(0,1), keepdims=True))/(img.max(axis=(0,1), keepdims=True) - img.min(axis=(0,1), keepdims=True)) - 1
    img = (img -img.min(axis=(0,1), keepdims=True))/(img.max(axis=(0,1), keepdims=True) - img.min(axis=(0,1), keepdims=True))
    return img

def create_stack_SAR(raster_path, start, end, mask=[0], mask_indx = 0):
    list_dir = os.listdir(raster_path)
    list_dir.sort(key=lambda f: int((f.split('_')[0])))
    list_dir = [os.path.join(raster_path,f) for f in list_dir[start:end]]    
    
    img_stack = [load_norm(img, mask, mask_indx) for img in list_dir]
    img_stack = np.array(img_stack)
    
    img_stack = np.moveaxis(img_stack,1,0)    
    return img_stack

def create_stack_GT(path,start,end):
    list_dir = sorted(glob.glob(path + '*.tif'))
    list_dir = list_dir[start:end]
    img_stack = [np.expand_dims(read_tiff(img).astype('uint32'), 0) for img in list_dir]
    
    img_stack = np.concatenate(img_stack, axis=0)
  
    return img_stack

    
def fun_sort(x):
    return int(x.split('_')[0])



def balance_coords(coords, labels, samples_per_class=None):
    '''Function to balance samples using data augmentation
        input:
            data: patches (num_pacthes, row, col, bands)
            labels: class for each patch
            samples_per_class: number of samples for each class
    '''
    
    # get classes
    coords = np.array(coords).transpose()
    uniq, count = np.unique(labels, axis = 0, return_counts=True)
    if not samples_per_class:
        samples_per_class = np.max(count)
    # total output samples = num_class * samples_per_class
    num_total_samples = len(uniq)*samples_per_class
    
    # create empty matrices for output labels and patches
    out_labels = np.zeros((num_total_samples,labels.shape[-1]))
    out_coords = np.zeros((num_total_samples,2), dtype='int64')

    k = 0
    for clss in uniq:
        # get total samples of class = clss
        index_labels = np.where(np.all(labels==clss,axis=1))
        index_labels = index_labels[0]
        num_samples = len(index_labels)
        
        if num_samples > samples_per_class:
            # if num_samples > samples_per_class choose samples randomly
            index = range(num_samples)
            index = np.random.choice(index, samples_per_class, replace=False)
            # write in output matrices
            out_labels[k*samples_per_class:(k+1)*samples_per_class,:] = labels[index_labels[index]]
            out_coords[k*samples_per_class:(k+1)*samples_per_class,:] = coords[index_labels[index]]

        else:
            index = range(num_samples)
            index = np.random.choice(index, samples_per_class, replace=True)
            # write in output matrices
            out_labels[k*samples_per_class:(k+1)*samples_per_class,:] = labels[index_labels[index]]
            out_coords[k*samples_per_class:(k+1)*samples_per_class,:] = coords[index_labels[index],:]
        k += 1
        

    # Permute samples randomly
    idx = np.random.permutation(out_labels.shape[0])
    out_labels = out_labels[idx]
    out_coords = out_coords[idx]

    return out_coords, out_labels

def add_padding(img, psize, overl):
    '''Function to padding image
        input:
            patches_size: psize
            stride: stride
            img: image (row,col,bands)
    '''    

    try:
        bands, t, row, col = img.shape
    except:
        bands = 0
        try:
            t, row, col = img.shape
        except:
            t = 0
            row, col = img.shape
        
    # Percent of overlap between consecutive patches.
    # The overlap will be multiple of 2
    overlap = int(round(psize * overl))
    overlap -= overlap % 2
    stride = psize - overlap

    # Add Padding to the image to match with the patch size and the overlap
    row += overlap//2
    col += overlap//2
    step_row = (stride - row % stride) % stride
    step_col = (stride - col % stride) % stride
    
    if bands>0:
        npad_img = ((0,0),(0,0),(overlap//2, step_row+overlap), (overlap//2, step_col+overlap))
    elif bands==0 and t>0:        
        npad_img = ((0,0),(overlap//2, step_row+overlap), (overlap//2, step_col+overlap))
    else:
        npad_img = ((overlap//2, step_row+overlap), (overlap//2, step_col+overlap))
        
        
    # padd with symetric (espelhado)    
    pad_img = np.pad(img, npad_img, mode='symmetric')

    # Number of patches: k1xk2
    k1, k2 = (row+step_row)//stride, (col+step_col)//stride
    print('Number of patches: %d' %(k1 * k2))

    return pad_img, stride, step_row, step_col, overlap


def extract_patches_coord(gt, psize, ovrl, train = False):
    '''Function to extract patches coordinates from rater images
        input:
            img: raster image 
            gt: shpafile raster
            psize: image patch size
            ovrl: overlap to extract patches
            model: model type

    '''
    # add padding to gt raster
    img_gt, stride, step_row, step_col, overlap = add_padding(gt, psize, ovrl)
    t,row,col = img_gt.shape
    
    # get classes
    if train:
        labels = img_gt[:,img_gt[0]!=0]
        unique_class = np.unique(labels, axis=1).T
    else:
        labels = img_gt[:,img_gt[0]!=0]
        unique_class = np.unique(labels, axis=1).T

    # loop over x,y coordinates and extract patches
    coord_list = list()
    classes = list()

    if train:
        for m in range(psize//2,row-psize//2,stride): 
            for n in range(psize//2,col-psize//2,stride):
                coord = [m,n]
                class_patch = img_gt[:,m,n]

                if class_patch in unique_class:
                    coord_list.append(coord)                    
                    classes.append(class_patch)
                            
                elif np.unique(class_patch) == 0:
                    lab_p = img_gt[:,coord[0]-psize//2:coord[0]+psize//2 + psize%2,coord[1]-psize//2:coord[1]+psize//2 + psize%2]
                    no_class = np.sum(lab_p[0]>0)
                    if no_class>0.1*psize**2:
                        coord_list.append(coord)
                        uniq, count = np.unique(lab_p[:,lab_p[0]!=0], axis=1, return_counts=True)
                        classes.append(uniq[:,np.argmax(count)])
                                        
                 
        uniq, count = np.unique(np.array(classes), axis=0, return_counts=True)             
        samples_per_class = np.max(count)
        num_total_samples = len(np.unique(classes, axis=0))*samples_per_class
        coordsx_tr = np.zeros((num_total_samples,2), dtype='int')
        labels_tr = np.zeros((num_total_samples,t), dtype='int')
        
        k = 0
        coord_list = np.array(coord_list)
        classes = np.array(classes)
        for key in uniq:
            # get total samples per class
            index = np.where((classes[:,0] == key[0])&(classes[:,1] == key[1]))[0]
            num_samples = len(index)
            
            if num_samples > samples_per_class:
                # if num_samples > samples_per_class choose samples randomly
                index = np.random.choice(index, samples_per_class, replace=False)
           
            else:
                index = np.random.choice(index, samples_per_class, replace=True)
                             
            
            coordsx_tr[k*samples_per_class:(k+1)*samples_per_class,:] = coord_list[index,:]
            labels_tr[k*samples_per_class:(k+1)*samples_per_class,:] = classes[index]
            k += 1
    
        # Permute samples randomly
        idx = np.random.permutation(num_total_samples)
        coordsx_tr = coordsx_tr[idx,:]
        labels_tr = labels_tr[idx,:]    

    else:
        for m in range(psize//2,row-psize//2+1,stride): 
            for n in range(psize//2,col-psize//2+1,stride):
                coord = [m,n]
                class_patch = img_gt[:,m,n]
                
                if class_patch in unique_class:
                    coord_list.append(coord)                    
                    classes.append(class_patch)
                
                elif np.unique(class_patch) == 0:
                    lab_p = img_gt[:,coord[0]-psize//2:coord[0]+psize//2 + psize%2,coord[1]-psize//2:coord[1]+psize//2 + psize%2]
                    if np.any(lab_p>0):
                        coord_list.append(coord)
                        classes.append(class_patch)
                            
        coordsx_tr = np.array(coord_list)
        labels_tr = np.array(classes)
    
    return coordsx_tr, labels_tr, img_gt


def classification_report_csv(report, out_path, acc, kappa):
    report_data = []
    lines = report.split('\n')
    cont = 0
    for line in lines[2:-1]:
        row = {}
        row_data = line.split('      ')
        try:
            row['class'] = row_data[1]
            row['precision'] = float(row_data[2])
            row['recall'] = float(row_data[3])
            row['f1_score'] = float(row_data[4])
            row['support'] = float(row_data[5])

            if cont==0:
                row['OA'] = acc
                row['Kappa'] = kappa
            else:
                row['OA'] = ''
                row['Kappa'] = '' 
        except:
            pass
        cont+=1
    
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(os.path.join(out_path,'val_classification_report.csv'), index = False)
