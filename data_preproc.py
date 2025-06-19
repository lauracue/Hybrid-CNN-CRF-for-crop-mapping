#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Elena Cué La Rosa

"""

import argparse
import os
import numpy as np
import errno
from src.utils import (
    read_tiff,
    create_stack_GT,
    create_stack_SAR
)


def check_folder(folder_dir):
    '''Create folder if not available
    '''
    if not os.path.exists(folder_dir):
        try:
            os.makedirs(folder_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

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
        t, row, col = img.shape
        
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
    else:        
        npad_img = ((0,0),(overlap//2, step_row+overlap), (overlap//2, step_col+overlap))
        
        
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.',
                        help="Experiment directory containing params.json")
    parser.add_argument("--data_path", type=str, default='C:/PostDoc/Thesis_project/Tesis_All_files/Campo_Verde/Rasters/', 
                        help="path to dataset folder")
    parser.add_argument("--gt_path", type=str, default='C:/PostDoc/Thesis_project/Tesis_All_files/Campo_Verde/Labels_uint8/', 
                        help="path to dataset gt")
    parser.add_argument("--mask_path", type=str, default='C:/PostDoc/Thesis_project/Tesis_All_files/Campo_Verde/New_Masks/TrainTestMasks/TrainTestMask_50_50_Dec.tif', 
                        help="path to dataset mask, values 1 for training fields and value 2 for test fields")
    parser.add_argument('--ovrl', default=0.95, 
                        help="Overlap")
    parser.add_argument('--p_size', default=128, 
                        help="Tile size")
    parser.add_argument("--img_start_end", type=int, default=[1,14],
                        help="list of start and end images")
    parser.add_argument("--lab_start_end", type=int, default=[1,9],
                        help="list of start and end labels")
    parser.add_argument("--set_name", type=str, default='train',
                        help="set name: train or test")
    
    args = parser.parse_args()
    
    ######## load data #####
    assert os.path.isdir(args.gt_path), "Couldn't find the dataset at {}".format(args.gt_path)
    assert os.path.isdir(args.data_path), "Couldn't find the dataset at {}".format(args.data_path)
    
    if not os.path.isfile(os.path.join(args.data_dir,'stack_images',f'raster_16_{args.set_name}.npy')):   
        mask_img = read_tiff(args.mask_path)
        
        ind_mask = 2 if args.set_name == 'train' else 1
    
        # Load label images and create stack     
        labels = create_stack_GT(args.gt_path,args.lab_start_end[0]-1,args.lab_start_end[1])
        labels = np.moveaxis(labels,0,2)
        labels = labels.astype('uint8')
        labels[mask_img==ind_mask] = 0
        length = labels.shape[-1]
               
        # load SAR image and normalize breween -1 and 1
        image_sar = create_stack_SAR(args.data_path, args.img_start_end[0]-1, args.img_start_end[1], mask_img, 1)
        
        image_sar, _, _, _, _ = add_padding(image_sar, args.p_size, args.ovrl)
        
        zero_img = np.zeros((image_sar.shape[0],1,image_sar.shape[2],image_sar.shape[3]))
        
        labels = np.moveaxis(labels,2,0)
        
        check_folder(os.path.join(args.data_dir,'stack_images'))
        
        np.save(os.path.join(args.data_dir,'stack_images',f'raster_16_{args.set_name}'), image_sar.astype('float16'))
        np.save(os.path.join(args.data_dir,'stack_images',f'labels_{args.set_name}'), labels)
        
    else:
        image_sar = np.load(os.path.join(args.data_dir,'stack_images',f'raster_16_{args.set_name}.npy'))
        labels = np.load(os.path.join(args.data_dir,'stack_images',f'labels_{args.set_name}.npy'))
    


    
    coords_tr, cl, labels = extract_patches_coord(labels, args.p_size, 0.0, train = False)    
    
    np.save(os.path.join(args.data_dir,'stack_images','coords_{args.set_name}'), coords_tr)
    np.save(os.path.join(args.data_dir,'stack_images',f'labels_pad_{args.set_name}'), labels)



                