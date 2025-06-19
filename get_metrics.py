"""Get metrcis"""

import os

import numpy as np
import glob
from src.utils import read_tiff, create_stack_GT
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score,precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 25})
sns.set_style("darkgrid")


def back2lab(predict,new_labels2labels):
    lbl_tmp = predict.copy()
    classes_pred = np.unique(predict)
    for j in range(len(classes_pred)):
        predict[lbl_tmp == classes_pred[j]] = new_labels2labels[classes_pred[j]]      

    predict+=1
    return predict


def save_metrics(model_dir,label,predict_mask_test,pred_name):
    # get classification report
    predict_mask_test = np.uint8(predict_mask_test)
    cohen_score = cohen_kappa_score(label, predict_mask_test)
    acc = accuracy_score(label, predict_mask_test)
    
    classes_all = np.unique(np.concatenate([label,predict_mask_test]))
    lab_class = np.unique(label)
    # save report in csv
    clf_rep = precision_recall_fscore_support(label, predict_mask_test)
    out_dict = {
                              "precision" :clf_rep[0].round(4)
                            ,"recall" : clf_rep[1].round(4)
                            ,"f1-score" : clf_rep[2].round(4)
                            ,"support" : clf_rep[3]                    
                            }
    
    out_df = pd.DataFrame(out_dict, index = classes_all)
    avg_tot = (out_df.apply(lambda x: round(np.sum(x)/len(lab_class), 4) if x.name!="support" else  round(x.sum(), 4)).to_frame().T)
    oa_acc = (out_df.apply(lambda x: round(acc, 4) if x.name=="precision" else  round(x.sum()*0, 1)).to_frame().T)
    kappa = (out_df.apply(lambda x: round(cohen_score, 4) if x.name=="precision" else  round(x.sum()*0, 1)).to_frame().T)
    avg_tot.index = ["avg/total"]
    oa_acc.index = ['OA']
    kappa.index = ['Kappa']
    out_df = out_df.append(avg_tot)
    out_df = out_df.append(oa_acc)
    out_df = out_df.append(kappa)
    
    out_df.to_csv(os.path.join(model_dir,'val_report_{}.csv'.format(pred_name)), sep='\t')
    
    # Get and save confusion matrix as figure
    classes = classes_all
    cm = confusion_matrix(label, predict_mask_test, labels =classes, normalize='true')
    cm1 = cm[np.sum(cm,1)!=0,:][:,np.sum(cm,1)!=0]
      
    return avg_tot['precision'][0], avg_tot['recall'][0], avg_tot['f1-score'][0], acc, cohen_score, cm1.diagonal()


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./exp',
                        help="Experiment directory containing params.json")
    parser.add_argument("--img_sar", type=str, default="D:\Projects\Campo_Verde_folders/Rasters/",
                        help="path to dataset folder")
    parser.add_argument("--gt_dir", type=str, default="D:\Projects\Campo_Verde_folders/Labels_unique/",
                        help="path to dataset gt")
    parser.add_argument("--mask_dir", type=str, default="D:\Projects\Campo_Verde_folders/TrainTestMask_50_50.tif",
                        help="path to dataset mask")
    parser.add_argument('--trans_path', default='trans_stack_orig_label.mat', 
                        help="Directory with the mask")
    parser.add_argument('--lab_start', default=1, 
                        help="Start label image")
    parser.add_argument('--lab_end', default=9, 
                        help="End label image")  

    # Load the parameters
    args = parser.parse_args()
    
    assert os.path.isdir(args.gt_dir), "Couldn't find the dataset at {}".format(args.gt_dir)
    assert os.path.isdir(args.img_sar), "Couldn't find the dataset at {}".format(args.img_sar)
    
    mask_img = read_tiff(args.mask_dir)
    
    print ('#' * 30)
    print('Load data ')
    print ('#' * 30)

    # get coordinates of training samples
    # mask_img = 1 
    coords = np.where(mask_img==2)

    labels_list = glob.glob(args.gt_dir + '/*.tif')
    labels_list.sort()     
    labels = create_stack_GT(args.gt_dir,args.lab_start-1,args.lab_end)
    labels = np.rollaxis(labels,0,3)
    labels = labels.astype('uint8')
    # np.save(os.path.join(args.model_dir,'labels'),labels[coords])
    
    # convert original classes to ordered classes
    labels_tr = labels.copy()
    max_class = np.max(labels_tr)
    labels_tr = labels_tr-1
    labels_tr[labels_tr==255] = max_class
    classes = np.unique(labels_tr)
    new_labels2labels = dict((i, c) for i, c in enumerate(classes))
    
    # v_exp = ['v3','v2','v5','v6','v7']
    # models = [0,1,2,3,4]
    v_exp = ['v1_cross','v2_tr_trans']
    v_names = ["$\it{CNN}$","$\it{CNN-Vit}$","$\it{CNN-CRF}_{A}$",
               "$\it{CNN-CRF}_{C}$-tanh-10"]
    
    forward_passes = 1
    
    oa_list = np.zeros((forward_passes,2,len(v_names),9))
    f_list =  np.zeros((forward_passes,2,len(v_names),9))
    aa_list = np.zeros((forward_passes,2,len(v_names),9))
    pre_list =  np.zeros((forward_passes,2,len(v_names),9))
    rec_list = np.zeros((forward_passes,2,len(v_names),9))
    kapp_list =  np.zeros((forward_passes,2,len(v_names),9))

    models = [0]
    for exp in range(len(v_exp)):
        
        for it in range(forward_passes):
     
            try:
                pred_class = np.load(os.path.join(args.model_dir,v_exp[exp],'model_0','pred_class_0.3_{}.npy'.format(it)))
                
                pred_seq = np.load(os.path.join(args.model_dir,v_exp[exp],'model_0','pred_vit_0.3_{}.npy'.format(it)))
                
                labels = np.load(os.path.join(args.model_dir,v_exp[exp],'model_0','class_0.3_{}.npy'.format(it)))
                
                for i in range(pred_seq.shape[1]):
                    pred = pred_class[:,i]
                    pred = back2lab(pred,new_labels2labels)
                    pred_vit = pred_seq[:,i]
                    pred_vit = back2lab(pred_vit,new_labels2labels)
                    lab = labels[:,i]
                    lab = back2lab(lab,new_labels2labels)

                    
                    p,r,f,oa,kapp,aa = save_metrics(os.path.join(args.model_dir,v_exp[exp]), lab, pred, 'class_{}'.format(i))
                    p1,r1,f1,oa1,kapp1,aa1 = save_metrics(os.path.join(args.model_dir,v_exp[exp]), lab, pred_vit, 'seq_{}'.format(i))
                    
                    aa_list[it,0,exp,i] = np.mean(aa)*100
                    pre_list[it,0,exp,i] = np.mean(p)*100
                    rec_list[it,0,exp,i] = np.mean(r)*100
                    f_list[it,0,exp,i] = np.mean(f)*100
                    oa_list[it,0,exp,i] = np.mean(oa)*100
                    kapp_list[it,0,exp,i] = np.mean(kapp)*100
                    
                    aa_list[it,1,exp,i] = np.mean(aa1)*100
                    pre_list[it,1,exp,i] = np.mean(p1)*100
                    rec_list[it,1,exp,i] = np.mean(r1)*100
                    f_list[it,1,exp,i] = np.mean(f1)*100
                    oa_list[it,1,exp,i] = np.mean(oa1)*100
                    kapp_list[it,1,exp,i] = np.mean(kapp1)*100
                    
                print ('*' * 50)
                print ('Classification done!!!!')
                print ('*' * 50) 

            except:
                print("No exp for this model")
                
                
    months = ['Oct','Nov','Dec','Jan','Feb','Mar','May','Jun','Jul']
               
    
    c1 = pd.DataFrame(np.array(oa_list[:,0,0,:]), columns=months).assign(Metric="OA").assign(Model=v_names[0])
    c2 = pd.DataFrame(np.array(f_list[:,0,0,:]), columns=months).assign(Metric="avgF1").assign(Model=v_names[0])
    c3 = pd.DataFrame(np.array(pre_list[:,0,0,:]), columns=months).assign(Metric="avgUA").assign(Model=v_names[0])
    c4 = pd.DataFrame(np.array(rec_list[:,0,0,:]), columns=months).assign(Metric="avgPA").assign(Model=v_names[0])
    
    cnn = pd.concat([c1, c2, c3, c4])
    
    for exp in range(len(v_exp)):
    
        c5 = pd.DataFrame(np.array(oa_list[:,1,exp,:]), columns=months).assign(Metric="OA").assign(Model=v_names[exp+1])
        c6 = pd.DataFrame(np.array(f_list[:,1,exp,:]), columns=months).assign(Metric="avgF1").assign(Model=v_names[exp+1])
        c7 = pd.DataFrame(np.array(pre_list[:,1,exp,:]), columns=months).assign(Metric="avgUA").assign(Model=v_names[exp+1])
        c8 = pd.DataFrame(np.array(rec_list[:,1,exp,:]), columns=months).assign(Metric="avgPA").assign(Model=v_names[exp+1])
        
        cnn1 = pd.concat([c5, c6, c7, c8])
        
        cnn =  pd.concat([cnn,cnn1])

    
    mdf = pd.melt(cnn, id_vars=['Model',"Metric"])
    mdf=mdf.rename(columns = {'variable':'Date', 'value':'Value (%)' })     # MELT
    
    ax = sns.catplot(x="Date", y="Value (%)",
    
                    hue="Model", col="Metric", col_wrap=2,
    
                    data=mdf, kind="bar", legend=False,
    
                    height=7, aspect=2, palette = sns.color_palette("tab10"))

    ax.set(ylim=(65, 100))
    plt.yticks(np.arange(65, 100, 5))
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 2.2), ncol=1)
    plt.savefig(os.path.join(args.model_dir,'metrics_all.svg'), 
                dpi = 1000, format='svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    