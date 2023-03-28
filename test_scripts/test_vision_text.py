import argparse
import os
import pathlib
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix
from src.datamodules.components import (CXRMultimodalv2,
                                        multimodal_collate_fn_v2)
from src.models import cxr_mm_model, cxr_mm_model_ke, cxr_vision_musicvit
from torch.utils.data import DataLoader
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=100)
parser.add_argument('--gpu_idx', type=str, default='0')
parser.add_argument('--log_dir', type=str, default="./results/mm/")
parser.add_argument('--message', '--msg', type=str, default=None)
parser.add_argument('--dataset', type=str, default='comp_v3')
parser.add_argument('--pretrained', type=str, default=None)
parser.add_argument('--gold', type=bool, default=False)
parser.add_argument('--resnet', type=bool, default=False)


def test(args, test_dataloader, model, device, log_dir):

    print('[*] Test Phase')
    model.to(device)
    model = model.eval()
    correct = 0
    total = 0

    overall_output = []
    overall_pred = []
    overall_logits = []
    overall_gt = []
    overall_pat = []
    iter_ = 0

    for idx, batch in tqdm.tqdm(enumerate(test_dataloader)):

        base_img, pair_img, cap, cap_l, report_sent, change_labels, disease_labels, patient_id, report_keys = batch
        
        base_img = base_img.to(device)
        pair_img = pair_img.to(device)
        caption_ids = cap['input_ids'].to(device)
        token_type_ids = cap['token_type_ids'].to(device)
        attention_mask = cap['attention_mask'].to(device)

        with torch.no_grad():
            
            _,_, outputs,_ = model(caption_ids,
                          None,
                          attention_mask,
                          token_type_ids,
                          base_img,
                          pair_img,
                           )
            
            labels = change_labels
            
            outputs = F.softmax(outputs , dim=1)
            logits, preds = outputs.max(1)
            preds = preds.cpu()

            new_labels = []
            for i in range(labels.shape[0]):
                new_labels.append(outputs[i, 1].cpu().detach().item())

            logits_cpu = logits.cpu().detach().numpy().tolist()
            preds_cpu = preds.cpu().detach().numpy().tolist()
            labels_cpu = labels.cpu().detach().numpy().tolist()
            
            overall_logits += logits_cpu
            overall_output += new_labels
            overall_pred += preds_cpu
            overall_gt += labels_cpu
            
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
        
        iter_ += 1
        idx += labels.shape[0]
    
    print('[*] Test Acc: {:5f}'.format(100.*correct/total))
    
    tn, fp, fn, tp = confusion_matrix(overall_gt, overall_pred).ravel()
    
    save_confusion_matrix(confusion_matrix(overall_gt, overall_pred), ['Improved','Worsened'], log_dir)
    save_results_metric(tn, tp, fn, fp, correct, total, log_dir)
    save_roc_auc_curve(overall_gt, overall_output, log_dir)
    save_csv(overall_pat, overall_gt, overall_pred, overall_output, log_dir)
        

def main(args):
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        
    # 0. device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = '{}_{}_{}'.format(today, args.message, args.dataset)
    
    log_dir = os.path.join(args.log_dir, folder_name)

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    

    # make datasets & dataloader (train & test)
    print('[*] prepare datasets & dataloader...')

    data_dir = '/mnt/nas252/Kyungjin.Cho/_jykim/MuSiC-ViT-main/MuSiC-ViT-main/json/'
    batch_size= 1
    num_workers= 4
    mean= 0.2
    std= 0.4
    split='valid'
    transform = {'resize': {'height': 512,
                            'width': 512}}
    
    dataset_config = {
        "max_words": 128,
        "d_num": 14,
        "pretrained_tokenizer": "emilyalsentzer/Bio_ClinicalBERT",
        "text_name": "_comp_fov_text_v3",
        "only_fu_text_name": True,
        "random_erase" : 0.99,
        "empty_fu_sent": False,
        }
    
    transforms_list = dict()
    transforms_list['train'] = transform
    transforms_list['valid'] = transform
    transforms_list['test'] = transform

    test_datasets = CXRMultimodalv2(data_dir=data_dir, 
                               split=split,
                               batch_size=batch_size, 
                               num_workers=num_workers,
                               mean=mean,
                               std=std,
                               transforms = transforms_list['test'],
                               gold=args.gold,
                               d_num = dataset_config["d_num"],
                               max_words = dataset_config['max_words'],
                               pretrained_tokenizer = dataset_config["pretrained_tokenizer"],
                               text_name = dataset_config["text_name"],
                               only_fu_text_name = dataset_config["only_fu_text_name"],
                               random_erase = dataset_config['random_erase'],
                              )

    test_dataloader = DataLoader(
            test_datasets,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=None,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
     
    # select network
    print('[*] build network...')
    
    model = cxr_mm_model_ke.CXRTextGuidedModelKE(
            in_channels= 1,
            stem_channels= 16,
            cmt_channelses= [46, 92, 184, 368],
            pa_channelses= [46, 92, 184, 368],
            R= 3.6,
            repeats= [2, 2, 10, 2],
            input_size= 512,
            sizes= [128, 64, 32, 16],
            patch_ker= 2,
            patch_str= 2,
            disease_classes= 14,
            num_labels= 2,
            freeze_bert= True,
            cxr_bert_pretrained = '/mnt/nas252/Kyungjin.Cho/_jykim/mm/FU_trainer/logs/CXR_FU_text/230211_v3_only_fu/2023-02-11_16-28-37/checkpoints/epoch_043.ckpt',
            bert_pretrained= "emilyalsentzer/Bio_ClinicalBERT",
            output_attentions= True,
            output_hidden_states= True)

    print("[*] model")

    model = model.to(device)  

    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=device)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {key.replace("model.", ""): value for key, value in pretrained_dict.items()}
        model.load_state_dict(pretrained_dict, strict=False)
        print("[*] checkpoint load completed")
    
    test(args, test_dataloader, model, device, log_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    print('-'*50)
    print(args)
    print('-'*50)
    main(args)
