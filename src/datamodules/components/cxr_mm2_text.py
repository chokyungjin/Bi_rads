import json
import random

import numpy as np
import torch
import torch.utils.data as data
from transformers import BertTokenizer, GPT2Tokenizer

from ._base import BaseDataset
from .utils import get_imgs, label_process


class CXRMultimodalv2_text(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 transforms: dict,
                 mean: float,
                 std: float,
                 max_words: int,
                 pretrained_tokenizer: str,
                 d_num = int,
                 text_name = str,
                 empty_fu_sent : bool = False,
                 empty_fu_sent_dummy : bool = False,
                 random_erase : float = 0.0,
                 only_base_text_name : bool = False,
                 only_fu_text_name : bool = False,
                 use_kd_get_item : bool = False,
                 ch_noch : bool = False,
                 gold : bool = False,
                 **kwargs,
    ):
        
        super().__init__(data_dir, split, transforms)

        # build transforms
        self.transforms = self._get_transform(
            transforms=transforms,
            mean=mean,
            std=std,
        )
        self.d_num = d_num
        self.text_name = text_name
        self.token_pretrained_model = pretrained_tokenizer
        self.only_base_text_name = only_base_text_name
        self.only_fu_text_name = only_fu_text_name
        self.random_erase = random_erase
        self.empty_fu_sent = empty_fu_sent
        self.empty_fu_sent_dummy = empty_fu_sent_dummy
        self.ch_noch = ch_noch
        self.gold = gold
        self.use_kd_get_item = use_kd_get_item

        if self.gold:
            self.json_name = '{}mimic_gold_comp_fov_text.json'.format(data_dir)
        elif self.ch_noch:
            self.json_name = '{}/new_mimic_cig_{}_chnoch_fov_text_v3.json'.format(
                        data_dir, split)
        else:
            self.json_name = '{}/new_mimic_cig_{}_comp_fov_text_v3.json'.format(
                    data_dir, split)
        # self.json_name = '{}/new_mimic_cig_{}_chnoch_fov_text.json'.format(
        #             data_dir, split)

        self.json_name = self.json_name.replace('_comp_fov_text_v3', self.text_name)

        with open(self.json_name , 'r') as f:
            self.json_file = json.load(f)

        print('[*] {} loaded successfully'.format(self.json_name))   
            
        # load studies and study to text mapping
        self.tokenizer = BertTokenizer.from_pretrained(self.token_pretrained_model)

        self.max_words = max_words
        
        if self.use_kd_get_item:
            self.collate_fn = multimodal_collate_fn_v3
        else:
            self.collate_fn = multimodal_collate_fn_v2
        
    def __len__(self):
        return len(self.json_file['reports_sent'])

    def get_caption(self, report):
        
        if self.empty_fu_sent_dummy:
            report = 'clinical correlation required'
        if len(report) == 0:
            print("no sentence for path")
            #raise Exception("no sentence for path")

        # separate different sentences
        report = list(filter(lambda x: x != "", report))
        sent = " ".join(report)

        tokens = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=self.max_words,
            return_token_type_ids=True,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )

        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len
    
    def get_caption_text_mask(self, report):
        
        report = list(filter(lambda x: x != "", report))
        sent = " ".join(report)

        tokenized_text = self.tokenizer.tokenize(sent)
        
        for i in range(len(tokenized_text)):
            if self.random_erase > random.random():
                tokenized_text[i] = '[MASK]'
                
        # Add the [CLS] and [SEP] tokens
        tokenized_text = ['[CLS]'] + tokenized_text + ['[SEP]']
        # Convert tokenized sentence to input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        input_ids = input_ids[:self.max_words] + [0] * (self.max_words - len(input_ids))
        # Create the attention mask and token type ids
        attention_mask = [1 if i != 0 else 0 for i in input_ids]
        token_type_ids = [0] * len(input_ids)
        
        tokens_dict = {"input_ids" : torch.tensor(input_ids).unsqueeze(0),
               "token_type_ids": torch.tensor(token_type_ids).unsqueeze(0),
               "attention_mask": torch.tensor(attention_mask).unsqueeze(0)
              }

        x_len = len([t for t in tokens_dict["input_ids"][0] if t != 0])

        return tokens_dict, x_len

    def __getitem__(self, index):
        
        if self.only_fu_text_name and self.use_kd_get_item:
            return self.fu_kd_get_item(index)
        elif self.only_base_text_name:
            return self.base_get_item(index)
        elif self.only_fu_text_name:
            return self.fu_get_item(index)
        return self.base_fu_get_item(index)
    
    
    def base_fu_get_item(self, index):

        imgs = self.json_file['imgs'][index]
        change_labels = self.json_file['change_labels'][index]
        disease_labels = label_process(self.json_file, index, self.d_num)
        disease_labels = [np.array(disease_labels[0]).astype(np.float),
                          np.array(disease_labels[1]).astype(np.float)]
        reports = self.json_file['reports_sent'][index][0] + self.json_file['reports_sent'][index][1]
        patient_id = self.json_file['patient_id'][index]
        fov = self.json_file['fov'][index]
               
        report_keys = self.json_file['imgs'][index][0].split('/')[-2] + '_' + self.json_file['imgs'][index][0].split('/')[-1].split('.dcm')[0] \
                + '_' + self.json_file['imgs'][index][1].split('/')[-2] + '_' + self.json_file['imgs'][index][1].split('/')[-1].split('.dcm')[0]
    
        caps, cap_len = self.get_caption(reports)
        base_img, pair_img = get_imgs(imgs, fov, self.transforms, multiscale=False)
        return base_img, pair_img, caps, cap_len, reports, change_labels, disease_labels, patient_id, report_keys
    
    def base_get_item(self, index):
        
        imgs = self.json_file['imgs'][index]
        change_labels = self.json_file['change_labels'][index]
        disease_labels = label_process(self.json_file, index, self.d_num)
        disease_labels = [np.array(disease_labels[0]).astype(np.float),
                          np.array(disease_labels[1]).astype(np.float)]
        
        reports = self.json_file['reports_sent'][index][0]
        patient_id = self.json_file['patient_id'][index]
        fov = self.json_file['fov'][index]
               
        report_keys = self.json_file['imgs'][index][1].split('/')[-2] + '_' + self.json_file['imgs'][index][1].split('/')[-1].split('.dcm')[0]
        caps, cap_len = self.get_caption(reports)
        
        if self.empty_fu_sent:
            cap_len = 0
        
        base_img, pair_img = get_imgs(imgs, fov, self.transforms, multiscale=False)
        return base_img, pair_img, caps, cap_len, reports, change_labels, disease_labels, patient_id, report_keys

    def fu_get_item(self, index):
        
        imgs = self.json_file['imgs'][index]
        change_labels = self.json_file['change_labels'][index]
        disease_labels = label_process(self.json_file, index, self.d_num)
        disease_labels = [np.array(disease_labels[0]).astype(np.float),
                          np.array(disease_labels[1]).astype(np.float)]
        
        reports = self.json_file['reports_sent'][index][1]
        patient_id = self.json_file['patient_id'][index]
        fov = self.json_file['fov'][index]
               
        report_keys = self.json_file['imgs'][index][1].split('/')[-2] + '_' + self.json_file['imgs'][index][1].split('/')[-1].split('.dcm')[0]
        if self.random_erase > -10 :
            caps, cap_len = self.get_caption_text_mask(reports)
        else:
            caps, cap_len = self.get_caption(reports)
        
        if self.empty_fu_sent:
            cap_len = 0
        
        base_img, pair_img = get_imgs(imgs, fov, self.transforms, multiscale=False)
        return base_img, pair_img, caps, cap_len, reports, change_labels, disease_labels, patient_id, report_keys
    
    def fu_kd_get_item(self, index):
        
        imgs = self.json_file['imgs'][index]
        change_labels = self.json_file['change_labels'][index]
        disease_labels = label_process(self.json_file, index, self.d_num)
        disease_labels = [np.array(disease_labels[0]).astype(np.float),
                          np.array(disease_labels[1]).astype(np.float)]
        
        reports = self.json_file['reports_sent'][index][1]
        patient_id = self.json_file['patient_id'][index]
        fov = self.json_file['fov'][index]
               
        report_keys = self.json_file['imgs'][index][1].split('/')[-2] + '_' + self.json_file['imgs'][index][1].split('/')[-1].split('.dcm')[0]
        caps_mask, cap_len_mask = self.get_caption_text_mask(reports)
        caps, cap_len = self.get_caption(reports)
        
        if self.empty_fu_sent:
            cap_len = 0
        
        base_img, pair_img = get_imgs(imgs, fov, self.transforms, multiscale=False)
        return base_img, pair_img, caps, cap_len, caps_mask, cap_len_mask, reports, change_labels, disease_labels, patient_id, report_keys
    
def multimodal_collate_fn_v2(batch):
    
    """sort sequence"""
    base_imgs, pair_imgs, cap_len, ids, tokens, attention = [], [], [], [], [], []
    report, change, disease, pid, report_key = [], [], [], [], []
    for b in batch:
        base_img, pair_img, cap, cap_l, report_sent, change_labels, disease_labels, patient_id, report_keys = b
        base_imgs.append(base_img)
        pair_imgs.append(pair_img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        report.append(report_sent)
        change.append(change_labels)
        disease.append(disease_labels)
        pid.append(patient_id)
        report_key.append(report_keys)

    # stack
    base_imgs = torch.stack(base_imgs)
    pair_imgs = torch.stack(pair_imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    change = torch.tensor(np.array(change))
    disease = torch.tensor(np.array(disease))
    pid = np.array(pid)
    report_key = np.array(report_key)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)
    path = np.array(report)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "change_labels": change[sorted_cap_indices],
        "disease_labels": disease[sorted_cap_indices],
        "base_imgs": base_imgs[sorted_cap_indices],
        "pair_imgs": pair_imgs[sorted_cap_indices],
        "pid": pid[sorted_cap_indices],
        "report_key": report_key[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices]
    }
    return return_dict

def multimodal_collate_fn_v3(batch):
    
    """sort sequence"""
    base_imgs, pair_imgs, cap_len, ids, tokens, attention, mask_ids, mask_tokens, mask_attention = [], [], [], [], [], [], [], [], []
    report, change, disease, pid, report_key = [], [], [], [], []
    for b in batch:
        base_img, pair_img, cap, cap_l, caps_mask, cap_len_mask, report_sent, change_labels, disease_labels, patient_id, report_keys = b
        base_imgs.append(base_img)
        pair_imgs.append(pair_img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        
        mask_ids.append(caps_mask["input_ids"])
        mask_tokens.append(caps_mask["token_type_ids"])
        mask_attention.append(caps_mask["attention_mask"])
        
        report.append(report_sent)
        change.append(change_labels)
        disease.append(disease_labels)
        pid.append(patient_id)
        report_key.append(report_keys)

    # stack
    base_imgs = torch.stack(base_imgs)
    pair_imgs = torch.stack(pair_imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    
    mask_ids = torch.stack(mask_ids).squeeze()
    mask_tokens = torch.stack(mask_tokens).squeeze()
    mask_attention = torch.stack(mask_attention).squeeze()
    
    change = torch.tensor(np.array(change))
    disease = torch.tensor(np.array(disease))
    pid = np.array(pid)
    report_key = np.array(report_key)

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)
    path = np.array(report)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        
        "mask_caption_ids": mask_ids[sorted_cap_indices],
        "mask_token_type_ids": mask_tokens[sorted_cap_indices],
        "mask_attention_mask": mask_attention[sorted_cap_indices],
        
        "change_labels": change[sorted_cap_indices],
        "disease_labels": disease[sorted_cap_indices],
        "base_imgs": base_imgs[sorted_cap_indices],
        "pair_imgs": pair_imgs[sorted_cap_indices],
        "pid": pid[sorted_cap_indices],
        "report_key": report_key[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices]
    }
    return return_dict


# if __name__ == "__main__":
    #from .transforms import DataTransforms
    # transform = DataTransforms(is_train=True)
    # dataset = Multimodal(split="train", transform=transform)
    # data = dataset[0]
    # print(data)