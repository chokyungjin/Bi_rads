import json

import numpy as np
import torch
import torch.utils.data as data
from transformers import BertTokenizer

from ._base import BaseDataset
from .utils import get_imgs, label_process


class CXRMultimodal(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 transforms: dict,
                 mean: float,
                 std: float,
                 max_words: int,
                 pretrained_tokenizer: str,
                 d_num=int,
                 text_name = str,
                 ch_noch = bool,
                 gold = bool,
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
        self.ch_noch = ch_noch
        self.gold = gold
        self.token_pretrained_model = pretrained_tokenizer

        if self.ch_noch:
            self.json_name = '{}/new_mimic_cig_{}_chnoch_fov.json'.format(
                        data_dir, split)
            self.json_name_text = self.json_name.replace('_chnoch_fov', '_chnoch_fov_text')
                    
            with open(self.json_name , 'r') as f:
                self.json_file = json.load(f)
                
            with open(self.json_name_text , 'r') as f:
                self.json_file_text = json.load(f)
        elif self.gold:
            self.json_name = '{}mimic_gold_comp_fov.json'.format(data_dir)
            self.json_name_text = self.json_name.replace('_comp_fov', '_comp_fov_text')
                    
            with open(self.json_name , 'r') as f:
                self.json_file = json.load(f)
                
            with open(self.json_name_text , 'r') as f:
                self.json_file_text = json.load(f)
        else:
            self.json_name = '{}/new_mimic_cig_{}_comp_fov.json'.format(
                        data_dir, split)
            self.json_name_text = self.json_name.replace('_comp_fov', self.text_name)
                    
            with open(self.json_name , 'r') as f:
                self.json_file = json.load(f)
                
            with open(self.json_name_text , 'r') as f:
                self.json_file_text = json.load(f)

        print('[*] {} loaded successfully'.format(self.json_name))
        print('[*] {} loaded successfully'.format(self.json_name_text))
                

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.json_file, self.json_file_text
        self.tokenizer = BertTokenizer.from_pretrained(
            self.token_pretrained_model)
        self.max_words = max_words
        
        self.collate_fn = multimodal_collate_fn
        
    def __len__(self):
        return len(self.json_file['reports'])

    def get_caption(self, path):
        series_sents = self.path2sent[path]
        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

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

    def __getitem__(self, index):
        
        if 'fu' in self.json_name_text:
            return self.fu_get_item(index)
        return self.base_fu_get_item(index)
    
    
    def base_fu_get_item(self, index):

        imgs = self.filenames['imgs'][index]
        change_labels = self.filenames['change_labels'][index]
        disease_labels = label_process(self.filenames, index, self.d_num)
        disease_labels = [np.array(disease_labels[0]).astype(np.float), 
                          np.array(disease_labels[1]).astype(np.float)]
        reports = self.filenames['reports'][index]
        patient_id = self.filenames['patient_id'][index]
        fov = self.filenames['fov'][index]
        
        # report_keys = reports[1].split('/')[-1].split('.txt')[0]
        
        report_keys = reports[0].split('/')[-1].split('.txt')[0] \
            + '_' + reports[1].split('/')[-1].split('.txt')[0]
            
        # report_keys = self.filenames['imgs'][index][0].split('/')[-2] + '_' + self.filenames['imgs'][index][0].split('/')[-1].split('.dcm')[0] \
        #         + '_' + self.filenames['imgs'][index][1].split('/')[-2] + '_' + self.filenames['imgs'][index][1].split('/')[-1].split('.dcm')[0]
    
        caps, cap_len = self.get_caption(report_keys)
        base_img, pair_img = get_imgs(imgs, fov, self.transforms, multiscale=False)
        return base_img, pair_img, caps, cap_len, reports, change_labels, disease_labels, patient_id, report_keys
    
    def fu_get_item(self, index):
        
        sid = list(self.path2sent.keys())[index]    
        fu_text_paths = [i[1] for i in self.filenames['reports']]
        
        for i, element in enumerate(fu_text_paths):
            if sid in element:
                index = i
                break
            
        imgs = self.filenames['imgs'][index]
        change_labels = self.filenames['change_labels'][index]
        # disease_labels = label_process(self.filenames['disease_labels'][index])  # Todo
        reports = self.filenames['reports'][index]
        patient_id = self.filenames['patient_id'][index]
        fov = self.filenames['fov'][index]

        disease_labels = change_labels # Todo

        report_keys = sid
        caps, cap_len = self.get_caption(report_keys)
        base_img, pair_img = get_imgs(imgs, fov, self.transforms, multiscale=False)
        return base_img, pair_img, caps, cap_len, reports, change_labels, disease_labels, patient_id, report_keys


def multimodal_collate_fn(batch):
    
    """sort sequence"""
    base_imgs, pair_imgs, cap_len, ids, tokens, attention = [], [], [], [], [], []
    report_path, change, disease, pid, report_key = [], [], [], [], []
    for b in batch:
        base_img, pair_img, cap, cap_l, report_paths, change_labels, disease_labels, patient_id, report_keys = b
        base_imgs.append(base_img)
        pair_imgs.append(pair_img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        report_path.append(report_paths)
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

    path = np.array(report_path)

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


# if __name__ == "__main__":
    #from .transforms import DataTransforms
    # transform = DataTransforms(is_train=True)
    # dataset = Multimodal(split="train", transform=transform)
    # data = dataset[0]
    # print(data)