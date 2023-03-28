import json

import numpy as np

from ._base import BaseDataset
from .utils import get_imgs, label_process


class CXRMultimodal_vision(BaseDataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 transforms: dict,
                 mean: float,
                 std: float,
                 ch_noch: bool = False,
                 gold: bool = False,
                 d_num=int,
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
        self.gold = gold
        self.ch_noch = ch_noch
        # print(self.gold)
        
        if self.gold:
            json_name = '{}mimic_gold_comp_fov.json'.format(data_dir)
        elif self.ch_noch:
            json_name = '{}new_mimic_cig_{}_chnoch_fov_text_v3.json'.format(data_dir, split)
        else:
            json_name = '{}new_mimic_cig_{}_comp_fov_text_v3.json'.format(data_dir, split)
            
        # print('[*] Loading Json from {}'.format(json_name)) 

        with open(json_name , 'r') as f:
            self.json_file = json.load(f)

        print('[*] {} loaded successfully'.format(json_name))     
        self.filenames = self.json_file
   
    def __len__(self):
        return len(self.filenames['reports'])
      
    def __getitem__(self, index):

        imgs = self.filenames['imgs'][index]
        change_labels = self.filenames['change_labels'][index]
        disease_labels = label_process(self.filenames, index, self.d_num)
        disease_labels = [np.array(disease_labels[0]).astype(np.float), np.array(disease_labels[1]).astype(np.float)]
        reports = self.filenames['reports'][index]
        # patient_id = self.filenames['patient_id'][index]
        patient_id = [self.filenames['imgs'][index][0].split('files')[-1], self.filenames['imgs'][index][1].split('files')[-1]]
        fov = self.filenames['fov'][index]
  
        base_img, pair_img = get_imgs(imgs, fov, self.transforms, multiscale=False)

        return base_img, pair_img, reports, change_labels, disease_labels, patient_id

