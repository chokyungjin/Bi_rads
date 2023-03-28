import json 
import numpy as np
import tqdm
import re
from nltk.tokenize import RegexpTokenizer

from src.datamodules.components.mimic_prefunc import text_to_sectionize
from src.datamodules.components.utils import get_imgs, label_process

def load_text_data(json_file):

    path2sent = create_path_2_fu_sent_mapping(json_file)
    return path2sent

def create_path_2_sent_mapping(json_file):
    sent_lens, num_sents = [], []
    path2sent = {}

    for i in tqdm.tqdm(range(len(json_file['reports']))):
        with open(json_file['reports'][i][0], 'r') as fp:
            b_text = ''.join(fp.readlines())

        with open(json_file['reports'][i][1], 'r') as fp:
            fu_text = ''.join(fp.readlines())

        pair_imgs = json_file['reports'][i][0].split('/')[-1].split('.txt')[0] \
            + '_' + json_file['reports'][i][1].split('/')[-1].split('.txt')[0]
        # split text into sections
        b_study_sectioned = text_to_sectionize(b_text)
        fu_study_sectioned = text_to_sectionize(fu_text)

        b_captions = ""
        if b_study_sectioned["impression"] == None:
            b_captions = ""
        else:
            b_captions += b_study_sectioned["impression"]
        b_captions += " "
        if b_study_sectioned["findings"] == None:
            b_captions += ""
        else:
            b_captions += b_study_sectioned["findings"]    
        b_captions += " "
        if b_study_sectioned["last_paragraph"] == None:
            b_captions += ""
        else:
            b_captions += b_study_sectioned["last_paragraph"]

        fu_captions = ""
        if fu_study_sectioned["impression"] == None:
            fu_captions = ""
        else:
            fu_captions += fu_study_sectioned["impression"]
        fu_captions += " "
        if fu_study_sectioned["findings"] == None:
            fu_captions += ""
        else:
            fu_captions += fu_study_sectioned["findings"]
        fu_captions += " "
        if fu_study_sectioned["last_paragraph"] == None:
            fu_captions += ""
        else:
            fu_captions += fu_study_sectioned["last_paragraph"]

        # use space instead of newline
        captions = b_captions + fu_captions

        # use space instead of newline
        captions = b_captions + fu_captions
        captions = captions.replace("\n", " ")

        # split sentences
        splitter = re.compile("[0-9]+\.")
        captions = splitter.split(captions)
        captions = [point.split(".") for point in captions]
        captions = [sent for point in captions for sent in point]

        cnt = 0
        study_sent = []
        # create tokens from captions
        for cap in captions:
            if len(cap) == 0:
                continue

            cap = cap.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(cap.lower())
            # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 0:
                study_sent.append(" ".join(included_tokens))

            cnt += len(included_tokens)

        if cnt >= 3:
            sent_lens.append(cnt)
            num_sents.append(len(study_sent))
            path2sent[pair_imgs] = study_sent

    # get report word/setence statistics
    sent_lens = np.array(sent_lens)
    num_sents = np.array(num_sents)
    return path2sent


def create_path_2_fu_sent_mapping(json_file):
    sent_lens, num_sents = [], []
    path2sent = {}

    for i in tqdm.tqdm(range(len(json_file['reports']))):
        
        with open(json_file['reports'][i][1], 'r') as fp:
            fu_text = ''.join(fp.readlines())

        pair_imgs = json_file['reports'][i][1].split('/')[-1].split('.txt')[0]
        # split text into sections
        fu_study_sectioned = text_to_sectionize(fu_text)

        fu_captions = ""
        
        if fu_study_sectioned["impression"] == None:
            fu_captions = ""
        else:
            fu_captions += fu_study_sectioned["impression"]
        fu_captions += " "
        if fu_study_sectioned["findings"] == None:
            fu_captions += ""
        else:
            fu_captions += fu_study_sectioned["findings"]
        fu_captions += " "
        if fu_study_sectioned["last_paragraph"] == None:
            fu_captions += ""
        else:
            fu_captions += fu_study_sectioned["last_paragraph"]

        captions = fu_captions
        captions = captions.replace("\n", " ")

        # split sentences
        splitter = re.compile("[0-9]+\.")
        captions = splitter.split(captions)
        captions = [point.split(".") for point in captions]
        captions = [sent for point in captions for sent in point]

        cnt = 0
        study_sent = []
        # create tokens from captions
        for cap in captions:
            if len(cap) == 0:
                continue

            cap = cap.replace("\ufffd\ufffd", " ")
            # picks out sequences of alphanumeric characters as tokens
            # and drops everything else
            tokenizer = RegexpTokenizer(r"\w+")
            tokens = tokenizer.tokenize(cap.lower())
            # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
            if len(tokens) <= 1:
                continue

            # filter tokens for current sentence
            included_tokens = []
            for t in tokens:
                t = t.encode("ascii", "ignore").decode("ascii")
                if len(t) > 0:
                    included_tokens.append(t)

            if len(included_tokens) > 0:
                study_sent.append(" ".join(included_tokens))

            cnt += len(included_tokens)
            
        if cnt >= 3:
            sent_lens.append(cnt)
            num_sents.append(len(study_sent))
            path2sent[pair_imgs] = study_sent
            

    # get report word/setence statistics
    sent_lens = np.array(sent_lens)
    num_sents = np.array(num_sents)
    return path2sent


def main():
    
    data_dir = '/mnt/nas252/Kyungjin.Cho/_jykim/MuSiC-ViT-main/MuSiC-ViT-main/json'
    split = ['train', 'valid', 'test']
    d_num = 14
    
    for idx in split:
        print(idx)
        json_name = '{}/new_mimic_cig_{}_comp_fov.json'.format(
                        data_dir, idx)
        with open(json_name , 'r') as f:
            json_file = json.load(f)

        path2sent = load_text_data(json_file)

        dump_name = json_name.replace('_fov.json', '_fov_fu_text.json')

        with open(dump_name, "w") as f:
            json.dump(path2sent, f)

if __name__ == '__main__':
    main()
