from pytorch_lightning import LightningModule


def is_zero_rank(self: LightningModule):
    if self.global_rank == 0:
        return True
    return False


def add_prefix_to_dict(dict:dict, prefix:str):
    """Add a prefix to all keys in a dict"""
    new_dict = {}
    for key in dict.keys():
        new_dict[prefix + key] = dict[key]
    return new_dict


def key_to_abbreviation(dict:dict):
    """Convert a key to an abbreviation"""
    new_dict = {}
    for key in dict.keys():
        if "Precision" in key:
            new_key = key.replace("Precision", "pre")
            new_dict[new_key] = dict[key]
            continue
        if "Recall" in key:
            new_key = key.replace("Recall", "rec")
            new_dict[new_key] = dict[key]
            continue
        if "F1Score" in key:
            new_key = key.replace("F1Score", "f1")
            new_dict[new_key] = dict[key]
            continue
        if "Accuracy" in key:
            new_key = key.replace("Accuracy", "acc")
            new_dict[new_key] = dict[key]
            continue
        new_dict[key] = dict[key]
    return new_dict
