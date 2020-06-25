import torch.utils.data as datautils
from utils.randombatchsampler import RandomBatchSampler
from utils.audiodataset import AudioDataset

def collate_zip(batch):
    r"""
    """
    inputs, labels = zip(*batch)
    adjusted_labels = {}
    for feature_class in labels[0].keys():
        adjusted_labels[feature_class] = [label[feature_class] for label in labels]
    return inputs, adjusted_labels

def make_loader(lang, data_tsv, preproc, batch_size, num_workers=4):
    r"""
    """
    dataset = AudioDataset(lang, data_tsv, preproc, batch_size)
    return datautils.DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=RandomBatchSampler(dataset, batch_size),
                            num_workers=num_workers,
                            collate_fn=collate_zip,
                            drop_last=True)
