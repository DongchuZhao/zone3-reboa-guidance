# data/patient_dataset.py
import torch.utils.data as data

class PatientDataset(data.Dataset):
    def __init__(self, ids, sample_loader):
        self.ids = list(ids)
        self.sample_loader = sample_loader

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.sample_loader(self.ids[idx])
