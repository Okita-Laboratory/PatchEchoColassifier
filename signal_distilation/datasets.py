import os
import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split

import augment4sig as aug

class CustomTransform:
    def __init__(self, signal_length, is_training=True, jitter_strength=0.1, flip_prob=0.5):
        self.signal_length = signal_length
        self.is_training = is_training
        self.jitter_strength = jitter_strength
        self.flip_prob = flip_prob

    def __call__(self, x):
        if self.is_training:
            # Add random jitter
            from scipy import signal
            y = np.zeros((x.shape[0], self.signal_length))
            for i in range(x.shape[0]):
                y[i] = signal.resample(x[i], self.signal_length)
            y = torch.from_numpy(y.copy()).float()
            jitter = torch.randn_like(y) * self.jitter_strength
            y = y + jitter

            # Random flip
            if random.random() < self.flip_prob:
                y = torch.flip(y, dims=[-1])

            # Ensure the signal length is correct
            if y.shape[-1] != self.signal_length:
                y = nn.functional.interpolate(x.unsqueeze(0), size=self.signal_length, mode='linear', align_corners=False).squeeze(0)

        return y

def create_transform(signal_length, is_training=True, jitter_strength=0.1, flip_prob=0.5):
    return CustomTransform(
        signal_length=signal_length,
        is_training=is_training,
        jitter_strength=jitter_strength,
        flip_prob=flip_prob
    )

class TriaxialSignalDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data :  containing triaxial signal data of shape (num_samples, 3, signal_length)
            labels :  containing labels of shape (num_samples,)
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        if self.transform is not None:
            signal = self.transform(signal)
        return signal, label



def build_dataset(args):
    if args.data == "SHL2024":   
        acc_x = np.loadtxt("../../SHL/SHL_2024/train/Hips/Acc_x.txt")
        acc_y = np.loadtxt("../../SHL/SHL_2024/train/Hips/Acc_y.txt")
        acc_z = np.loadtxt("../../SHL/SHL_2024/train/Hips/Acc_z.txt")
        label = np.loadtxt("../../SHL/SHL_2024/train/Hips/Label.txt", dtype=np.int64)
        label = label[:,250]
        label = np.array([i-1 for i in label])
    
        acc_xyz = np.stack([acc_x, acc_y, acc_z], axis=1)
        nb_classes = 8
    elif args.data == "SHL2023":   
        acc_xyz = np.load("../../SHL/SHL_2023/100hz_5.0s_overlap0.0s/train_clear/Hips_Acc.npy")
        label = np.load("../../SHL/SHL_2023/100hz_5.0s_overlap0.0s/train_clear/Hips_Label.npy")

        label= np.array([i-1 for i in label])

        nb_classes = 8
    elif args.data == "SHL2023_test":   
        #500サンプル(ウィンドウ幅), 100Hz(1秒で100個)*5秒, 196104フレーム
        acc_xyz = np.load("../../SHL/SHL_2023/100hz_5.0s_overlap0.0s/test_clear/Hips_Acc.npy")
        label = np.load("../../SHL/SHL_2023/100hz_5.0s_overlap0.0s/test_clear/Hips_Label.npy")

        label= np.array([i-1 for i in label])

        nb_classes = 8
    elif args.data == "ADL":
        acc = np.load("/home/masaharu/remote/ssl-data/downstream/adl_30hz_clean/X.npy")
        label = np.load("/home/masaharu/remote/ssl-data/downstream/adl_30hz_clean/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 5
    elif args.data == "PAMAP":
        acc = np.load("/home/SHL/ssl-data/downstream/pamap_100hz_w10_o5/X.npy")
        label = np.load("/home/SHL/ssl-data/downstream/pamap_100hz_w10_o5/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 8
    elif args.data == "REALWORLD":
        acc = np.load("/home/SHL/ssl-data/downstream/realworld_30hz_clean/X.npy")
        label = np.load("/home/SHL/ssl-data/downstream/realworld_30hz_clean/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 8
    elif args.data == "WISDM":
        acc = np.load("/home/SHL/ssl-data/downstream/wisdm_30hz_clean/X.npy")
        label = np.load("/home/SHL/ssl-data/downstream/wisdm_30hz_clean/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 18
    elif args.data == "CAPTURE":
        acc = np.load("/home/SHL/ssl-data/downstream/capture24_30hz_full/X.npy")
        label = np.load("/home/SHL/ssl-data/downstream/capture24_30hz_full/Y.npy")
        acc_xyz = np.transpose(acc, (0, 2, 1))
        unique_label = np.unique(label)
        label2int = {label: idx for idx, label in enumerate(unique_label)}
        label = np.array([label2int[a] for a in label])
        nb_classes = 4
    
    if not args.data == "SHL2023_test": 
        index = np.array(range(label.shape[0]))
        train_label, val_label, train_index, val_index = train_test_split(label, index, train_size=0.9, random_state=0)
    else:
        index = np.array(range(label.shape[0]))
        train_label, val_label, train_index, val_index = train_test_split(label, index, train_size=0.8, random_state=0)
        val_label, _, val_index, _ = train_test_split(val_label, val_index, train_size=0.5, random_state=0)
        
    train_data = TriaxialSignalDataset(data=acc_xyz[train_index], labels=label[train_index],transform=build_transform(is_train=True, args=args))
    val_data = TriaxialSignalDataset(data=acc_xyz[val_index], labels=label[val_index], transform=build_transform(is_train=False, args=args))
    

    return train_data, val_data, nb_classes


def build_transform(is_train=True, args=None):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            signal_length=args.input_size,
            is_training=True,
            jitter_strength=0.1,
            flip_prob = 0.5 if args.flip_on else 0.0
            )
        
        if args.student == "DeepConvLSTM":
            return transforms.Compose([transform, aug.Signal4DeepConvLSTM()])

        return transform

    t = []
    
    t.append(aug.Resample(num_sample=496))
    t.append(aug.Signal2Tensor())
    
    if args.student == "DeepConvLSTM":
        return t.append(aug.Signal4DeepConvLSTM())
    
    return transforms.Compose(t)