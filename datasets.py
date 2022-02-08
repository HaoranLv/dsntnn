import numpy as np
from PIL import Image
import cv2
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from utils import ResziePadding
from torch.utils.data import Dataset
import os
from torch.utils.data.sampler import BatchSampler

class TVDataset(Dataset):
    def __init__(self, root_dir, transform, is_train=True, imgsz=512):
        self.train=is_train
        self.transform=transform
        self.imgsz=imgsz
        self.root_dir = root_dir
        if self.train:
            self.imgs=os.listdir(os.path.join(self.root_dir,'images','train'))
            self.labs=os.listdir(os.path.join(self.root_dir,'labels','train'))
        else:
            self.imgs=os.listdir(os.path.join(self.root_dir,'images','val'))
            self.labs=os.listdir(os.path.join(self.root_dir,'labels','val'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.train:
            img=os.path.join(self.root_dir,'images','train',self.imgs[index])
            lab=os.path.join(self.root_dir,'labels','train',self.imgs[index][:-4]+'.npy')
        else:
            img=os.path.join(self.root_dir,'images','val',self.imgs[index])
            lab=os.path.join(self.root_dir,'labels','val',self.imgs[index][:-4]+'.npy')
#         print(cv2.imread(img).shape)
#         labs=np.load(lab)
#         print(labs.shape)

        label=np.load(lab)
#         print(label)
        img_data=cv2.imread(img)
        h,w,_ = img_data.shape
        label[:,0]=label[:,0]*w
        label[:,1]=label[:,1]*h
        image_size=torch.Tensor([w,h])
        target = (torch.Tensor(label) * 2 + 1) / image_size - 1
        img_data=ResziePadding(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB), fixed_side=self.imgsz)
        img = Image.fromarray(img_data, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
#         try:
#             label=np.load(lab)
# #             print(label)
#             img_data=cv2.imread(img)
#             h,w,_ = img_data.shape
#             label[:,0]=label[:,0]*w
#             label[:,1]=label[:,1]*h
#             image_size=torch.Tensor([w,h])
#             target = (torch.Tensor(label) * 2 + 1) / image_size - 1
#             img_data=ResziePadding(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB), fixed_side=self.imgsz)
#             img = Image.fromarray(img_data, mode='RGB')
#             if self.transform is not None:
#                 img = self.transform(img)
#             return img, target
#         except:
#             print(lab)
#             pass
class TV_point(Dataset):

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

class SiameseBLDataset(Dataset):
    def __init__(self, root_dir, transform, is_train=True, imgsz=512, test_rate=0.2):
        self.train=is_train
        self.transform=transform
        self.imgsz=imgsz
        self.root_dir = root_dir
        self.true_imgs=os.listdir(os.path.join(self.root_dir,'true_bl'))
        self.false_imgs=os.listdir(os.path.join(self.root_dir,'false_bl'))
        for files in self.true_imgs:
            if os.path.splitext(files)[-1] != '.jpg':
                self.true_imgs.remove(files)
        for files in self.false_imgs:
            if os.path.splitext(files)[-1] != '.jpg':
                self.false_imgs.remove(files)
        self.false_train, self.false_test=train_test_split(self.false_imgs, test_size=test_rate, random_state=42)
        self.true_train, self.true_test=train_test_split(self.true_imgs, test_size=test_rate, random_state=42)
        if self.train:
            self.train_data = self.true_train+self.false_train
            self.train_labels = [0]*len(self.true_train)+[1]*len(self.false_train)
            self.labels_set = set([0,1])
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_data = self.true_test+self.false_test
            self.test_labels = [0]*len(self.true_test)+[1]*len(self.false_test)
            self.labels_set = set([0,1])
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

        
    def __len__(self):
        if self.train:
#             print(len(self.true_train))
            return len(self.true_train)
        else:
            return len(self.true_test)

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
        
        img1_data=ResziePadding(cv2.cvtColor(cv2.imread(os.path.join(self.root_dir,'bl',img1)),cv2.COLOR_BGR2RGB), fixed_side=self.imgsz)
        img2_data=ResziePadding(cv2.cvtColor(cv2.imread(os.path.join(self.root_dir,'bl',img2)),cv2.COLOR_BGR2RGB), fixed_side=self.imgsz)
        
        # print(img1_data.shape)
        img1 = Image.fromarray(img1_data, mode='RGB')
        img2 = Image.fromarray(img2_data, mode='RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

class BLDataset(Dataset):
    def __init__(self, root_dir, transform, is_train=True, imgsz=512, test_rate=0.2):
        self.train=is_train
        self.transform=transform
        self.imgsz=imgsz
        self.root_dir = root_dir
        self.true_imgs=os.listdir(os.path.join(self.root_dir,'true_bl'))
        self.false_imgs=os.listdir(os.path.join(self.root_dir,'false_bl'))
        for files in self.true_imgs:
            if os.path.splitext(files)[-1] != '.jpg':
                self.true_imgs.remove(files)
        for files in self.false_imgs:
            if os.path.splitext(files)[-1] != '.jpg':
                self.false_imgs.remove(files)
        self.false_train, self.false_test=train_test_split(self.false_imgs, test_size=test_rate, random_state=42)
        self.true_train, self.true_test=train_test_split(self.true_imgs, test_size=test_rate, random_state=42)
        if self.train:
            self.train_data = self.true_train+self.false_train
            self.train_labels = [0]*len(self.true_train)+[1]*len(self.false_train)
            self.labels_set = set([0,1])

        else:
            # generate fixed pairs for testing
            self.test_data = self.true_test+self.false_test
            self.test_labels = [0]*len(self.true_test)+[1]*len(self.false_test)
            self.labels_set = set([0,1])

        
    def __len__(self):
        if self.train:
            return len(self.true_train+self.false_train)
        else:
            return len(self.true_test+self.false_test)

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]
        try:
#             print(os.path.join(self.root_dir,'bl',img))
            img_data=ResziePadding(cv2.cvtColor(cv2.imread(os.path.join(self.root_dir,'bl',img)),cv2.COLOR_BGR2RGB), fixed_side=self.imgsz)
            img = Image.fromarray(img_data, mode='RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, label
        except:
            pass
        

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
