import numpy as np
import torch
import sys
import os
import pathlib
import urllib.request
import torchaudio
import tarfile
import zipfile
import math
import csv
import sktime.utils.load_data
import collections as co
import controldiffeq

def normalize(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # numerator = data-np.min(data)
    # denominator = np.max(data) - np.min(data)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def to_tensor(data):
    return torch.from_numpy(data).float()


def batch_generator(dataset, batch_size):
    # import pdb;pdb.set_trace()
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch

class SineDataset(torch.utils.data.Dataset):
    def __init__(self, no, seq_len,dim,data_name,missing_rate):
        base_loc = here / 'datasets'
        loc = here / 'datasets'/(data_name+str(missing_rate))
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
            self.data = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.data = np.array(self.data)
            self.size = len(self.data)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)    
            if not os.path.exists(loc):
                os.mkdir(loc)
            self.data = list()
            self.original_sample = list()
            generator = torch.Generator().manual_seed(56789)
            for i in range(no):
                tmp = list()
                for k in range(dim):
                    freq = np.random.uniform(0, 0.1)
                    phase = np.random.uniform(0, 0.1)
                    tmp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                    tmp.append(tmp_data)
                tmp = np.transpose(np.asarray(tmp))
                tmp = (tmp + 1) * 0.5
                self.original_sample.append(tmp.copy())
                removed_points = torch.randperm(tmp.shape[0], generator=generator)[:int(tmp.shape[0] * missing_rate)].sort().values
                tmp[removed_points] = float('nan')
                idx = np.array(range(seq_len)).reshape(-1,1)
                tmp = np.concatenate((tmp,idx),axis=1)
                self.data.append(tmp)
            self.data = np.array(self.data)
            self.original_sample = np.array(self.original_sample)
            norm_data_tensor = torch.Tensor(self.data[:,:,:-1]).float().cuda()
            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.data[:,:,-1][:,-1]).float()
            self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
            self.original_sample = torch.tensor(self.original_sample)
            self.data = torch.tensor(self.data)
            save_data(loc,data=self.data,
                    original_data = self.original_sample,
                    train_a=self.train_coeffs[0], 
                    train_b=self.train_coeffs[1], 
                    train_c=self.train_coeffs[2],
                    train_d=self.train_coeffs[3],
                    )
            self.original_sample = np.array(self.original_sample)
            self.data = np.array(self.data)
            self.size = len(self.data)
    def __getitem__(self, batch_size):
        dataset_size = len(self.data)
        idx = torch.randperm(dataset_size)
        batch_idx = idx[:batch_size]
        original_batch = torch.stack([to_tensor(self.original_sample[i]) for i in batch_idx])
        batch = torch.stack([to_tensor(self.data[i]) for i in batch_idx])
        a, b, c, d = self.train_coeffs
        batch_a = torch.stack([a[i] for i in batch_idx])
        batch_b = torch.stack([b[i] for i in batch_idx])
        batch_c = torch.stack([c[i] for i in batch_idx])
        batch_d = torch.stack([d[i] for i in batch_idx])
        batch_coeff = (batch_a, batch_b, batch_c, batch_d)
        self.sample = {'data': batch , 'inter': batch_coeff, 'original_data':original_batch}
        return self.sample
    def __len__(self):
        return len(self.data)

class SineDataset_t(torch.utils.data.Dataset):
    def __init__(self, no, seq_len, dim, missing_rate = 0.0):
        self.data = list()
        generator = torch.Generator().manual_seed(56789)
        for i in range(no):
            tmp = list()
            for k in range(dim):
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                tmp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                tmp.append(tmp_data)
            tmp = np.transpose(np.asarray(tmp))
            tmp = (tmp + 1) * 0.5
            removed_points = torch.randperm(tmp.shape[0], generator=generator)[:int(tmp.shape[0] * missing_rate)].sort().values
            tmp[removed_points] = float('nan')
            idx = np.array(range(seq_len)).reshape(-1,1)
            tmp = np.concatenate((tmp,idx),axis=1)
            self.data.append(tmp)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

here = pathlib.Path(__file__).resolve().parent


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors

class MujocoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_name, missing_rate = 0.0):
        # import pdb;pdb.set_trace()
        base_loc = here / 'datasets'
        loc = here / 'datasets'/(data_name+str(missing_rate))
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)    
            if not os.path.exists(loc):
                os.mkdir(loc)

            tensors = load_data(data_path)
            time = tensors['train_X'][:,:,:1].cpu().numpy()
            data = tensors['train_X'][:,:,1:].reshape(-1,14).cpu().numpy()

            self.original_sample = []
            norm_data = normalize(data)
            norm_data = norm_data.reshape(4620,24,14)
            idx = torch.randperm(len(norm_data))
            
            for i in range(len(norm_data)):
                self.original_sample.append(norm_data[idx[i]].copy())
            self.X_mean = np.mean(np.array(self.original_sample),axis=0).reshape(1,np.array(self.original_sample).shape[1],np.array(self.original_sample).shape[2])
            generator = torch.Generator().manual_seed(56789)
            for i in range(len(norm_data)):
                removed_points = torch.randperm(norm_data[i].shape[0], generator=generator)[:int(norm_data[i].shape[0] * missing_rate)].sort().values
                norm_data[i][removed_points] = float('nan')    
            norm_data = np.concatenate((norm_data,time),axis=2)
            self.samples = []
            for i in range(len(norm_data)):
                self.samples.append(norm_data[idx[i]])
                        
            self.samples = np.array(self.samples)
            
            norm_data_tensor = torch.Tensor(self.samples[:,:,:-1]).float().cuda()
            # time 0 ~ seq 까지 다 해줌 
            # 학습할때는 final index만 가져감 
            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.samples[:,:,-1][:,-1]).float()
            self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)
            save_data(loc,data=self.samples,
                    original_data = self.original_sample,
                    train_a=self.train_coeffs[0], 
                    train_b=self.train_coeffs[1], 
                    train_c=self.train_coeffs[2],
                    train_d=self.train_coeffs[3],
                    )
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

    def __getitem__(self, batch_size):
        # import pdb;pdb.set_trace()
        dataset_size = len(self.samples)
        idx = torch.randperm(dataset_size)
        batch_idx = idx[:batch_size]
        original_batch = torch.stack([to_tensor(self.original_sample[i]) for i in batch_idx])
        batch = torch.stack([to_tensor(self.samples[i]) for i in batch_idx])

        # batch _idx -> batch 만큼 가져고 
        a, b, c, d = self.train_coeffs
        batch_a = torch.stack([a[i] for i in batch_idx])
        batch_b = torch.stack([b[i] for i in batch_idx])
        batch_c = torch.stack([c[i] for i in batch_idx])
        batch_d = torch.stack([d[i] for i in batch_idx])

        batch_coeff = (batch_a, batch_b, batch_c, batch_d)
        
        self.sample = {'data': batch , 'inter': batch_coeff, 'original_data':original_batch}

        return self.sample # self.samples[index]

    def __len__(self):
        return len(self.samples)

class MujocoDataset_t(torch.utils.data.Dataset):
    def __init__(self, data_path, missing_rate = 0.0):
        # import pdb;pdb.set_trace()
        tensors = load_data(data_path)
        time = tensors['train_X'][:,:,:1].cpu().numpy()
        data = tensors['train_X'][:,:,1:].reshape(-1,14).cpu().numpy()
        self.original_sample = []
        norm_data = normalize(data)
        norm_data = norm_data.reshape(4620,24,14)
        idx = torch.randperm(len(norm_data))
        for i in range(len(norm_data)):
            self.original_sample.append(norm_data[idx[i]].copy())
        # import pdb;pdb.set_trace()
        self.X_mean = np.mean(np.array(self.original_sample),axis=0).reshape(1,np.array(self.original_sample).shape[1],np.array(self.original_sample).shape[2])

        generator = torch.Generator().manual_seed(56789)
        for i in range(len(norm_data)):
            removed_points = torch.randperm(norm_data[i].shape[0], generator=generator)[:int(norm_data[i].shape[0] * missing_rate)].sort().values
            norm_data[i][removed_points] = float('nan')    
        # total_length = len(norm_data)
        # idx = np.array(range(total_length)).reshape(-1,1)
        norm_data = np.concatenate((norm_data,time),axis=2)#맨 뒤에 관측시간에 대한 정보 저장

        # seq_data = []
        # for i in range(len(norm_data) - seq_len + 1):
        #     x = norm_data[i : i + seq_len]
        #     seq_data.append(x)
        self.samples = []
        for i in range(len(norm_data)):
            self.samples.append(norm_data[idx[i]])

    def __getitem__(self, index):
        return self.samples[index]#,self.original_sample[index]

    def __len__(self):
        return len(self.samples)

# class MujocoDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, missing_rate = 0.0):
#         # import pdb;pdb.set_trace()
#         tensors = load_data(data_path)
#         time = tensors['train_X'][:,:,:1].cpu().numpy()
#         data = tensors['train_X'][:,:,1:].reshape(-1,14).cpu().numpy()
#         # data = np.loadtxt(data_path, delimiter=",", skiprows=1)
#         # total_length = len(data)
#         # data = data[::-1]
        
#         # self.min_val = np.min(data, 0)
#         # self.max_val = np.max(data, 0) - np.min(data, 0)
#         # import pdb;pdb.set_trace()
#         norm_data = normalize(data)
#         norm_data = norm_data.reshape(4620,24,14)
#         generator = torch.Generator().manual_seed(56789)
#         removed_points = torch.randperm(norm_data.shape[0], generator=generator)[:int(norm_data.shape[0] * missing_rate)].sort().values
#         norm_data[removed_points] = float('nan')
#         total_length = len(norm_data)
#         idx = np.array(range(total_length)).reshape(-1,1)
#         norm_data = np.concatenate((norm_data,time),axis=2)#맨 뒤에 관측시간에 대한 정보 저장

#         # seq_data = []
#         # for i in range(len(norm_data) - seq_len + 1):
#         #     x = norm_data[i : i + seq_len]
#         #     seq_data.append(x)
#         self.samples = []
#         idx = torch.randperm(len(norm_data))
#         for i in range(len(norm_data)):
#             self.samples.append(norm_data[idx[i]])

#     def __getitem__(self, index):
#         return self.samples[index]

#     def __len__(self):
#         return len(self.samples)

def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')

import pathlib
here = pathlib.Path(__file__).resolve().parent


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len, data_name, missing_rate = 0.0):
        # import pdb;pdb.set_trace()
        base_loc = here / 'datasets'
        loc = here / 'datasets'/(data_name+str(missing_rate))
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)    
            if not os.path.exists(loc):
                os.mkdir(loc)
            data = np.loadtxt(data_path, delimiter=",", skiprows=1)
            total_length = len(data)
            data = data[::-1]
            self.min_val = np.min(data, 0)
            self.max_val = np.max(data, 0) - np.min(data, 0)

            self.original_sample = []
            norm_data = normalize(data)
            ori_seq_data = []
            for i in range(len(norm_data) - seq_len + 1):
                x = norm_data[i : i + seq_len].copy()
                ori_seq_data.append(x)
            idx = torch.randperm(len(ori_seq_data))
            for i in range(len(ori_seq_data)):
                self.original_sample.append(ori_seq_data[idx[i]])
            self.X_mean = np.mean(np.array(self.original_sample),axis=0).reshape(1,np.array(self.original_sample).shape[1],np.array(self.original_sample).shape[2])
            generator = torch.Generator().manual_seed(56789)
            removed_points = torch.randperm(norm_data.shape[0], generator=generator)[:int(norm_data.shape[0] * missing_rate)].sort().values
            norm_data[removed_points] = float('nan')
            total_length = len(norm_data)
            index = np.array(range(total_length)).reshape(-1,1)
            norm_data = np.concatenate((norm_data,index),axis=1)#맨 뒤에 관측시간에 대한 정보 저장
            seq_data = []
            for i in range(len(norm_data) - seq_len + 1):
                x = norm_data[i : i + seq_len]
                seq_data.append(x)
            self.samples = []
            for i in range(len(seq_data)):
                self.samples.append(seq_data[idx[i]])
            
            self.samples = np.array(self.samples)
            
            norm_data_tensor = torch.Tensor(self.samples[:,:,:-1]).float().cuda()
            # time 0 ~ seq 까지 다 해줌 
            # 학습할때는 final index만 가져감 
            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.samples[:,:,-1][:,-1]).float()
            self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)
            save_data(loc,data=self.samples,
                    original_data = self.original_sample,
                    train_a=self.train_coeffs[0], 
                    train_b=self.train_coeffs[1], 
                    train_c=self.train_coeffs[2],
                    train_d=self.train_coeffs[3],
                    )
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

        # save_data()

    def __getitem__(self, batch_size):
        # import pdb;pdb.set_trace()
        dataset_size = len(self.samples)
        idx = torch.randperm(dataset_size)
        batch_idx = idx[:batch_size]
        original_batch = torch.stack([to_tensor(self.original_sample[i]) for i in batch_idx])
        batch = torch.stack([to_tensor(self.samples[i]) for i in batch_idx])

        # batch _idx -> batch 만큼 가져고 
        a, b, c, d = self.train_coeffs
        batch_a = torch.stack([a[i] for i in batch_idx])
        batch_b = torch.stack([b[i] for i in batch_idx])
        batch_c = torch.stack([c[i] for i in batch_idx])
        batch_d = torch.stack([d[i] for i in batch_idx])

        batch_coeff = (batch_a, batch_b, batch_c, batch_d)
        
        self.sample = {'data': batch , 'inter': batch_coeff, 'original_data':original_batch}

        return self.sample # self.samples[index]

    def __len__(self):
        return len(self.samples)

class TimeDataset_j(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        total_length = len(data)
        data = data[::-1]
        self.min_val = np.min(data, 0)
        self.max_val = np.max(data, 0) - np.min(data, 0)
        norm_data = normalize(data)
        total_length = len(norm_data)
        idx = np.array(range(total_length)).reshape(-1,1)
        norm_data = np.concatenate((norm_data,idx),axis=1)#맨 뒤에 관측시간에 대한 정보 저장
        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])
    def __getitem__(self, index):
        return self.samples[index]
    def __len__(self):
        return len(self.samples)

class TimeDataset_t(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len, missing_rate=0.0):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        total_length = len(data)
        data = data[::-1]
        
        self.min_val = np.min(data, 0)
        self.max_val = np.max(data, 0) - np.min(data, 0)
        self.original_sample = []
        # import pdb;pdb.set_trace()
        norm_data = normalize(data)
        ori_seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len].copy()
            ori_seq_data.append(x)
        idx = torch.randperm(len(ori_seq_data))
        # import pdb;pdb.set_trace()
        for i in range(len(ori_seq_data)):
            self.original_sample.append(ori_seq_data[idx[i]])
        self.X_mean = np.mean(np.array(self.original_sample),axis=0).reshape(1,np.array(self.original_sample).shape[1],np.array(self.original_sample).shape[2])
        # import pdb;pdb.set_trace()
        generator = torch.Generator().manual_seed(56789)
        removed_points = torch.randperm(norm_data.shape[0], generator=generator)[:int(norm_data.shape[0] * missing_rate)].sort().values
        norm_data[removed_points] = float('nan')
        total_length = len(norm_data)
        index = np.array(range(total_length)).reshape(-1,1)
        norm_data = np.concatenate((norm_data,index),axis=1)#맨 뒤에 관측시간에 대한 정보 저장
        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])

    def __getitem__(self, index):
        return self.samples[index]#,self.original_sample[index]

    def __len__(self):
        return len(self.samples)

class TimeDataset_ori(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len, missing_rate=0.0):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        total_length = len(data)
        data = data[::-1]
        
        self.min_val = np.min(data, 0)
        self.max_val = np.max(data, 0) - np.min(data, 0)
        norm_data = normalize(data)
        generator = torch.Generator().manual_seed(56789)
        removed_points = torch.randperm(norm_data.shape[0], generator=generator)[:int(norm_data.shape[0] * missing_rate)].sort().values
        norm_data[removed_points] = float('nan')
        total_length = len(norm_data)
        idx = np.array(range(total_length)).reshape(-1,1)
        norm_data = np.concatenate((norm_data,idx),axis=1)#맨 뒤에 관측시간에 대한 정보 저장
        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class TimeDataset_irregular(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len, data_name, missing_rate = 0.0):
        # import pdb;pdb.set_trace()
        base_loc = here / 'datasets'
        loc = here / 'datasets'/(data_name+str(missing_rate))
        if os.path.exists(loc):
            tensors = load_data(loc)
            self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
            self.samples = tensors['data']
            self.original_sample = tensors['original_data']
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)
        else:
            if not os.path.exists(base_loc):
                os.mkdir(base_loc)    
            if not os.path.exists(loc):
                os.mkdir(loc)
            data = np.loadtxt(data_path, delimiter=",", skiprows=1)
            total_length = len(data)
            data = data[::-1]
            self.min_val = np.min(data, 0)
            self.max_val = np.max(data, 0) - np.min(data, 0)

            self.original_sample = []
            norm_data = normalize(data)
            ori_seq_data = []
            for i in range(len(norm_data) - seq_len + 1):
                x = norm_data[i : i + seq_len].copy()
                ori_seq_data.append(x)
            idx = torch.randperm(len(ori_seq_data))
            for i in range(len(ori_seq_data)):
                self.original_sample.append(ori_seq_data[idx[i]])
            self.X_mean = np.mean(np.array(self.original_sample),axis=0).reshape(1,np.array(self.original_sample).shape[1],np.array(self.original_sample).shape[2])
            generator = torch.Generator().manual_seed(56789)
            removed_points = torch.randperm(norm_data.shape[0], generator=generator)[:int(norm_data.shape[0] * missing_rate)].sort().values
            norm_data[removed_points] = float('nan')
            total_length = len(norm_data)
            index = np.array(range(total_length)).reshape(-1,1)
            norm_data = np.concatenate((norm_data,index),axis=1)#맨 뒤에 관측시간에 대한 정보 저장
            seq_data = []
            for i in range(len(norm_data) - seq_len + 1):
                x = norm_data[i : i + seq_len]
                seq_data.append(x)
            self.samples = []
            for i in range(len(seq_data)):
                self.samples.append(seq_data[idx[i]])
            
            self.samples = np.array(self.samples)
            
            norm_data_tensor = torch.Tensor(self.samples[:,:,:-1]).float().cuda()
            # time 0 ~ seq 까지 다 해줌 
            # 학습할때는 final index만 가져감 
            time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
            self.last = torch.Tensor(self.samples[:,:,-1][:,-1]).float()
            self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
            self.original_sample = torch.tensor(self.original_sample)
            self.samples = torch.tensor(self.samples)
            save_data(loc,data=self.samples,
                    original_data = self.original_sample,
                    train_a=self.train_coeffs[0], 
                    train_b=self.train_coeffs[1], 
                    train_c=self.train_coeffs[2],
                    train_d=self.train_coeffs[3],
                    )
            self.original_sample = np.array(self.original_sample)
            self.samples = np.array(self.samples)
            self.size = len(self.samples)

        # save_data()

    def __getitem__(self, batch_size):
        # import pdb;pdb.set_trace()
        dataset_size = len(self.samples)
        idx = torch.randperm(dataset_size)
        batch_idx = idx[:batch_size]
        original_batch = torch.stack([to_tensor(self.original_sample[i]) for i in batch_idx])
        batch = torch.stack([to_tensor(self.samples[i]) for i in batch_idx])

        # batch _idx -> batch 만큼 가져고 
        a, b, c, d = self.train_coeffs
        batch_a = torch.stack([a[i] for i in batch_idx])
        batch_b = torch.stack([b[i] for i in batch_idx])
        batch_c = torch.stack([c[i] for i in batch_idx])
        batch_d = torch.stack([d[i] for i in batch_idx])

        batch_coeff = (batch_a, batch_b, batch_c, batch_d)
        
        self.sample = {'data': batch , 'inter': batch_coeff, 'original_data':original_batch}

        return self.sample # self.samples[index]

    def __len__(self):
        return len(self.samples)

class TimeDataset_regular(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        total_length = len(data)
        data = data[::-1]
        
        self.min_val = np.min(data, 0)
        self.max_val = np.max(data, 0) - np.min(data, 0)
        
        norm_data = normalize(data)
        total_length = len(norm_data)
        idx = np.array(range(total_length)).reshape(-1,1)
        norm_data = np.concatenate((norm_data,idx),axis=1)#맨 뒤에 관측시간에 대한 정보 저장

        seq_data = []
        for i in range(len(norm_data) - seq_len + 1):
            x = norm_data[i : i + seq_len]
            seq_data.append(x)
        self.samples = []
        idx = torch.randperm(len(seq_data))
        for i in range(len(seq_data)):
            self.samples.append(seq_data[idx[i]])

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)