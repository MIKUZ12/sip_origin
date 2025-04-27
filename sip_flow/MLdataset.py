from torch.utils.data import Dataset, DataLoader
import scipy.io
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import math,random
import scipy.io
from scipy.io import loadmat
def loadMfDIMvMlDataFromMatRatio(mat_path, fold_mat_path, fold_idx=0, ratio_threshold=0.01):
    # load multiple folds double incomplete multi-view multi-label data and labels
    # mark sure the out dimension is n x d, where n is the number of samples
    data = scipy.io.loadmat(mat_path)
    datafold = scipy.io.loadmat(fold_mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    index_file = loadmat(fold_mat_path)
    train_indices = index_file['train_indices'].reshape(-1)
    val_indices = index_file['val_indices'].reshape(-1)
    test_indices = index_file['test_indices'].reshape(-1)
    all_indices = np.concatenate([train_indices, val_indices, test_indices])
    
    mask_train = index_file['mask_train']
    mask_val = index_file['mask_val']
    mask_test = index_file['mask_test']
    combined_mask = np.vstack([mask_train, mask_val, mask_test])
    inc_view_indicator = combined_mask
    train_label_mask = index_file['label_M']
    total_sample_num = labels.shape[0]
    
    original_labels = labels.copy()  # 保存原始标签，用于筛选
    labels = labels[all_indices]
    inc_labels = labels
    inc_label_indicator = train_label_mask
    
    # 筛选过程：计算每个样本标签中1的个数和0的个数的比例
    def filter_by_label_ratio(indices, sample_labels):
        # 获取相应索引的标签
        filtered_labels = sample_labels[indices]
        # 计算每个样本标签中1的个数和0的个数
        ones_count = np.sum(filtered_labels == 1, axis=1)
        zeros_count = np.sum(filtered_labels == 0, axis=1)
        # 避免除零错误
        zeros_count = np.maximum(zeros_count, 1)
        # 计算比例
        ratios = ones_count / zeros_count
        # 返回比例大于阈值的样本索引
        valid_mask = ratios > ratio_threshold
        return indices[valid_mask], valid_mask
    
    # 分别对训练集、验证集、测试集进行筛选
    filtered_train_indices, train_mask = filter_by_label_ratio(train_indices, original_labels)
    filtered_val_indices, val_mask = filter_by_label_ratio(val_indices, original_labels)
    filtered_test_indices, test_mask = filter_by_label_ratio(test_indices, original_labels)
    
    # 更新所有索引
    filtered_all_indices = np.concatenate([filtered_train_indices, filtered_val_indices, filtered_test_indices])
    
    # 更新视图指示器
    filtered_mask_train = mask_train[train_mask]
    filtered_mask_val = mask_val[val_mask]
    filtered_mask_test = mask_test[test_mask]
    filtered_inc_view_indicator = np.vstack([filtered_mask_train, filtered_mask_val, filtered_mask_test])
    
    # 获取筛选后的标签
    filtered_labels = original_labels[filtered_all_indices]
    
    # 按照筛选后的索引重新排列每个视图中的数据
    inc_mv_data = [(StandardScaler().fit_transform(v_data.astype(np.float32))) for
                   v, v_data in enumerate(mv_data)]
    inc_mv_data_new = [v_data[filtered_all_indices,:] for v_data in inc_mv_data]
    
    return inc_mv_data_new, filtered_labels, filtered_labels, filtered_inc_view_indicator, inc_label_indicator, total_sample_num, filtered_train_indices, filtered_val_indices, filtered_test_indices

def loadMfDIMvMlDataFromMat(mat_path, fold_mat_path,fold_idx=0):
    # load multiple folds double incomplete multi-view multi-label data and labels
    # mark sure the out dimension is n x d, where n is the number of samples
    data = scipy.io.loadmat(mat_path)
    datafold = scipy.io.loadmat(fold_mat_path)
    mv_data = data['X'][0]
    labels = data['label']
    labels = labels.astype(np.float32)
    index_file = loadmat(fold_mat_path)
    train_indices = index_file['train_indices'].reshape(-1)
    # print(train_indices.shape)
    val_indices = index_file['val_indices'].reshape(-1)
    test_indices = index_file['test_indices'].reshape(-1)
    all_indices = np.concatenate([train_indices, val_indices, test_indices])
    # contains_zero = 0 in all_indices
    # # 输出结果
    # if contains_zero:
    #     print("The index contains 0.")
    # else:
    #     print("The index does not contain 0.")
    mask_train = index_file['mask_train']
    mask_val = index_file['mask_val']
    mask_test = index_file['mask_test']
    combined_mask = np.vstack([mask_train, mask_val, mask_test])
    inc_view_indicator = combined_mask
    train_label_mask = index_file['label_M']
    total_sample_num = labels.shape[0]
    labels=labels[all_indices]
    inc_labels = labels
    inc_label_indicator = train_label_mask
    # print(train_label_mask.shape)
    inc_mv_data = [(StandardScaler().fit_transform(v_data.astype(np.float32))) for
                   v, v_data in enumerate(mv_data)]
    # 假设all_indices已经通过np.concatenate生成
    # 按照all_indices重新排列每个视图中的数据
    inc_mv_data_new = [v_data[all_indices,:] for v_data in inc_mv_data]
    return inc_mv_data_new,inc_labels,labels,inc_view_indicator,inc_label_indicator,total_sample_num,train_indices,val_indices,test_indices
class IncDataset(Dataset):
    def __init__(self,mat_path, fold_mat_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',semisup=False):
        inc_mv_data, inc_labels, labels, inc_V_ind, inc_L_ind, total_sample_num,train_indices,val_indices,test_indices = loadMfDIMvMlDataFromMatRatio(mat_path,
                                                                                                          fold_mat_path,
                                                                                                          fold_idx)
        self.train_sample_num = len(train_indices)
        # print(self.train_sample_num)
        self.val_sample_num =len(val_indices)
        # print(self.val_sample_num)
        self.test_sample_num = len(test_indices)
        # print(self.test_sample_num)
        if mode == 'train':
            self.cur_mv_data = [v_data[:self.train_sample_num] for v_data in inc_mv_data]
            self.cur_inc_V_ind = inc_V_ind[:self.train_sample_num]
            self.cur_inc_L_ind = inc_L_ind[:self.train_sample_num]
            self.cur_labels = inc_labels[:self.train_sample_num]* self.cur_inc_L_ind
        elif mode == 'val':
            self.cur_mv_data = [v_data[self.train_sample_num:self.train_sample_num + self.val_sample_num] for v_data in
                                inc_mv_data]
            self.cur_labels = labels[self.train_sample_num:self.train_sample_num + self.val_sample_num]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num:self.train_sample_num + self.val_sample_num]
            self.cur_inc_L_ind = np.ones_like(self.cur_labels)
            # print('self.cur_inc_V_ind=', self.cur_inc_V_ind)
            # print('self.cur_inc_L_ind=', self.cur_inc_L_ind)
        else:
            self.cur_mv_data = [v_data[self.train_sample_num + self.val_sample_num:] for v_data in inc_mv_data]
            self.cur_labels = labels[self.train_sample_num + self.val_sample_num:]
            self.cur_inc_V_ind = inc_V_ind[self.train_sample_num + self.val_sample_num:]
            self.cur_inc_L_ind = np.ones_like(self.cur_labels)
        self.mode = mode
        self.classes_num = labels.shape[1]
        self.d_list = [da.shape[1] for da in inc_mv_data]
        self.view_num=len(inc_mv_data)
    def __len__(self):
        if self.mode == 'train':
            return self.train_sample_num
        elif self.mode == 'val':
            return self.val_sample_num
        else: return self.test_sample_num
    def __getitem__(self, index):
        # index = index if self.is_train else self.train_sample_num+index
        data = [torch.tensor(v[index],dtype=torch.float) for v in self.cur_mv_data]
        label = torch.tensor(self.cur_labels[index], dtype=torch.float)
        inc_V_ind = torch.tensor(self.cur_inc_V_ind[index], dtype=torch.int32)
        inc_L_ind = torch.tensor(self.cur_inc_L_ind[index], dtype=torch.int32)
        return data,label,inc_V_ind,inc_L_ind
def getIncDataloader(matdata_path, fold_matdata_path, training_ratio=0.7, val_ratio=0.15, fold_idx=0, mode='train',batch_size=1,num_workers=1,shuffle=False):
    dataset = IncDataset(matdata_path, fold_matdata_path, training_ratio=training_ratio, val_ratio=val_ratio, mode=mode, fold_idx=fold_idx)
    dataloder = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    # for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(dataloder):
    #     print(i,'=',inc_V_ind)
    return dataloder,dataset
if __name__=='__main__':
    # dataloder,dataset = getComDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0.5_LabelMaskRatio_0.5_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    labels = torch.tensor(dataset.cur_labels).float()
