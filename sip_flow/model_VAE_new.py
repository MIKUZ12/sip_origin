import torch
# from utils.expert import weight_sum_var, ivw_aggregate_var
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
def simplified_manifold_loss(x, labels):
        """
        计算简化版的流形损失，不需要视图掩码和标签掩码
        
        参数:
        - x: 形状为(n,d)的张量，表示n个样本的d维特征
        - labels: 形状为(n,c)的张量，表示n个样本的c维标签/伪标签
        
        返回:
        - loss: 标量，表示损失值
        """
        # 安全处理输入，防止NaN
        x = torch.nan_to_num(x, nan=0.0)
        labels = torch.nan_to_num(labels, nan=0.0)
        
        n = x.size(0)  # 样本数量
        if n <= 1:  # 至少需要2个样本
            return torch.tensor(0.0, device=x.device)
        
        # 计算标签相似度矩阵
        # 对标签进行归一化，确保数值稳定性
        normalized_labels = F.normalize(labels, p=2, dim=1)
        label_sim = torch.matmul(normalized_labels, normalized_labels.T)
        
        # 对角线置零，不考虑自身与自身的相似度
        label_sim = label_sim.fill_diagonal_(0)
        
        # 计算特征相似度
        x_normalized = F.normalize(x, p=2, dim=1)  # L2归一化
        feature_sim = torch.matmul(x_normalized, x_normalized.T)  # 余弦相似度
        
        # 限制相似度范围，提高数值稳定性
        feature_sim = torch.clamp(feature_sim, -1.0, 1.0)
        
        # 将余弦相似度转换到[0,1]区间
        feature_sim = (1 + feature_sim) / 2
        
        # 对角线置零
        feature_sim = feature_sim.fill_diagonal_(0)
        
        # 创建一个掩码，排除对角线元素
        mask = 1.0 - torch.eye(n, device=x.device)
        
        # 计算损失：让特征相似度与标签相似度尽量匹配
        # 使用二元交叉熵作为相似度匹配的损失函数
        feature_sim_flat = feature_sim.view(-1)
        label_sim_flat = label_sim.view(-1)
        mask_flat = mask.view(-1)
        
        # 确保相似度在有效范围内，防止log(0)问题
        feature_sim_flat = torch.clamp(feature_sim_flat, 1e-6, 1-1e-6)
        
        # 计算二元交叉熵损失
        pos_loss = label_sim_flat * torch.log(feature_sim_flat)
        neg_loss = (1 - label_sim_flat) * torch.log(1 - feature_sim_flat)
        bce_loss = -(pos_loss + neg_loss)
        
        # 应用掩码并归一化
        masked_loss = bce_loss * mask_flat
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-9)
        
        # 添加正则化项，鼓励特征多样性
        # 这可以防止所有特征都塌缩到相同的值
        # diversity_loss = torch.mean(torch.square(feature_sim - 0.5) * mask)
        
        # 返回总损失，可以调整权重
        return loss 
def compute_cosine_similarity_list(z_sample_view_s, z_sample_view_p):
    """
    计算每个视图的共享特征和私有特征之间的余弦相似度
    Args:
        z_sample_view_s: 视图共享特征 list of [batch_size, feature_dim]
        z_sample_view_p: 视图私有特征 list of [batch_size, feature_dim]
    Returns:
        cos_loss: 余弦相似度损失
    """
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    total_loss = 0
    
    # 确保两个列表长度相同
    assert len(z_sample_view_s) == len(z_sample_view_p)
    
    # 对每个视图分别计算其共享特征和私有特征的余弦相似度
    for z_s, z_p in zip(z_sample_view_s, z_sample_view_p):
        cos_similarity = cos(z_s, z_p)  # [batch_size]
        # 使用线性变换将余弦相似度转换为损失
        loss = 1 - ((cos_similarity + 1) / 2)  # 映射到[0,1]区间
        total_loss += torch.mean(loss)
    
    return total_loss / len(z_sample_view_s)


def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)/(label.sum(dim=1,keepdim=True)+1e-8)
    new_x =  x_embedding*inc_V_ind.T.unsqueeze(-1) + fea.unsqueeze(0)*(1-inc_V_ind.T.unsqueeze(-1))
    return new_x
class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,dropout_rate=0.,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            # layers.append(nn.Dropout(dropout_rate))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                # layers.append(nn.Dropout(dropout_rate))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            # layers.append(nn.Dropout(dropout_rate))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
            # x = x + y
        return x
class sharedQz_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(sharedQz_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        hidden_features = self.mlp(x)
        z_mu = self.z_loc(hidden_features)
        z_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
        return z_mu, z_sca
    
class inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        hidden_features = self.mlp(x)
        # class_feature  = self.z
        return hidden_features
    
class Px_generation_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[512]):
        super(Px_generation_mlp, self).__init__()
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim,final_act=False,final_norm=False)
        # self.transfer_act = nn.ReLU
        # self.px_layer = mlp_layers_creation(self.z_dim, self.x_dim, self.layers, self.transfer_act)
        # self.px_z = nn.Sequential(*self.px_layer)
    def forward(self, z):
        xr = self.mlp(z)
        return xr

class VAE(nn.Module):
    def __init__(self, d_list,z_dim,class_num):
        super(VAE, self).__init__()
        self.x_dim_list = d_list
        self.k = class_num
        self.z_dim = z_dim
        self.num_views = len(d_list)
    # self.switch_layers = switch_layers(z_dim,self.num_views)
        self.z_inference_s = []
        self.z_inference_p = []
        for v in range(self.num_views):
            self.z_inference_s.append(inference_mlp(self.x_dim_list[v], self.z_dim))
            self.z_inference_p.append(inference_mlp(self.x_dim_list[v], self.z_dim))
        self.qz_inference_s = nn.ModuleList(self.z_inference_s)
        self.qz_inference_p = nn.ModuleList(self.z_inference_p)
        self.mlp_2view = MLP(z_dim * (self.num_views - 1), z_dim)  # 连接除最后一个外的所有视图
        self.qz_inference_header = sharedQz_inference_mlp(self.z_dim, self.z_dim)
        self.x_generation_s = []
        self.x_generation_p = []
        for v in range(self.num_views):
            # 为每一个视图都创建一个独立的解码器
            self.x_generation_s.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
            self.x_generation_p.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
        self.px_generation_s = nn.ModuleList(self.x_generation_s)
        self.px_generation_p = nn.ModuleList(self.x_generation_p)

    def inference_z1(self, x_list):
        uniview_mu_s_list = []
        uniview_sca_s_list = []
        fea_s_list = []
        fea_p_list = []
        fea_origin_list = []
        origin_mu_s_list = []
        origin_sca_s_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            # 每一个view都通过qz_inference提取特征
            fea_s = self.qz_inference_s[v](x_list[v])
            fea_p = self.qz_inference_p[v](x_list[v])  
            fea_origin_list.append(fea_s) 
            fea_s_list.append(fea_s)
            fea_p_list.append(fea_p)
        fea_concat = torch.cat(fea_s_list[:-1], dim=1)
        mapped_fea = self.mlp_2view(fea_concat)
        map_loss = F.mse_loss(mapped_fea, fea_s_list[-1])

        fea_s_list[-1] = mapped_fea
        for fea_s in (fea_s_list):
            z_mu_v_s, z_sca_v_s = self.qz_inference_header(fea_s)
            uniview_mu_s_list.append(z_mu_v_s)
            uniview_sca_s_list.append(z_sca_v_s)
        for fea_origin in (fea_origin_list):
            z_mu_v_s, z_sca_v_s = self.qz_inference_header(fea_origin)
            origin_mu_s_list.append(z_mu_v_s)
            origin_sca_s_list.append(z_sca_v_s)
        return uniview_mu_s_list, uniview_sca_s_list, fea_p_list, mapped_fea, map_loss, origin_mu_s_list, origin_sca_s_list
    
    def inference_z_womap(self, x_list):
        uniview_mu_s_list = []
        uniview_sca_s_list = []
        fea_s_list = []
        fea_p_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            # 每一个view都通过qz_inference提取特征
            fea_s = self.qz_inference_s[v](x_list[v])
            fea_p = self.qz_inference_p[v](x_list[v])   
            fea_s_list.append(fea_s)
            fea_p_list.append(fea_p)
        for fea_s in (fea_s_list):
            z_mu_v_s, z_sca_v_s = self.qz_inference_header(fea_s)
            uniview_mu_s_list.append(z_mu_v_s)
            uniview_sca_s_list.append(z_sca_v_s)
        return uniview_mu_s_list, uniview_sca_s_list, fea_p_list
    
    def generation_x(self, z):
        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation_s[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist
    
    def generation_x_p(self, z):
        
        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation2[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist
    
    def poe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        # mask_matrix = torch.stack(mask, dim=0)
        mask_matrix_new = torch.cat([torch.ones([1,mask_matrix.shape[1],mask_matrix.shape[2]]).cuda(),mask_matrix],dim=0)
        p_z_mu = torch.zeros([1,mu.shape[1],mu.shape[2]]).cuda()
        p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        mu_new = torch.cat([p_z_mu,mu],dim=0)
        var_new = torch.cat([p_z_var,var],dim=0)
        exist_mu = mu_new * mask_matrix_new
        T = 1. / (var_new+eps)
        if torch.sum(torch.isnan(exist_mu)).item()>0:
            print('.')
        if torch.sum(torch.isinf(T)).item()>0:
            print('.')
        exist_T = T * mask_matrix_new
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)
        # if torch.sum(torch.isnan(aggregate_var)).item()>0:
        #     print('.')
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / (aggregate_T + eps)
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        return aggregate_mu, aggregate_var
    def moe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        exist_mu = mu * mask_matrix
        exist_var = var * mask_matrix
        aggregate_var = exist_var.sum(dim=0)
        aggregate_mu = exist_mu.sum(dim=0)
        return aggregate_mu,aggregate_var
    def weighted_feature_aggregate(self, features_list, weights, mask=None, eps=1e-5):
        """
        使用已归一化的权重对输入特征进行加权融合
        
        参数:
        - features_list: 各视图的特征列表，每个元素形状为 [N, D]
        - weights: 已归一化的各视图权重，形状为 [V]，满足 sum(weights) = 1
        - mask: 视图掩码 [N, V]
        - eps: 数值稳定性常数
        
        返回:
        - aggregate_features: 融合后的特征，形状为 [N, D]
        """
        # 将特征列表转换为张量 [V, N, D]
        features = torch.stack(features_list, dim=0)
        
        # 将权重改变形状以便进行广播
        weights = weights.view(-1, 1, 1)  # [V, 1, 1]
        weights = weights.expand(-1, features.shape[1], features.shape[2])  # [V, N, D]
        weights = weights.to(features.device)
        
        # 直接计算加权特征并求和
        aggregate_features = torch.sum(features * weights, dim=0)  # [N, D]
        
        # 数值检查
        if torch.isnan(aggregate_features).any():
            print('警告: 融合特征中存在NaN')
            aggregate_features = torch.nan_to_num(aggregate_features, nan=0.0)
        
        return aggregate_features
    def forward(self, x_list, mode,mask=None):
        # uniview_mu_list, uniview_sca_list = self.inference_z(x_list)
        if (mode == 1):
            mu_s_list, sca_s_list, fea_p_list, mapped_fea, map_loss, origin_mu_s_list, origin_sca_s_list = self.inference_z1(x_list)
        else:
            mu_s_list, sca_s_list, fea_p_list = self.inference_z_womap(x_list)
            mapped_fea = None
            map_loss = None
        z_mu = torch.stack(mu_s_list,dim=0) # [v n d]
        z_sca = torch.stack(sca_s_list,dim=0) # [v n d]
        if torch.sum(torch.isnan(z_mu)).item() > 0:
            print("z:nan")
            pass
        # if self.training:
        #     z_mu = fill_with_label(label_embedding_mu,label,z_mu,mask)
        #     z_sca = fill_with_label(label_embedding_var,label,z_sca,mask)
        fusion_mu, fusion_sca = self.poe_aggregate(z_mu, z_sca, mask)
        fusion_origin_mu, fusion_origin_sca = self.poe_aggregate(torch.stack(origin_mu_s_list,dim=0), torch.stack(origin_sca_s_list,dim=0), mask)
        z_sample_list_s = []
        for i in range(len(sca_s_list)):
            z_sample_view_s = gaussian_reparameterization_var(mu_s_list[i], sca_s_list[i], times=5)
            z_sample_list_s.append(z_sample_view_s)
        if torch.sum(torch.isnan(fusion_mu)).item() > 0:
            pass
        assert torch.sum(fusion_sca<0).item() == 0
        z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca,times=10)
        z_origin_sample = gaussian_reparameterization_var(fusion_origin_mu, fusion_origin_sca,times=10)
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            print("z:nan")
            pass
        xr_list = self.generation_x(z_sample)
        xr_p_list = []
        for v in range(self.num_views):
            reconstruct_x_p = self.px_generation_p[v](fea_p_list[v])
            xr_p_list.append(reconstruct_x_p)
        cos_loss = compute_cosine_similarity_list(z_sample_list_s, fea_p_list)
        # z_sample_list = []
        # for i,mu in enumerate(uniview_mu_list):
        #     z_sample_list.append(gaussian_reparameterization_var(uniview_mu_list[i],uniview_sca_list[i]))
            # z_sample_list.append((mu))
        # xr_list_views = self.generation_x_s1(z_sample_list)
        # c_z_sample = self.gaussian_rep_function(fusion_z_mu, fusion_z_sca)
        return z_sample, mu_s_list, sca_s_list, fusion_mu, fusion_sca, xr_list, xr_p_list, cos_loss, mapped_fea, map_loss, fea_p_list, z_origin_sample
