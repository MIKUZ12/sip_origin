import torch
import torch.nn as nn 
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder
import random
import copy
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from layers_new import GIN, FDModel, MLP,GAT
from model_VAE_new import VAE
from model_VAE_new import simplified_manifold_loss
def gaussian_reparameterization_std(means, std, times=1):
    std = std.abs()
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var+1e-8)
    assert torch.sum(std<0).item()==0
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def setEmbedingModel(d_list,d_out):
    return nn.ModuleList([Mlp(d,d,d_out)for d in d_list])

def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)
    new_x =  x_embedding*inc_V_ind.unsqueeze(-1) + fea.unsqueeze(1)
    return new_x

class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,final_act=True,final_norm=True):
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

class Qc_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(Qc_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)
    def forward(self, x):
        assert torch.sum(torch.isnan(x)).item() == 0
        hidden_features = self.mlp(x)
        c_mu = self.z_loc(hidden_features)
        c_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
        if torch.sum(torch.isnan(c_mu)).item() >0:
            pass
        assert torch.sum(torch.isinf(c_mu)).item() == 0
        return c_mu, c_sca

class Net(nn.Module):
    def __init__(self, d_list,num_classes,z_dim,adj,rand_seed=0):
        super(Net, self).__init__()
        # self.configs = configs
        self.rand_seed = rand_seed
        # Label semantic encoding module
        self.label_embedding_u = nn.Parameter(torch.eye(num_classes),
                                            requires_grad=True)
        self.label_embedding_std = nn.Parameter(torch.ones(num_classes),
                                            requires_grad=True)
        # self.label_embedding = nn.Parameter(torch.randn([num_classes,num_classes]),
        #                                     requires_grad=True)
        self.label_adj = nn.Parameter(torch.eye(num_classes),
                                      requires_grad=True)
        self.adj = adj
        self.z_dim = z_dim
        self.batchnorm = nn.BatchNorm1d(z_dim)
        # self.GIN_encoder = GIN(2, num_classes, z_dim,
        #                        [math.ceil(z_dim / 2)])
        self.GAT_encoder = GAT(z_dim, z_dim,dropout=0.0)
        self.label_mlp = Qc_inference_mlp(num_classes, z_dim)
        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.k = num_classes
        # VAE module
        self.VAE = VAE(d_list=d_list,z_dim=z_dim,class_num=num_classes)

        # Classifier
        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  z_dim*2, groups=num_classes)
        
        self.cls = nn.Linear(z_dim, num_classes)
        self.view_cls = nn.ModuleList([nn.Linear(z_dim, num_classes) for i in range(len(d_list)) ])
        self.set_prior()
        self.cuda()
        # self.reset_parameters()
    def set_prior(self):
        # set prior components weights
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        # set prior gaussian components mean
        # self.mix_mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)
        self.mix_mu = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
        # set prior gaussian components scale
        self.mix_sca = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        # nn.init.normal_(self.label_embedding_u)
        nn.init.normal_(self.label_adj)
        # self.GIN_encoder.reset_parameters()
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()
        
    def get_config_optim(self):
        return [{'params': self.GAT_encoder.parameters()},
                {'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]
        
    def poe_two(self, z_mu, z_var, c_mu, c_var, eps=1e-5):
        z_mu = z_mu.unsqueeze(1)
        z_var = z_var.unsqueeze(1)
        c_mu = c_mu.unsqueeze(0)
        c_var = c_var.unsqueeze(0)
        # p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        # mu_new = torch.cat([p_z_mu,mu],dim=0)
        # var_new = torch.cat([p_z_var,var],dim=0)
        s_mu = torch.zeros([z_mu.shape[0],c_mu.shape[1],z_mu.shape[2]]).cuda()
        s_var = torch.ones([z_mu.shape[0],c_mu.shape[1],z_mu.shape[2]]).cuda()
        # mu_new = torch.cat([p_z_mu,mu],dim=0)
        # var_new = torch.cat([p_z_var,var],dim=0)
        T_x = 1. / (z_var+eps)
        T_c = 1. / (c_var+eps)
        T_s = 1. / (s_var+eps)
        T_sum = T_x + T_c + T_s
        aggregate_mu = (z_mu*T_x+c_mu*T_c+ s_mu*T_s)/T_sum
        aggregate_var = 1. / T_sum
        # mask_matrix = torch.stack(mask, dim=0)
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        assert torch.sum(torch.isnan(aggregate_mu)).item()==0
        assert torch.sum(torch.isinf(aggregate_mu)).item()==0
        assert torch.sum(torch.isnan(aggregate_var)).item()==0
        assert torch.sum(torch.isinf(aggregate_var)).item()==0
        return aggregate_mu, aggregate_var
    def forward(self, x_list,mode,mask):
        masked_x_list = []
        dropout_rate = 0.2
        for x in x_list:
            # 生成与x相同形状的随机掩码
            random_mask = torch.rand_like(x) > dropout_rate
            # 应用掩码（相当于dropout）
            masked_x = x * random_mask.float()
            masked_x_list.append(masked_x)

        label_embedding  =  self.label_embedding_u
        # label_embedding = gaussian_reparameterization_std(self.label_embedding_u,self.label_embedding_std)
        # label_embedding = self.GAT_encoder(label_embedding, self.adj)
        assert torch.sum(torch.isnan(self.label_embedding_u)).item() == 0
        assert torch.sum(torch.isinf(self.label_embedding_u)).item() == 0
        label_embedding, label_embedding_var = self.label_mlp(self.label_embedding_u)
        label_embedding_sample = gaussian_reparameterization_var(label_embedding,label_embedding_var,10)
        assert torch.sum(torch.isnan(label_embedding_sample)).item() == 0
        assert torch.sum(torch.isinf(label_embedding_sample)).item() == 0
        # label_embedding_sample = self.GAT_encoder(label_embedding_sample, self.adj)
        if torch.sum(torch.isnan(label_embedding)).item() > 0:
            assert torch.sum(torch.isnan(label_embedding)).item() == 0
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, xr_p_list, cos_loss, mapped_fea, mapped_loss,fea_p_list, z_origin_sample = self.VAE(masked_x_list,mode,mask=None)  #Z[i]=[128, 260, 512] b c d_e
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            pass 
        p_vae_p_list = []
        for v in range(len(fea_p_list)):
            qc_p_z = torch.cat((fea_p_list[v].unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(fea_p_list[v].shape[0],1,1)),dim=-1)
            p_p = self.cls_conv(qc_p_z).squeeze(-1)
            p_p = torch.sigmoid(p_p)
            p_vae_p_list.append(p_p)
        loss_manifold_p = []
        for v in range(len(p_vae_p_list)):
            loss_manifold_p.append(simplified_manifold_loss(fea_p_list[v], p_vae_p_list[v]))
        loss_manifold_p = torch.tensor(loss_manifold_p)
        loss_manifold_p_avg = loss_manifold_p.mean()
        weights_p = 1.0 / (loss_manifold_p + 1e-10)
        weights_p = weights_p / (weights_p.sum() + 1e-10)
        fusion_p = self.VAE.weighted_feature_aggregate(fea_p_list, weights_p)
        fusion_p_sigmoid = fusion_p.sigmoid()
        fusion_fea = z_sample * fusion_p_sigmoid

        qc_origin_z = torch.cat((z_origin_sample.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(z_origin_sample.shape[0],1,1)),dim=-1)
        p_origin = self.cls_conv(qc_origin_z).squeeze(-1)
        p_origin = torch.sigmoid(p_origin)


        fusion_fea = self.batchnorm(fusion_fea)
        qc_z = torch.cat((fusion_fea.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(fusion_fea.shape[0],1,1)),dim=-1)
        p = self.cls_conv(qc_z).squeeze(-1)
        p = torch.sigmoid(p)

        loss_align = torch.pow(p_origin - p, 2).mean()

        return z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_embedding_sample, p, xr_p_list, cos_loss, mapped_loss, loss_manifold_p_avg, fusion_p, mapped_fea, loss_align

def get_model(d_list,num_classes,z_dim,adj,rand_seed=0):
    model = Net(d_list,num_classes=num_classes,z_dim=z_dim,adj=adj,rand_seed=rand_seed)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() 
                                    else 'cpu'))
    return model
    
if __name__=="__main__":
    # input=torch.ones([2,10,768])
    from MLdataset import getIncDataloader
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0_LabelMaskRatio_0_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    input = next(iter(dataloder))[0]
    model=get_model(num_classes=260,beta=0.2,in_features=1,class_emb=260,rand_seed=0)
    input = [v_data.to('cuda:0') for v_data in input]
    # print(input[0])
    pred,_,_=model(input)