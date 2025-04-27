import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model_new import get_model
import evaluation
import torch
import numpy as np
import copy
from myloss import Loss, vade_trick
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
# from constructGraph import getMvKNNGraph, getIncMvKNNGraph
import time
import multiprocessing

def view_mixup(data_list,label,v_mask,l_mask):
    new_data = []
    new_labels = label.unsqueeze(0).repeat(len(data_list),1,1)
    num = data_list[0].shape[0]
    for i,view_data in enumerate(data_list):
        indices = torch.randperm(num).cuda()
        new_data.append(torch.index_select(view_data, 0, indices))
        new_labels[i] = torch.index_select(new_labels[i],0,indices)
        v_mask[:,i] = torch.index_select(v_mask[:,i],0,indices)
    label = new_labels.sum(dim=0)
    label = torch.masked_fill(label,label>0,1)
    l_mask = torch.masked_fill(l_mask,label>0,1)
    return new_data,label,v_mask,l_mask


def train(loader, model, loss_model, opt, sche, epoch,dep_graph,last_preds,logger,selected_view):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mce= nn.MultiLabelSoftMarginLoss()
    model.train()
    end = time.time()
    All_preds = torch.tensor([]).cuda()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')
        data_selected = [data[i] for i in selected_view]
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_emb_sample, pred, xr_p_list, cos_loss, mapped_loss,loss_manifold_p_avg, fusion_p, mapped_fea = model(data_selected,mode=0,mask=inc_V_ind )
        All_preds = torch.cat([All_preds,pred],dim=0)
        if epoch<args.pre_epochs:
            loss_mse_viewspec = 0
            loss_CL_views = 0
            loss_list=[]
            loss = loss_CL_views
            assert torch.sum(torch.isnan(loss)).item() == 0
            assert torch.sum(torch.isinf(loss)).item() == 0
        else:
            loss_CL = loss_model.weighted_BCE_loss(pred,label,inc_L_ind)
            z_c_loss = loss_model.z_c_loss_new(z_sample, label, label_emb_sample,inc_L_ind)
            cohr_loss = loss_model.corherent_loss(uniview_mu_list, uniview_sca_list,fusion_z_mu, fusion_z_sca,mask=inc_V_ind)
            loss_mse = 0
            loss_mse_p = 0
            for v in range(len(data_selected)):
                loss_mse += loss_model.weighted_wmse_loss(data_selected[v],xr_list[v],inc_V_ind[:,v],reduction='mean')
            for v in range(len(data_selected)):
                ## xr_list是每个重构的视图，这个损失是用来约束VAE的，使得潜在空间的特征能够很好的表征原数据
                loss_mse_p += loss_model.weighted_wmse_loss(data_selected[v],xr_p_list[v],inc_V_ind[:,v],reduction='mean')
            assert torch.sum(torch.isnan(loss_mse)).item() == 0
            loss = loss_CL + loss_mse *args.alpha + z_c_loss*args.beta + cohr_loss *args.sigma
        # loss = loss_CL
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        opt.step()
        # print(model.classifier.parameters().grad)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,All_preds,label_emb_sample

def test(loader, model, loss_model, epoch,logger,selected_view):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        # data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        data_selected = [data[i] for i in selected_view]
        # pred,_,_ = model(data,mask=torch.ones_like(inc_V_ind).to('cuda:0'))
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_emb_sample, qc_z, xr_p_list, cos_loss, _ ,_,_,_ = model(data_selected,mode=0,mask=inc_V_ind.to('cuda:0'))
        # qc_x = vade_trick(fusion_z_mu, model.mix_prior, model.mix_mu, model.mix_sca)
        pred = qc_z
        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        loss=loss_model.weighted_BCE_loss(pred,label,inc_L_ind)
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)
    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        losses = losses,
                        ap=evaluation_results[4],
                        hl=evaluation_results[0],
                        rl=evaluation_results[3],
                        auc=evaluation_results[5]
                        ))
    return evaluation_results

def main(args,file_path):
    panduan_list=[0]
    for panduan in panduan_list:
        # if panduan==5:
        #     args.label_missing_rates = [0]
        #     args.feature_missing_rates = [0]
        #     args.folds_num=10
        if panduan==0:
            args.label_missing_rates=[0]
            args.feature_missing_rates=[0]
            args.folds_num=1
        # if panduan==1:
        #     args.label_missing_rates = [0, 0.1, 0.3, 0.7]
        #     args.feature_missing_rates = [0.5]
        #     args.folds_num = 3
        # if panduan==2:
        #     args.label_missing_rates = [0.1]
        #     args.feature_missing_rates = [0.1]
        #     args.folds_num = 3
        # if panduan==3:
        #     args.label_missing_rates = [0.3]
        #     args.feature_missing_rates = [0.3]
        #     args.folds_num = 3
        # if panduan==4:
        #     args.label_missing_rates = [0.7]
        #     args.feature_missing_rates = [0.7]
        #     args.folds_num = 3

        for label_missing_rate in args.label_missing_rates:
            for feature_missing_rate in args.feature_missing_rates:
                folds_num = args.folds_num
                
                # 使用字典而不是列表来存储不同视图组合的结果
                test_acc_list = {}
                test_one_error_list = {}
                test_coverage_list = {}
                test_ranking_loss_list = {}
                test_precision_list = {}
                test_auc_list = {}
                test_f1_list = {}
                training_time_list = {}
                for fold_idx in range(folds_num):
                    seed=fold_idx
                    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'.mat')
                    # fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset + str(
                    #     mask_label_ratio) + '_LabelMaskRatio_' + str(mask_view_ratio) + '_MaskRatios_' + '.mat')
                    dataset_name = args.dataset
                    folder_name = f'/root/lqj/dy_missing_sip/sip_origin/sip_origin/Index/{dataset_name}'
                    file_name = f'{dataset_name}_label_missing_rate_{label_missing_rate}_feature_missing_rate_{feature_missing_rate}_seed_{seed + 1}.mat'
                    data_path = osp.join(args.root_dir, args.dataset + '.mat')
                    index_file_path = os.path.join(folder_name, file_name)
                    fold_data_path=index_file_path
                    folds_results = [AverageMeter() for i in range(9)]
                    if args.logs:
                        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                                    args.mask_view_ratio) + '_L_' +
                                                    str(args.mask_label_ratio) + '_T_' +
                                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
                    else:
                        logfile=None
                    logger = utils.setLogger(logfile)
                    device = torch.device('cuda:0')
                    label_Inp_list = []
                    fold_idx=fold_idx
                    train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = True,num_workers=0)
                    test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=0)
                    val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=0)
                    d_list = train_dataset.d_list
                    classes_num = train_dataset.classes_num
                    labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
                    dep_graph = torch.matmul(labels.T,labels)
                    dep_graph = dep_graph/(torch.diag(dep_graph).unsqueeze(1)+1e-10)
                    # dep_graph[dep_graph<=args.sigma]=0.
                    dep_graph.fill_diagonal_(fill_value=0.)
                    pri_c = train_dataset.cur_labels.sum(axis=0)/train_dataset.cur_labels.shape[0]
                    pri_c = torch.tensor(pri_c).cuda()
                    
                    # 定义所有需要测试的视图组合
                    all_selected_views = [
                        [0,1,2,3,4,5],  # 所有视图
                    ]
                    
                    
                    # 创建一个字典来存储所有视图组合的结果
                    all_views_results = {}
                    
                    # 为每个视图组合进行训练和测试
                    for selected_view in all_selected_views:
                        view_name = ','.join(map(str, selected_view))
                        logger.info(f"Training with selected views: {view_name}")
                        
                        start = time.time()
                        d_list_selected = [train_dataset.d_list[i] for i in selected_view]  
                        model = get_model(d_list_selected, num_classes=classes_num, z_dim=args.z_dim, adj=dep_graph, rand_seed=0)
                        loss_model = Loss()
                        optimizer = Adam(model.parameters(), lr=args.lr)
                        scheduler = None
                        
                        static_res = 0
                        epoch_results = [AverageMeter() for i in range(9)]
                        total_losses = AverageMeter()
                        train_losses_last = AverageMeter()
                        best_epoch = 0
                        best_model_dict = {'model': model.state_dict(), 'epoch': 0}
                        
                        for epoch in range(args.epochs):
                            if epoch == 0:
                                All_preds = None
                            train_losses, model, All_preds, label_emb_sample = train(train_dataloder, model, loss_model, optimizer, scheduler, epoch, dep_graph, All_preds, logger, selected_view)
                            label_InP = label_emb_sample.mm(label_emb_sample.t())
                            
                            if epoch >= args.pre_epochs:
                                val_results = test(val_dataloder, model, loss_model, epoch, logger, selected_view)
                                if val_results[0]*0.25+val_results[4]*0.25+val_results[5]*0.25 >= static_res:
                                    static_res = val_results[0]*0.25+val_results[4]*0.25+val_results[5]*0.25
                                    best_model_dict['model'] = copy.deepcopy(model.state_dict())
                                    best_model_dict['epoch'] = epoch
                                    best_epoch = epoch
                                train_losses_last = train_losses
                                total_losses.update(train_losses.sum)
                        
                        model.load_state_dict(best_model_dict['model'])
                        end = time.time()
                        logger.info(f"Best epoch: {best_model_dict['epoch']}")
                        test_results = test(test_dataloder, model, loss_model, epoch, logger, selected_view)
                        
                        # 存储当前视图组合的结果
                        all_views_results[view_name] = {
                            'accuracy': test_results[0],
                            'one_error': test_results[1],
                            'coverage': test_results[2],
                            'ranking_loss': test_results[3],
                            'precision': test_results[4],
                            'auc': test_results[5],
                            'f1': test_results[6],
                            'training_time': end - start,
                            'best_epoch': best_epoch
                        }
                        
                        # 清理显存
                        del model
                        torch.cuda.empty_cache()
                    
                    # 收集当前fold的结果，保存到对应的列表中
                    for view_name, results in all_views_results.items():
                        if view_name not in test_acc_list:
                            test_acc_list[view_name] = []
                            test_one_error_list[view_name] = []
                            test_coverage_list[view_name] = []
                            test_ranking_loss_list[view_name] = []
                            test_precision_list[view_name] = []
                            test_auc_list[view_name] = []
                            test_f1_list[view_name] = []
                            training_time_list[view_name] = []
                        
                        test_acc_list[view_name].append(results['accuracy'])
                        test_one_error_list[view_name].append(results['one_error'])
                        test_coverage_list[view_name].append(results['coverage'])
                        test_ranking_loss_list[view_name].append(results['ranking_loss'])
                        test_precision_list[view_name].append(results['precision'])
                        test_auc_list[view_name].append(results['auc'])
                        test_f1_list[view_name].append(results['f1'])
                        training_time_list[view_name].append(results['training_time'])
                if 1 + 1 == 2:
                    # 创建保存结果的目录
                    saved_path = f"./test_results/{dataset_name}/"
                    if not os.path.exists(saved_path):
                        os.makedirs(saved_path)
                    
                    # 创建一个汇总文件，包含所有视图组合的结果
                    summary_file = saved_path + f"all_views_summary_gamma_{args.gamma}_alpha_{args.alpha}_label_missing_rate_{label_missing_rate}_feature_missing_rate_{feature_missing_rate}.txt"
                    
                    with open(summary_file, 'w') as file:
                        file.write('============================ Multi-View Results Summary ============================\n')
                        file.write('>>>>>>>> Experimental Setup:\n')
                        file.write('>>>>>>>> Dataset Info:\n')
                        file.write(f'Dataset name: {dataset_name}\n')
                        file.write(f'Label missing rate: {label_missing_rate}, Feature missing rate: {feature_missing_rate}\n')
                        file.write(f'Alpha: {args.alpha}, Beta: {args.beta}, Gamma: {args.gamma}, Sigma: {args.sigma}\n\n')
                        
                        # 为每个视图组合写入结果
                        for view_name in sorted(test_acc_list.keys()):
                            file.write(f'===== View Combination: {view_name} =====\n')
                            file.write('>>>>>>>> Mean Score (standard deviation):\n')
                            file.write(
                                'Accuracy: {0:.4f} ({1:.4f})\n'.format(np.mean(test_acc_list[view_name]),
                                                                       np.std(test_acc_list[view_name])))
                            file.write('one_error: {0:.4f} ({1:.4f})\n'.format(
                                np.mean(test_one_error_list[view_name]),
                                np.std(test_one_error_list[view_name])))
                            file.write(
                                'coverage: {0:.4f} ({1:.4f})\n'.format(np.mean(test_coverage_list[view_name]),
                                                                       np.std(test_coverage_list[view_name])))
                            file.write('ranking_loss: {0:.4f} ({1:.4f})\n'.format(
                                np.mean(test_ranking_loss_list[view_name]),
                                np.std(test_ranking_loss_list[view_name])))
                            file.write('Precision: {0:.4f} ({1:.4f})\n'.format(
                                np.mean(test_precision_list[view_name]),
                                np.std(test_precision_list[view_name])))
                            file.write(
                                'AUC: {0:.4f} ({1:.4f})\n'.format(np.mean(test_auc_list[view_name]),
                                                                  np.std(test_auc_list[view_name])))
                            file.write(
                                'F1: {0:.4f} ({1:.4f})\n'.format(np.mean(test_f1_list[view_name]),
                                                                 np.std(test_f1_list[view_name])))
                            file.write(
                                '>>>>>>>> Mean Training Time (standard deviation)(s):{0:.4f} ({1:.4f})\n\n'.format(
                                    np.mean(training_time_list[view_name]), np.std(training_time_list[view_name])))
                    
                    # 为每个视图组合保存一个单独的详细文件
                    for view_name in test_acc_list.keys():
                        view_file = saved_path + f"view_{view_name.replace(',', '_')}_gamma_{args.gamma}_alpha_{args.alpha}_label_missing_rate_{label_missing_rate}_feature_missing_rate_{feature_missing_rate}.txt"
                        
                        with open(view_file, 'w') as file:
                            file.write(
                                '============================ Summary Board ============================\n')
                            file.write('>>>>>>>> Experimental Setup:\n')
                            file.write('>>>>>>>> Dataset Info:\n')
                            file.write(f'Dataset name: {dataset_name}\n')
                            file.write(f'Selected views: {view_name}\n')
                            file.write('>>>>>>>> Experiment Results:\n')
                            for i in range(1, len(test_acc_list[view_name]) + 1):
                                file.write(f'Test{i}:  ')
                                file.write(
                                    'Accuracy:{0:.4f} | one_error:{1:.4f}| coverage:{2:.4f} | ranking_loss:{3:.4f}| Precision:{4:.4f} | AUC:{5:.4f} | F1:{6:.4f} | \n'.format(
                                        test_acc_list[view_name][i - 1], test_one_error_list[view_name][i - 1],
                                        test_coverage_list[view_name][i - 1],
                                        test_ranking_loss_list[view_name][i - 1], test_precision_list[view_name][i - 1],
                                        test_auc_list[view_name][i - 1], test_f1_list[view_name][i - 1]))
                            file.write('>>>>>>>> Mean Score (standard deviation):\n')
                            file.write(
                                'Accuracy: {0:.4f} ({1:.4f})\n'.format(np.mean(test_acc_list[view_name]),
                                                                       np.std(test_acc_list[view_name])))
                            file.write('one_error: {0:.4f} ({1:.4f})\n'.format(
                                np.mean(test_one_error_list[view_name]),
                                np.std(test_one_error_list[view_name])))
                            file.write(
                                'coverage: {0:.4f} ({1:.4f})\n'.format(np.mean(test_coverage_list[view_name]),
                                                                       np.std(test_coverage_list[view_name])))
                            file.write('ranking_loss: {0:.4f} ({1:.4f})\n'.format(
                                np.mean(test_ranking_loss_list[view_name]),
                                np.std(test_ranking_loss_list[view_name])))
                            file.write('Precision: {0:.4f} ({1:.4f})\n'.format(
                                np.mean(test_precision_list[view_name]),
                                np.std(test_precision_list[view_name])))
                            file.write(
                                'AUC: {0:.4f} ({1:.4f})\n'.format(np.mean(test_auc_list[view_name]),
                                                                  np.std(test_auc_list[view_name])))
                            file.write(
                                'F1: {0:.4f} ({1:.4f})\n'.format(np.mean(test_f1_list[view_name]),
                                                                 np.std(test_f1_list[view_name])))
                            file.write(
                                '>>>>>>>> Mean Training Time (standard deviation)(s):{0:.4f} ({1:.4f})\n'.format(
                                    np.mean(training_time_list[view_name]), np.std(training_time_list[view_name])))
                            for itr in range(len(training_time_list[view_name])):
                                file.write(f"{itr}: {training_time_list[view_name][itr]}\n")

def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH',
                        default='/root/lqj/dy_missing_sip/sip_origin/sip_origin/data')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k_six_view'])#'mirflickr_six_view','pascal07_six_view' 30
    # parser.add_argument('--mask-view-ratio', type=list, default=[0.5])
    # # parser.add_argument('--mask-label-ratio', type=list, default=[0.79,0.82,0.85,0.88,0.91,0.94,0.97])
    # parser.add_argument('--mask-label-ratio', type=list, default=[0.5])
    parser.add_argument('--feature_missing_rates', type=list, default=[0.5])
    parser.add_argument('--label_missing_rates', type=float, default=[0.5])
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=3, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100) #200 for corel5k  100 for iaprtc12 50 for pascal07 30 for espgame
    parser.add_argument('--pre_epochs', type=int, default=0)
    # Training args
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=0.)
    args = parser.parse_args()
    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    lr_list = [1e-3]
    alpha_list = [1e0]#
    beta_list = [1e-3]#1e-3 for corel5k and mirflickr, 1e0 for pascal07, 1e-1 for espgame, 1e0 for iaprtc12
    gamma_list = [0]
    sigma_list = [1e0]#1e0for others ,1e-1 for mirflickr
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for sigma in sigma_list:
                        args.sigma = sigma
                        for dataset in args.datasets:
                            args.dataset = dataset
                            file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                                            0.5) + '_LM_' +
                                            str(0.5) + '_T_' +
                                            str(args.training_sample_ratio) + '.txt')
                            args.file_path = file_path
                            existed_params = filterparam(file_path,[-5,-4,-3,-2,-1])
                            if [args.lr,args.alpha,args.beta,args.gamma,args.sigma] in existed_params:
                                print('existed param! beta:{}'.format(args.beta))
                                # continue
                            if __name__ == '__main__':
                                main(args,file_path)