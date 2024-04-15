import argparse
import os
import shutil
import time
import random
import math
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import Bar, Logger, AverageMeter, accuracy, WeightEMA, interleave, save_checkpoint
from tensorboardX import SummaryWriter
from datasets.datasets_wws import get_dataset_class
from utils.evaluate_utils import hungarian_evaluate
from models.build_model_hierarchical import build_model
from utils.uncr_util import uncr_generator
from utils.sinkhorn_knopp import SinkhornKnopp
# from utils.cutmix import Mixup_transmix
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class LearnableBetaDistribution(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(LearnableBetaDistribution, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return torch.distributions.beta.Beta(self.alpha, self.beta).log_prob(x)


parser = argparse.ArgumentParser(description='TRSSL Training')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='train batchsize')
parser.add_argument('--num_protos', default=200, type=int, metavar='N', help='number of subclass')
parser.add_argument('--num_concepts', default=50, type=int, metavar='N', help='number of concepts')
parser.add_argument('--num-workers', default=4, type=int, help='number of dataloader workers')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wdecay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--warmup-epochs', default=10, type=int, help='number of warmup epochs')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Method options
parser.add_argument('--lbl-percent', type=int, default=10, help='Percentage of labeled data')
parser.add_argument('--novel-percent', default=50, type=int, help='Percentage of novel classes, default 50')
parser.add_argument('--train-iteration', type=int, default=1024, help='Number of iteration per epoch')
parser.add_argument('--out', default='outputs/hierarchical_wws/', help='/root/workspace/wangyu/ICCV2023/TRSSL-main/outputs/base/')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100',"cifar100_20"], help='dataset name')
parser.add_argument('--data-root', default=f'data', help='directory to store data')
parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet50'], help='model architecure')
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--imagenet-classes", default=100, type=int, help="number of ImageNet classes")
parser.add_argument('--description', default="default_run", type=str, help='description of the experiment')
parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
parser.add_argument("--uncr-freq", default=1, type=int, help="frequency of generating uncertainty scores")
parser.add_argument("--threshold", default=0.5, type=float, help="threshold for hard pseudo-labeling")
parser.add_argument("--imb-factor", default=1, type=float, help="imbalance factor of the data, default 1")


def sim_matrix(a, b, args, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.sum((input1 - input2)**2)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()

args.data_root = os.path.join(args.data_root, args.dataset)
os.makedirs(args.data_root, exist_ok=True)

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy

if args.dataset == "cifar10":
    args.no_class = 10
elif args.dataset == "cifar100":
    args.no_class = 100
elif args.dataset == "cifar100_20":
    args.no_class = 90
elif args.dataset == "tinyimagenet":
    args.no_class = 200
elif args.dataset == "stanfordcars":
    args.no_class = 196
elif args.dataset == "aircraft":
    args.no_class = 100
elif args.dataset == "oxfordpets":
    args.no_class = 37
elif args.dataset == "imagenet100":
    args.no_class = 100


def main():
    global best_acc
    run_started = datetime.today().strftime('%d-%m-%y_%H%M%S')
    args.exp_name = f'dataset_{args.dataset}_arch_{args.arch}_lbl_percent_{args.lbl_percent}_novel_percent_{args.novel_percent}_{args.description}_{run_started}'
    args.out = os.path.join(args.out, args.exp_name)

    os.makedirs(args.out, exist_ok=True)

    with open(f'{args.out}/parameters.txt', 'a+') as ofile:
        ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))

    # load dataset
    args.no_seen = args.no_class - int((args.novel_percent*args.no_class)/100)
    dataset_class = get_dataset_class(args)
    train_labeled_dataset, train_unlabeled_dataset, uncr_dataset, test_dataset_all, test_dataset_seen, test_dataset_novel = dataset_class.get_dataset()



    labeled_trainloader = data.DataLoader(train_labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    uncr_loader = data.DataLoader(uncr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_all = data.DataLoader(test_dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_seen = data.DataLoader(test_dataset_seen, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_novel = data.DataLoader(test_dataset_novel, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # build models
    model = build_model(args)

    ema_model = build_model(args, ema=True)

    # Sinkorn-Knopp
    sinkhorn = SinkhornKnopp(args)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        warmup_start_lr=0.001,
        eta_min=0.001,
    )

    ema_optimizer= WeightEMA(args, model, ema_model)
    start_epoch = 0

    # Resume
    title = f'ood-{args.dataset}'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['-Train Loss-', '-Test Acc. Seen-', '-Test Acc. Novel-', '-Test NMI Novel-', '-Test Acc. All-', '-Test NMI All-'])

    writer = SummaryWriter(args.out)
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss = train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, sinkhorn, epoch, use_cuda)
        all_cluster_results = test_cluster(args, test_loader_all, ema_model, epoch)
        novel_cluster_results = test_cluster(args, test_loader_novel, ema_model, epoch, offset=args.no_seen)
        test_acc_seen = test_seen(args, test_loader_seen, ema_model, epoch)

        if args.uncr_freq > 0:
            if (epoch+1)%args.uncr_freq == 0:
                temp_uncr = uncr_generator(args, uncr_loader, ema_model)
                train_labeled_dataset, train_unlabeled_dataset = dataset_class.get_dataset(temp_uncr=temp_uncr)
            labeled_trainloader = data.DataLoader(train_labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
            unlabeled_trainloader = data.DataLoader(train_unlabeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

        test_acc = all_cluster_results["acc"]

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print(f'epoch: {epoch}, acc-seen: {test_acc_seen}')
        print(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}')
        print(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}')

        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('test/1.acc_seen', test_acc_seen, epoch)
        writer.add_scalar('test/2.acc_novel', novel_cluster_results['acc'], epoch)
        writer.add_scalar('test/3.nmi_novel', novel_cluster_results['nmi'], epoch)
        writer.add_scalar('test/4.acc_all', all_cluster_results['acc'], epoch)
        writer.add_scalar('test/5.nmi_all', all_cluster_results['nmi'], epoch)

        # append logger file
        logger.append([train_loss, test_acc_seen, novel_cluster_results['acc'], novel_cluster_results['nmi'], all_cluster_results['acc'], all_cluster_results['nmi']])

        # save model
        model_to_save = model.module if hasattr(model, "module") else model
        ema_model_to_save = ema_model.module if hasattr(ema_model, "module") else ema_model

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_model_to_save.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.out)
        test_accs.append(test_acc)

        #call scheduler
        scheduler.step()
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, sinkhorn, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sim_losses = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x, _, temp_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _, temp_x = next(labeled_train_iter)
            
        try:
            (inputs_u, inputs_u2, inputs_us), _, _, temp_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_us), _, _, temp_u = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, args.no_class).scatter_(1, targets_x.view(-1,1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_us = inputs_u.cuda(), inputs_u2.cuda() , inputs_us.cuda()
            temp_x, temp_u = temp_x.cuda(), temp_u.cuda()

        # normalize classifier weights
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                w = model.module.fc.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                model.module.fc.weight.copy_(w)

                w2 = model.module.prototypes.weight.data.clone()
                w2 = F.normalize(w2, dim=1, p=2)
                model.module.prototypes.weight.copy_(w2)

                w3 = model.module.concept_prototypes.weight.data.clone()
                w3 = F.normalize(w3, dim=1, p=2)
                model.module.concept_prototypes.weight.copy_(w3)

            else:
                w = model.fc.weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                model.fc.weight.copy_(w)


                w2 = model.prototypes.weight.data.clone()
                w2 = F.normalize(w2, dim=1, p=2)
                model.prototypes.weight.copy_(w2)
                
                w3 = model.concept_prototypes.weight.data.clone()
                w3 = F.normalize(w3, dim=1, p=2)
                model.concept_prototypes.weight.copy_(w3)


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u, sub_outputs_u , concept_outputs_u = model(inputs_u,return_feats=True)
            outputs_u2 , sub_outputs_u2 , concept_outputs_u2 = model(inputs_u2,return_feats=True)
            _ , sub_targets_l , concept_targets_l = model(inputs_x,return_feats=True)
            # cross pseudo-labeling
            targets_u = sinkhorn(outputs_u2)
            targets_u2 = sinkhorn(outputs_u)


            sub_targets_u = sinkhorn(sub_outputs_u2)
            sub_targets_u2 = sinkhorn(sub_outputs_u)


            concept_targets_u = sinkhorn(concept_outputs_u2)
            concept_targets_u2= sinkhorn(concept_outputs_u)


            _, sub_hard_targets_l =  torch.max(sub_targets_l, dim=-1)
            _, sub_hard_targets_u =  torch.max(sub_targets_u, dim=-1)
            _, sub_hard_targets_u2 =  torch.max(sub_targets_u2, dim=-1)



            _, concept_hard_targets_l =  torch.max(concept_targets_l, dim=-1)
            _, concept_hard_targets_u =  torch.max(concept_targets_u, dim=-1)
            _, concept_hard_targets_u2 =  torch.max(concept_targets_u2, dim=-1)




        # generate hard pseudo-labels for confident novel class samples
        targets_u_novel = targets_u[:, args.no_seen:]
        max_pred_novel, _ = torch.max(targets_u_novel, dim=-1)
        hard_novel_idx1 = torch.where(max_pred_novel>=args.threshold)[0]

        targets_u2_novel = targets_u2[:,args.no_seen:]
        max_pred2_novel, _ = torch.max(targets_u2_novel, dim=-1)
        hard_novel_idx2 = torch.where(max_pred2_novel>=args.threshold)[0]

        targets_u[hard_novel_idx1] = targets_u[hard_novel_idx1].ge(args.threshold).float()
        targets_u2[hard_novel_idx2] = targets_u2[hard_novel_idx2].ge(args.threshold).float()

        # # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u2], dim=0) # (bz,num_class)

        all_concept_targets = torch.cat([concept_hard_targets_l, concept_hard_targets_u, concept_hard_targets_u2], dim=0) #(bz.) # hard label
        all_concept_targets  = torch.zeros( all_concept_targets.shape[0], args.num_concepts).cuda().scatter_(1,  all_concept_targets.view(-1,1).long(), 1)


        all_temp = torch.cat([temp_x, temp_u, temp_u], dim=0)
        all_sub_targets = torch.cat([sub_hard_targets_l, sub_hard_targets_u,sub_hard_targets_u2],dim=-1)
        all_sub_targets = torch.zeros(all_sub_targets.shape[0], args.num_protos).cuda().scatter_(1, all_sub_targets.view(-1,1).long(), 1)

        l = np.random.beta(args.alpha, args.alpha)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        sub_target_a, sub_target_b = all_sub_targets , all_sub_targets[idx]
        concept_target_a, concept_target_b = all_concept_targets , all_concept_targets[idx]

        temp_a, temp_b = all_temp, all_temp[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        mixed_temp = l * temp_a + (1 - l) * temp_b
        mixed_sub_target = l * sub_target_a + (1 - l) * sub_target_b
        mixed_concept_target = l * concept_target_a + (1 - l) * concept_target_b

        #interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)



        temp_logits, temp_sub_logits,temp_concept_logits = model(mixed_input[0],return_feats=True)
        logits = [temp_logits]
        sub_logits = [ temp_sub_logits ]
        concept_logits = [temp_concept_logits]


        for input in mixed_input[1:]:
            temp_logits, temp_sub_logits , temp_concept_logits = model(input,return_feats=True)
            logits.append(temp_logits)
            sub_logits.append(temp_sub_logits)
            concept_logits.append(temp_concept_logits)



        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        logits = torch.cat((logits_x, logits_u), 0)

        sub_logits = interleave(sub_logits, batch_size)
        sub_logits_x = sub_logits[0]
        sub_logits_u = torch.cat(sub_logits[1:], dim=0)
        sub_logits = torch.cat((sub_logits_x, sub_logits_u), 0)   

        concept_logits = interleave(concept_logits, batch_size)
        concept_logits_x = concept_logits[0]
        concept_logits_u = torch.cat(concept_logits[1:], dim=0)
        concept_logits = torch.cat((concept_logits_x, concept_logits_u), 0)   

        # cluster loss under sub-layer and super-layer
        # sub cluster
        sub_preds = F.softmax(sub_logits /  mixed_temp.unsqueeze(1), dim=1)
        sub_preds =  torch.clamp(sub_preds, min = 1e-8)
        sub_preds = torch.log(sub_preds)
        sim_loss = - torch.mean(torch.sum(mixed_sub_target * sub_preds, dim=1))


        # concept  cluster
        concept_preds = F.softmax(concept_logits /  mixed_temp.unsqueeze(1), dim=1)
        concept_preds =  torch.clamp(concept_preds, min = 1e-8)
        concept_preds = torch.log(concept_preds)
        sim_loss  += - torch.mean(torch.sum(mixed_concept_target * concept_preds, dim=1))

        #cross_entropy loss for sub to center
        sub_to_center_sim = model.sub_to_pre_sim() # [200,100]
        sub_to_center_logits =  sub_logits @ sub_to_center_sim

        sub_to_center_preds = F.softmax(sub_to_center_logits / mixed_temp.unsqueeze(1), dim=1)
        sub_to_center_preds =  torch.clamp(sub_to_center_preds, min = 1e-8)
        sub_to_center_preds = torch.log(sub_to_center_preds)
        # alignment loss 1
        loss = - torch.mean(torch.sum( mixed_target * sub_to_center_preds, dim=1))



        #cross_entropy loss for  center to concept
        center_to_concep_sim = model.cls_to_concept_sim() # [50,100]
        center_to_concep_logits =  concept_logits  @ center_to_concep_sim

        center_to_concep_preds = F.softmax(center_to_concep_logits / mixed_temp.unsqueeze(1), dim=1)
        center_to_concep_preds =  torch.clamp(center_to_concep_preds, min = 1e-8)
        center_to_concep_preds = torch.log(center_to_concep_preds)
        # alignment loss 2
        loss = - torch.mean(torch.sum( mixed_target * center_to_concep_preds, dim=1))

        

        #cross_entropy loss
        preds = F.softmax(logits /mixed_temp.unsqueeze(1), dim=1)
        preds =  torch.clamp(preds, min = 1e-8)
        preds = torch.log(preds)
        loss  -= torch.mean(torch.sum(mixed_target * preds, dim=1))


        # weark2strong view
        # logits_us = model(inputs_us)
        logits_us, sub_logits_us, concept_logits_us = model(inputs_us,return_feats=True)

        #cross_entropy loss
        preds_us = F.softmax(logits_us / temp_u.unsqueeze(1), dim=1)
        preds_us =  torch.clamp(preds_us, min = 1e-8)
        preds_us = torch.log(preds_us)
        # print(targets_u.shape)#torch.Size([256, 100])
        # print(preds_us.shape)#torch.Size([256, 100])
        loss -= 0.33 * 0.5 * ( torch.mean(torch.sum(targets_u* preds_us, dim=1)) + torch.mean(torch.sum(targets_u2* preds_us, dim=1)) )


        sub_preds_us = F.softmax(sub_logits_us / temp_u.unsqueeze(1), dim=1)
        sub_preds_us =  torch.clamp(sub_preds_us, min = 1e-8)
        sub_preds_us = torch.log(sub_preds_us)
        # print(sub_targets_u.shape) 
        # print(sub_preds_us.shape)
        loss -= 0.33 * 0.5 * ( torch.mean(torch.sum(sub_targets_u* sub_preds_us, dim=1)) + torch.mean(torch.sum(sub_targets_u2* sub_preds_us, dim=1)) )

    
        #cross_entropy loss
        concept_preds_us = F.softmax(concept_logits_us / temp_u.unsqueeze(1), dim=1)
        concept_preds_us =  torch.clamp(concept_preds_us, min = 1e-8)
        concept_preds_us = torch.log(concept_preds_us)

        loss -= 0.33 *  0.5 * ( torch.mean(torch.sum(concept_targets_u* concept_preds_us , dim=1)) + torch.mean(torch.sum(concept_targets_u2* concept_preds_us , dim=1)) )

        # record loss
        total_loss = loss + sim_loss
        losses.update(loss.item(), inputs_x.size(0))
        sim_losses.update(sim_loss.item(), inputs_x.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | sim_loss:{sim_loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    sim_loss = sim_losses.avg,
                    )
        bar.next()
    bar.finish()

    return losses.avg


def test_seen(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    gt_targets =[]
    predictions = []
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            _, max_idx = torch.max(outputs, dim=1)
            predictions.extend(max_idx.cpu().numpy().tolist())
            gt_targets.extend(targets.cpu().numpy().tolist())
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    predictions = np.array(predictions)
    gt_targets = np.array(gt_targets)

    predictions = torch.from_numpy(predictions)
    gt_targets = torch.from_numpy(gt_targets)
    eval_output = hungarian_evaluate(predictions, gt_targets, offset)

    return eval_output


if __name__ == '__main__':
    main()

