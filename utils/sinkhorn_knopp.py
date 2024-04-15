import torch
import numpy as np


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module): #完成聚类，并且满足类的先验分布~
    def __init__(self, args):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk
        # self.epsilon2 = args.epsilon_kl
        self.imb_factor = args.imb_factor

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1] # Samples

        # obtain permutation/order from the marginals
        marginals_argsort = torch.argsort(Q.sum(1))
        marginals_argsort = marginals_argsort.detach()
        r = []
        for i in range(Q.shape[0]): # Classes
            r.append( ( 1/self.imb_factor )**(i / (Q.shape[0] - 1.0)) )


        r = np.array(r)
        r = r * (Q.shape[1]/Q.shape[0]) # Per-class distribution in the mini-batch
        r = torch.from_numpy(r).cuda(non_blocking=True)
        r[marginals_argsort] = torch.sort(r)[0] # Sort/permute based on the data order  
        r = torch.clamp(r, min=1) # Clamp the min to have a balance distribution for the tail classes
        r /= r.sum() # Scaling to make it prob
        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0) #保证每个样本所在的行，他的和为 1/bz
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


    @torch.no_grad()
    def forward(self, logits , prior_P=None ):
        # get assignments
        # import pdb; pdb.set_trace()
        B , K = logits.shape
        if prior_P is not None:
            denominator = self.epsilon + self.epsilon2
            prior_withreg = - torch.log(prior_P/B ) * self.epsilon2
            q = (logits + prior_withreg ) / denominator
        else:
            q = logits / self.epsilon
        # 归一化
        M = torch.max(q)
        q -= M

        q = torch.exp(q).t()
        return self.iterate(q)


    # def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    #     L = torch.exp(out / epsilon).t()  # K x B
    #     B = L.shape[1]
    #     K = L.shape[0]
    
    #     # make the matrix sums to 1
    #     sum_L = torch.sum(L)
    #     L /= sum_L
    
    #     for _ in range(sinkhorn_iterations):
    #         L /= torch.sum(L, dim=1, keepdim=True)
    #         L /= K
    
    #         L /= torch.sum(L, dim=0, keepdim=True)
    #         L /= B
    
    #     L *= B
    #     L = L.t()
    
    #     indexs = torch.argmax(L, dim=1)
    #     # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    #     L = F.gumbel_softmax(L, tau=0.5, hard=True)
    
    #     return L, indexs
  
# @torch.no_grad()
# def distributed_sinkhorn(out):
#     Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
#     B = Q.shape[1] * args.world_size # number of samples to assign
#     K = Q.shape[0] # how many prototypes

#     # make the matrix sums to 1
#     sum_Q = torch.sum(Q)
#     dist.all_reduce(sum_Q)
#     Q /= sum_Q

#     for it in range(args.sinkhorn_iterations):
#         # normalize each row: total weight per prototype must be 1/K
#         sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
#         dist.all_reduce(sum_of_rows)
#         Q /= sum_of_rows
#         Q /= K

#         # normalize each column: total weight per sample must be 1/B
#         Q /= torch.sum(Q, dim=0, keepdim=True)
#         Q /= B

#     Q *= B # the colomns must sum to 1 so that Q is an assignment
#     return Q.t()
