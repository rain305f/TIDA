import torch
import torch.nn as nn


def build_model(args, ema=False):
    if args.dataset in ['cifar10', 'cifar100', 'cifar100_20']:
        from . import resnet_cifar_hierarchical  as models 
    elif args.dataset == 'tinyimagenet':
        from . import resnet_tinyimagenet_hierarchical  as models
    else:
        from . import resnet_hierarchical  as models

    if args.arch == 'resnet18':
        model = models.resnet18(no_class=args.no_class,nmb_prototypes = args.num_protos, nmb_concepts= args.num_concepts)
    if args.arch == 'resnet50':
        model = models.resnet50(no_class=args.no_class,nmb_prototypes = args.num_protos, nmb_concepts= args.num_concepts)
    
    # use dataparallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model