import torch 
from utils import *

class Baser():

    def __init__(self, args, model, criterion, metric_fn=[accuracy]) -> None:
        self.args = args
        self.model = model
        self.criterion = criterion
        self.metric_fn = metric_fn

    def forward(self, samples, require_grad=True):
        imgs = samples['imgs'].cuda(non_blocking=True)
        labels=samples['labels'].cuda(non_blocking=True)

        if require_grad:
            # forward
            with torch.amp.autocast('cuda'):
                output = self.model(imgs, len_list=samples['imgs_len'], labels=labels, view_labels=samples['view_labels'])
        else:
            with torch.no_grad():
                #output = self.model(imgs, len_list=samples['imgs_len'])
                output = self.model(imgs, len_list=samples['imgs_len'], labels=labels, view_labels=samples['view_labels'])

        return output

    def compute_loss(self, output, labels):
        if self.args.ctiterion == 'in':
            return output['loss']
        return self.criterion(output['predicted'], labels)
    
    def metric(self, predicteds, targets):
        metrics = {}

        predicteds, targets = self.postprocess(predicteds, targets)

        for metric_fn in self.metric_fn:
            metrics[metric_fn.__name__] = metric_fn(predicteds, targets)

        return metrics

    def postprocess(self, predicteds, targets):
        predicteds = F.softmax(predicteds, dim=-1)
        predicteds = torch.argmax(predicteds, dim=-1).detach().cpu()
        targets = targets.detach().cpu()

        return predicteds, targets

