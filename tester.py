
from tqdm import tqdm
from Baser import Baser
from timm.utils import AverageMeter
import torch
from utils import *
import json


class Tester(Baser):
    def __init__(self, args, model, test_loader, criterion, metric_fn=[accuracy], save_results=None):
        super().__init__(args, model, criterion, metric_fn)
        #self.model = model
        self.test_loader = test_loader
        self.save_results = save_results
        #self.criterion = criterion


    def test(self, mode='test'):
        l = AverageMeter()

        predicteds = torch.tensor([], requires_grad=False)
        targets = torch.tensor([], requires_grad=False)
        self.model.eval()
        for samples in tqdm(self.test_loader):
            output = self.forward(samples, require_grad=False)
            labels = samples['labels'].cuda(non_blocking=True)
            loss = self.compute_loss(output, labels)

            l.update(loss.item())

            predicted = output['predicted'].cpu()
            labels = labels.cpu()
            predicteds = torch.concat([predicteds, predicted], dim=0)
            targets = torch.concat([targets, labels], dim=0)

            self.output_attn(samples, output)
        metrics = self.metric(predicteds, targets)

        #save result as figure
        if self.save_results is not None:
            predicteds, targets = self.postprocess(predicteds, targets)
            for save in self.save_results:
                save(predicteds, targets, output_name=f'{self.args.output_dir}/{save.__name__}_{mode}.png')

        print(f'Test loss is {l.avg}')
        print(f'metrics is {metrics}')
            
        
            
