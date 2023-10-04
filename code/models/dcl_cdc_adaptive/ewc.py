import torch
import torch.autograd as autograd
from torch import nn
from torch.autograd import Variable


from torch.nn import functional as F

USE_CUDA = torch.cuda.is_available()


def Variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        var = var.cuda()
    return var


class EWC(nn.Module):
    def __init__(self, ewc_lambda, if_online):
        super().__init__()

        self.online = if_online
        self.ewc_lambda = ewc_lambda
        self.tasks_encountered = []

        self.fisher = {}
        self.optpar = {}

    def forward(self, named_params):
        net_loss = Variable(torch.Tensor([0]))
        if not self.ewc_lambda:
            return net_loss
        for task_id in self.tasks_encountered:
            for name, param in named_params:
                if param.requires_grad:
                    fisher = self.fisher[task_id][name].cuda()
                    optpar = self.optpar[task_id][name].cuda()
                    net_loss += (fisher * (optpar - param).pow(2)).sum()
        return net_loss * self.ewc_lambda / 2

    def regularize(self, named_params):
        """Calculate the EWC regulonelinearization component in the overall loss.
        For all the tasks encountered in past, L2-norm loss is calculated
        between current model parameters and optimal parameters of previous
        tasks, weighted by terms from fisher matrix.
        Arguments
        =========
        named_params : generator
            Named parameters of model to be regularized.
        """
        return self.forward(named_params)

    # Update the Fisher Information
    def update_fisher_optpar(self, model, current_itr, dataset, sample_size, batch_size=20, consolidate=True):
        """
            current_itr:
        """
        if consolidate:
            if self.online:
                current_itr = 1
                self.tasks_encountered = [1]
            else:
                self.tasks_encountered.append(current_itr)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=0,
                                                            shuffle=True, drop_last=True)
        losses = []
        for idx, batch_data in enumerate(data_loader):
            # x = batch_data[0].view(batch_size, -1) # image data
            x = batch_data[1].cuda()

            # x = Variable(x).cuda() if USE_CUDA else Variable(x)
            # import pdb; pdb.set_trace()
            y = batch_data[2]['spoofing_label'].cuda()

            losses.append(
                F.log_softmax(model(x), dim=1)[range(batch_size), y.data]
            )
            del x, y
            if len(losses) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        parameter_dict = {}
        for k, v in model.named_parameters():
            if v.requires_grad:
                parameter_dict[k] = v
        sample_losses = torch.cat(losses).unbind()
        sample_grads__ = [autograd.grad(l, parameter_dict.values(),  retain_graph=(i < len(sample_losses)))
                             for i, l in enumerate(sample_losses, 1)]
        sample_grads_ = zip(*sample_grads__)


        sample_grads = [torch.stack(gs) for gs in sample_grads_]
        fisher_diagonals = [(g ** 2).mean(0) for g in sample_grads]
        self.fisher[current_itr] = {}
        self.optpar[current_itr] = {}

        with torch.no_grad():
            # import pdb; pdb.set_trace()
            # for name, param, fisher in zip(parameter_dict, fisher_diagonals):
            for idx, name in enumerate(parameter_dict.keys()):
                fisher = fisher_diagonals[idx]
                param = parameter_dict[name]
                assert fisher.shape == param.shape
                self.optpar[current_itr][name] = param.data.clone().cpu() # Optimal model of previous task
                self.fisher[current_itr][name] = fisher.detach().cpu()

        """
        sample_losses = torch.cat(losses).unbind()

        for (name, param) in zip(model.named_parameters):
            if param.requires_grad:
                autograd(l, param, retain_graph=True)
        """