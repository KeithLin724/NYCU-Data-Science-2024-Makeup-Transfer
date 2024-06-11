import torch
import torch.nn as nn
from torch.autograd import Variable


class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=True,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, input_tensor: torch.Tensor, target_is_real: bool):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (
                self.real_label_var.numel() != input_tensor.numel()
            )
            if create_label:
                # real_tensor = self.Tensor(input_tensor.size()).fill_(self.real_label)
                real_tensor = torch.full_like(input_tensor, self.real_label)

                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (
                self.fake_label_var.numel() != input_tensor.numel()
            )
            if create_label:
                # fake_tensor = self.Tensor(input_tensor.size()).fill_(self.fake_label)
                fake_tensor = torch.full_like(input_tensor, self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input_tensor: torch.Tensor, target_is_real: bool):
        target_tensor = self.get_target_tensor(input_tensor, target_is_real)
        return self.loss(input_tensor, target_tensor)
