import torch.nn as nn
import torch.nn.functional as F

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits=8, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training:
            if N % self.num_splits != 0:
                raise ValueError(
                    f"Batch size {N} must be divisible by "
                    f"num_splits={self.num_splits}"
                )

            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = F.batch_norm(
                input.reshape(-1, C * self.num_splits, H, W),
                running_mean_split,
                running_var_split,
                self.weight.repeat(self.num_splits),
                self.bias.repeat(self.num_splits),
                True,
                self.momentum,
                self.eps,
            ).reshape(N, C, H, W)

            self.running_mean.copy_(
                running_mean_split.reshape(self.num_splits, C).mean(dim=0)
            )
            self.running_var.copy_(
                running_var_split.reshape(self.num_splits, C).mean(dim=0)
            )

            return outcome

        return super().forward(input)