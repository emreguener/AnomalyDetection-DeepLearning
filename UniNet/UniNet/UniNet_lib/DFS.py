import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainRelated_Feature_Selection(nn.Module):
    def __init__(self, num_channels=256):
        super(DomainRelated_Feature_Selection, self).__init__()
        self.num_channels = num_channels

        # Theta parametreleri (başlangıçta 1 ile başlatıldı)
        self.theta1 = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.theta2 = nn.Parameter(torch.ones(1, num_channels * 2, 1, 1))
        self.theta3 = nn.Parameter(torch.ones(1, num_channels * 4, 1, 1))

    def forward(self, xs, priors, learnable=True, conv=False, max=True):
        features = []

        for idx, (x, prior) in enumerate(zip(xs, priors)):
            if x is None or prior is None:
                continue

            theta = 1.0
            if learnable:
                theta_name = f"theta{idx + 1}" if idx < 3 else f"theta{idx - 2}"
                theta_val = getattr(self, theta_name)
                theta = torch.clamp(torch.sigmoid(theta_val) + 0.5, max=1.0)

            b, c, h, w = x.shape
            prior_flat = prior.view(b, c, -1)

            if max:
                prior_max = prior_flat.max(dim=-1, keepdim=True)[0]
                prior_flat = prior_flat - prior_max

            weights = F.softmax(prior_flat, dim=-1).view(b, c, h, w)
            global_inf = prior.mean(dim=(-2, -1), keepdim=True)
            inter_weights = weights * (theta + global_inf)

            x_ = x * inter_weights
            features.append(x_)

        return features if features else None


def domain_related_feature_selection(xs, priors, max=True):
    features_list = []
    theta = 1
    for (x, prior) in zip(xs, priors):
        b, c, h, w = x.shape

        prior_flat = prior.view(b, c, -1)
        if max:
            prior_flat_ = prior_flat.max(dim=-1, keepdim=True)[0]
            prior_flat = prior_flat - prior_flat_
        weights = F.softmax(prior_flat, dim=-1).view(b, c, h, w)
        global_inf = prior.mean(dim=(-2, -1), keepdim=True)
        inter_weights = weights * (theta + global_inf)

        x_ = x * inter_weights
        features_list.append(x_)

    return features_list
