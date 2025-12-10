import torch
import torch.nn as nn
import torch.nn.functional as F


class ACDPrototypeBank(nn.Module):
    """Maintain running prototypes for semantic and angle branches."""

    def __init__(self, num_classes, feat_dim, angle_dim, momentum=0.9, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.angle_dim = angle_dim
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('sem_mean', torch.zeros(num_classes, feat_dim))
        self.register_buffer('ang_mean', torch.zeros(num_classes, angle_dim))
        self.register_buffer('counts', torch.zeros(num_classes))

        self.angle_proj = None

    def set_angle_projector(self, projector: nn.Module):
        """Attach the learnable angle projector used for decoupling."""
        self.angle_proj = projector

    def init_semantic(self, init_weight: torch.Tensor):
        if init_weight.shape == self.sem_mean.shape:
            self.sem_mean.copy_(init_weight.detach())

    @torch.no_grad()
    def update(self, sem_feat: torch.Tensor, ang_feat: torch.Tensor,
               labels: torch.Tensor):
        if sem_feat is None or labels is None:
            return

        for c in range(self.num_classes):
            mask = labels == c
            if not mask.any():
                continue

            feat_c = sem_feat[mask]
            ang_c = ang_feat[mask] if ang_feat is not None else None

            batch_mean_sem = feat_c.mean(dim=0)
            batch_mean_ang = ang_c.mean(dim=0) if ang_c is not None else None

            if self.counts[c] == 0:
                self.sem_mean[c] = batch_mean_sem.detach()
                if batch_mean_ang is not None:
                    self.ang_mean[c] = batch_mean_ang.detach()
                self.counts[c] = feat_c.size(0)
                continue

            m = self.momentum
            self.sem_mean[c] = m * self.sem_mean[c] + (1 - m) * batch_mean_sem
            if batch_mean_ang is not None:
                self.ang_mean[c] = m * self.ang_mean[c] + (1 - m) * batch_mean_ang
            self.counts[c] = self.counts[c] + feat_c.size(0)

    def _safe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1, eps=self.eps)

    def get_angle_mean(self):
        return self._safe_normalize(self.ang_mean)

    def get_prototypes(self, angle_proj: nn.Module = None):
        projector = angle_proj if angle_proj is not None else self.angle_proj
        assert projector is not None, 'Angle projector must be set before use.'

        P_init = self.sem_mean
        P_ang_proj = projector(self.ang_mean)

        P_init_norm = self._safe_normalize(P_init)
        P_ang_norm = self._safe_normalize(P_ang_proj)
        cos_sim = (P_init_norm * P_ang_norm).sum(dim=1)

        dot = (P_init * P_ang_proj).sum(dim=1, keepdim=True)
        ang_norm2 = (P_ang_proj ** 2).sum(dim=1, keepdim=True) + self.eps
        proj = dot / ang_norm2 * P_ang_proj
        P_pure = self._safe_normalize(P_init - proj)

        L_dec = (cos_sim ** 2).mean()
        return P_pure, self._safe_normalize(P_ang_proj), L_dec
