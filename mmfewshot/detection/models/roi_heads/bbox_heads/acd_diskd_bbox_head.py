import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import HEADS

from .kd_bbox_head import DisKDBBoxHead


@HEADS.register_module()
class ACDDisKDBBoxHead(DisKDBBoxHead):
    """Disentangled prototype-enhanced KDBBox head."""

    def __init__(self,
                 angle_dim=8,
                 lambda_dec=0.1,
                 lambda_sem=0.5,
                 lambda_ang=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.angle_dim = angle_dim
        self.lambda_dec = lambda_dec
        self.lambda_sem = lambda_sem
        self.lambda_ang = lambda_ang

        feat_dim = self.fc_cls.in_features

        self.category_proto_init = nn.Parameter(
            torch.randn(self.num_classes, feat_dim))
        self.angle_proto = nn.Parameter(
            torch.randn(self.num_classes, angle_dim))
        self.angle_proj = nn.Linear(angle_dim, feat_dim, bias=False)

        self.sem_embed = nn.Linear(feat_dim, feat_dim, bias=False)
        self.ang_embed = nn.Linear(feat_dim, feat_dim, bias=False)

        self._last_L_dec = None

    def _decouple_prototypes(self):
        """Decouple category and angle prototypes."""
        P_init = self.category_proto_init
        P_ang = self.angle_proto

        P_ang_proj = self.angle_proj(P_ang)

        P_init_norm = F.normalize(P_init, dim=1)
        P_ang_norm = F.normalize(P_ang_proj, dim=1)
        cos_sim = (P_init_norm * P_ang_norm).sum(dim=1)

        dot = (P_init * P_ang_proj).sum(dim=1, keepdim=True)
        ang_norm2 = (P_ang_proj ** 2).sum(dim=1, keepdim=True) + 1e-6
        proj = dot / ang_norm2 * P_ang_proj
        P_pure = P_init - proj
        P_pure = F.normalize(P_pure, dim=1)

        L_dec = (cos_sim ** 2).mean()
        return P_pure, F.normalize(P_ang_proj, dim=1), L_dec

    def forward(self, x, return_fc_feat=False):
        kd_loss_list = []
        x = x.flatten(1)
        alpha = self.base_alpha
        base_x = x
        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            kd_loss_list.append(torch.frobenius_norm(base_x - x, dim=-1))
            x = self.relu(x)
        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss

        self._last_L_dec = None

        bbox_preds = self.fc_reg(x)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                                       self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        cls_score_base = self.scale * cos_dist

        P_pure, P_ang_proj, L_dec = self._decouple_prototypes()
        z_sem = F.normalize(self.sem_embed(x), dim=1)
        z_ang = F.normalize(self.ang_embed(x), dim=1)
        m_sem = torch.matmul(z_sem, P_pure.t())
        m_ang = torch.matmul(z_ang, P_ang_proj.t())

        cls_score = cls_score_base + self.lambda_sem * m_sem + self.lambda_ang * m_ang

        if self.training:
            self._last_L_dec = L_dec

        return cls_score, bbox_preds

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             cos_dis=None,
             reduction_override=None):
        losses = super().loss(cls_score, bbox_pred, rois, labels, label_weights,
                              bbox_targets, bbox_weights, cos_dis,
                              reduction_override)
        if self.training and getattr(self, '_last_L_dec', None) is not None:
            losses['loss_decouple'] = self.lambda_dec * self._last_L_dec
        return losses
