import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import HEADS

from mmfewshot.detection.models.utils import ACDPrototypeBank
from .kd_bbox_head import DisKDBBoxHead


@HEADS.register_module()
class ACDDisKDBBoxHead(DisKDBBoxHead):
    """Disentangled prototype-enhanced KDBBox head."""

    def __init__(self,
                 angle_dim=8,
                 lambda_dec=0.1,
                 lambda_sem=0.5,
                 lambda_ang=0.2,
                 mix_weight=0.5,
                 prototype_bank=None,
                 use_running_proto=True,
                 proto_momentum=0.9,
                 **kwargs):
        super().__init__(**kwargs)
        self.angle_dim = angle_dim
        self.lambda_dec = lambda_dec
        self.lambda_sem = lambda_sem
        self.lambda_ang = lambda_ang
        self.mix_weight = mix_weight
        self.use_running_proto = use_running_proto
        self.proto_momentum = proto_momentum

        feat_dim = self.fc_cls.in_features

        self.register_buffer('category_proto_init',
                             torch.randn(self.num_classes, feat_dim))
        self.register_buffer('angle_proto_init',
                             torch.randn(self.num_classes, angle_dim))
        self.angle_proj = nn.Linear(angle_dim, feat_dim, bias=False)

        self.sem_embed = nn.Linear(feat_dim, feat_dim, bias=False)
        self.ang_embed = nn.Linear(feat_dim, feat_dim, bias=False)

        # Use the existing classifier weights to initialize the category
        # prototypes for a more stable starting point.
        with torch.no_grad():
            if self.fc_cls.weight.shape == self.category_proto_init.shape:
                self.category_proto_init.copy_(self.fc_cls.weight.data)

        if prototype_bank is None:
            self.prototype_bank = ACDPrototypeBank(
                self.num_classes, feat_dim, angle_dim,
                momentum=self.proto_momentum)
        elif isinstance(prototype_bank, ACDPrototypeBank):
            self.prototype_bank = prototype_bank
        else:
            raise TypeError('prototype_bank must be an ACDPrototypeBank or None.')
        self.prototype_bank.set_angle_projector(self.angle_proj)
        self.prototype_bank.init_semantic(self.category_proto_init)
        self.prototype_bank.ang_mean.copy_(self.angle_proto_init)

        self._last_L_dec = None
        self._last_sem_feat = None
        self._last_ang_feat = None

    def _decouple_prototypes(self):
        """Decouple category and angle prototypes."""
        return self.prototype_bank.get_prototypes(self.angle_proj)

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
        self._last_sem_feat = z_sem.detach()
        self._last_ang_feat = z_ang.detach()
        m_sem = torch.matmul(z_sem, P_pure.t())       # [N, num_classes]
        m_ang = torch.matmul(z_ang, P_ang_proj.t())   # [N, num_classes]

        # ACD semantic+angle logits
        cls_score_acd = self.lambda_sem * m_sem + self.lambda_ang * m_ang

        # Interpolate between base classifier and ACD logits
        C_all = cls_score_base.size(1)
        C_fg = self.num_classes

        if C_all == C_fg:
            # No explicit background class
            cls_score = ((1 - self.mix_weight) * cls_score_base +
                         self.mix_weight * cls_score_acd)
        elif C_all == C_fg + 1:
            # Only add metric logits to foreground classes
            fg_score = ((1 - self.mix_weight) * cls_score_base[:, :C_fg] +
                        self.mix_weight * cls_score_acd)
            bg_score = cls_score_base[:, C_fg:].clone()
            cls_score = torch.cat([fg_score, bg_score], dim=1)
        else:
            raise RuntimeError(
                f'Unexpected cls_score_base dim: {C_all}, num_classes={C_fg}'
            )

        if self.training:
            self._last_L_dec = L_dec

        if return_fc_feat:
            return cls_score, bbox_preds, x
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
        if self.training and self.use_running_proto:
            self.prototype_bank.update(self._last_sem_feat, self._last_ang_feat,
                                       labels)

        losses = super().loss(cls_score, bbox_pred, rois, labels, label_weights,
                              bbox_targets, bbox_weights, cos_dis,
                              reduction_override)
        if self.training and getattr(self, '_last_L_dec', None) is not None:
            losses['loss_decouple'] = self.lambda_dec * self._last_L_dec
        return losses
