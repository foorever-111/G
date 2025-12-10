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
                 mix_weight=0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.angle_dim = angle_dim
        self.lambda_dec = lambda_dec
        self.lambda_sem = lambda_sem
        self.lambda_ang = lambda_ang
        self.mix_weight = mix_weight

        feat_dim = self.fc_cls.in_features

        self.category_proto_init = nn.Parameter(
            torch.randn(self.num_classes, feat_dim))
        self.angle_proto = nn.Parameter(
            torch.randn(self.num_classes, angle_dim))
        self.angle_proj = nn.Linear(angle_dim, feat_dim, bias=False)

        self.sem_embed = nn.Linear(feat_dim, feat_dim, bias=False)
        self.ang_embed = nn.Linear(feat_dim, feat_dim, bias=False)

        # Use the existing classifier weights to initialize the category
        # prototypes for a more stable starting point.
        with torch.no_grad():
            if self.fc_cls.weight.shape == self.category_proto_init.shape:
                self.category_proto_init.copy_(self.fc_cls.weight.data)

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
        losses = super().loss(cls_score, bbox_pred, rois, labels, label_weights,
                              bbox_targets, bbox_weights, cos_dis,
                              reduction_override)
        if self.training and getattr(self, '_last_L_dec', None) is not None:
            losses['loss_decouple'] = self.lambda_dec * self._last_L_dec
        return losses
