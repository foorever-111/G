import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import RPNHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class PARPNHead(RPNHead):
    """Prototype and angle guided RPN head."""

    def __init__(self,
                 num_protos,
                 proto_feat_dim,
                 angle_dim,
                 lambda_fg=0.5,
                 use_angle_gate=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_protos = num_protos
        self.proto_feat_dim = proto_feat_dim
        self.angle_dim = angle_dim
        self.lambda_fg = lambda_fg
        self.use_angle_gate = use_angle_gate

        self.rpn_sem_embed = nn.Conv2d(self.in_channels, proto_feat_dim, 1)

        if use_angle_gate:
            self.gate_fc = nn.Sequential(
                nn.Conv2d(proto_feat_dim + angle_dim, proto_feat_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(proto_feat_dim, 1, 1),
                nn.Sigmoid())
        else:
            self.gate_fc = None

        self.prototype_source = None

    def set_prototype_source(self, prototype_source):
        self.prototype_source = prototype_source

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = self.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)

        if self.prototype_source is not None:
            P_pure, _, _ = self.prototype_source.get_prototypes()
            P_ang_raw = self.prototype_source.get_angle_mean()

            z_sem = self.rpn_sem_embed(x)
            N, D, H, W = z_sem.shape
            z_flat = z_sem.permute(0, 2, 3, 1).reshape(-1, D)
            z_flat = F.normalize(z_flat, dim=1)

            m_sem = torch.matmul(z_flat, P_pure.t())
            s_proto, _ = m_sem.max(dim=1)
            s_proto = s_proto.view(N, 1, H, W)

            if self.use_angle_gate and self.gate_fc is not None:
                P_ep_ang = P_ang_raw.mean(dim=0, keepdim=True)
                P_ep_ang_map = P_ep_ang.view(1, self.angle_dim, 1, 1).expand(
                    N, -1, H, W)
                gate_in = torch.cat([z_sem, P_ep_ang_map], dim=1)
                alpha = self.gate_fc(gate_in)
                s_proto = alpha * s_proto

            rpn_cls_score = rpn_cls_score + self.lambda_fg * s_proto

        return rpn_cls_score, rpn_bbox_pred
