"""
PyTorch re-implementation of the ClassAwareRegularization (CAR) loss/module
(修复：避免在 forward 中创建未注册的层；正确更新 running buffer)
"""
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H * W, C)

def _one_hot_labels(labels: torch.Tensor, num_class: int, ignore_index: int = 255, device=None):
    B, H, W = labels.shape
    flat = labels.reshape(B, -1)  # [B, HW]
    device = device if device is not None else labels.device
    one_hot = torch.zeros(B, flat.shape[1], num_class, device=device, dtype=torch.float32)  # [B, HW, C]
    valid_mask = (flat != ignore_index)
    if valid_mask.any():
        batch_idx, pos_idx = torch.where(valid_mask)
        idx = flat[valid_mask].long()
        one_hot[batch_idx, pos_idx, idx] = 1.0
    return one_hot, valid_mask.reshape(B, H * W)


class ClassAwareRegularization(nn.Module):
    def __init__(
        self,
        train_mode: bool = False,
        use_inter_class_loss: bool = True,
        use_intra_class_loss: bool = True,
        intra_class_loss_remove_max: bool = False,
        use_inter_c2c_loss: bool = True,
        use_inter_c2p_loss: bool = False,
        intra_class_loss_rate: float = 1.0,
        inter_class_loss_rate: float = 1.0,
        num_class: int = 21,
        ignore_label: int = 255,
        pooling_rates: Optional[List[int]] = None,
        use_batch_class_center: bool = True,
        use_last_class_center: bool = False,
        last_class_center_decay: float = 0.9,
        inter_c2c_loss_threshold: float = 0.5,
        inter_c2p_loss_threshold: float = 0.25,
        filters: int = 512,
        apply_convs: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.train_mode = train_mode
        self.use_inter_class_loss = use_inter_class_loss
        self.use_intra_class_loss = use_intra_class_loss
        self.intra_class_loss_remove_max = intra_class_loss_remove_max
        self.use_inter_c2c_loss = use_inter_c2c_loss
        self.use_inter_c2p_loss = use_inter_c2p_loss
        self.intra_class_loss_rate = intra_class_loss_rate
        self.inter_class_loss_rate = inter_class_loss_rate
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.pooling_rates = pooling_rates or [1]
        self.use_batch_class_center = use_batch_class_center
        self.use_last_class_center = use_last_class_center
        self.last_class_center_decay = float(last_class_center_decay)
        self.inter_c2c_loss_threshold = float(inter_c2c_loss_threshold)
        self.inter_c2p_loss_threshold = float(inter_c2p_loss_threshold)
        self.filters = filters
        self.apply_convs = apply_convs
        self.device = device

        # running (last) class centers for momentum update (not trainable)
        if self.use_last_class_center:
            self.register_buffer("last_class_center", torch.zeros(1, self.num_class, self.filters, dtype=torch.float32))

        # tiny convs if desired (kept for API parity; optional)
        if self.apply_convs:
            self.linear_conv = nn.Conv2d(self.filters, self.filters, kernel_size=1, bias=True)
            self.end_conv = nn.Conv2d(self.filters, self.filters, kernel_size=1, bias=False)
            self.end_bn = nn.BatchNorm2d(self.filters)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.linear_conv = None

        # projection conv if feature channels != filters.
        # We create it lazily and register properly on first forward to avoid forward-time-only creation.
        self.proj = None  # will be set to nn.Conv2d(...) if needed

    def _create_proj_if_needed(self, in_channels):
        if self.proj is None and in_channels != self.filters:
            proj = nn.Conv2d(in_channels, self.filters, kernel_size=1, bias=True)
            # register module so it's part of parameters
            self.add_module("proj", proj)
            self.proj = proj

    def _compute_class_sum_and_counts(self, feats_flat: torch.Tensor, one_hot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, HW, D = feats_flat.shape
        _, _, C = one_hot.shape
        class_sum = torch.bmm(one_hot.permute(0, 2, 1), feats_flat)  # [B, C, D]
        counts = one_hot.sum(dim=1)  # [B, C]
        return class_sum, counts

    @staticmethod
    def _compute_class_centers(class_sum: torch.Tensor, counts: torch.Tensor, eps: float = 1e-6):
        counts_unsq = counts.unsqueeze(-1)  # [B,C,1]
        centers = class_sum / (counts_unsq + eps)
        return centers

    def _update_running_centers(self, centers: torch.Tensor, counts: torch.Tensor):
        present = (counts > 0).float().unsqueeze(-1)  # [B, C, 1]
        summed_centers = (centers * present).sum(dim=0, keepdim=True)  # [1, C, D]
        present_counts = present.sum(dim=0, keepdim=True).squeeze(-1)  # [1, C]
        mean_centers = summed_centers / (present_counts.unsqueeze(-1) + 1e-6)  # [1, C, D]
        # update buffer in-place where present_counts>0
        mask = (present_counts.squeeze(-1) > 0).float().unsqueeze(-1)  # [1, C, 1]
        new = self.last_class_center * self.last_class_center_decay + mean_centers * (1 - self.last_class_center_decay)
        # in-place update to preserve buffer registration
        self.last_class_center[mask.bool().squeeze(-1)] = new[mask.bool().squeeze(-1)]

    def _inter_class_c2c_loss(self, centers: torch.Tensor, threshold: float) -> torch.Tensor:
        if centers.dim() == 3:
            centers = centers.mean(dim=0, keepdim=True)
        centers = centers.squeeze(0)
        C, D = centers.shape
        if C <= 1:
            return centers.new_tensor(0.0)
        c_norm = F.normalize(centers, p=2, dim=1)
        sim = torch.matmul(c_norm, c_norm.t())
        mask = ~torch.eye(C, dtype=torch.bool, device=sim.device)
        sim_offdiag = sim[mask].view(C, C - 1)
        violations = F.relu(sim_offdiag - threshold)
        return violations.sum() / (C + 1e-6)

    def _pixel_inter_class_c2p_loss(self, feats_flat: torch.Tensor, centers: torch.Tensor, one_hot: torch.Tensor, threshold: float) -> torch.Tensor:
        B, HW, D = feats_flat.shape
        _, _, C = one_hot.shape
        loss = feats_flat.new_tensor(0.0)
        total = 0
        for b in range(B):
            non_ignore_mask = one_hot[b].sum(dim=1) > 0
            if non_ignore_mask.sum() == 0:
                continue
            feats_b = feats_flat[b, non_ignore_mask]
            labels_b = torch.argmax(one_hot[b, non_ignore_mask], dim=1)
            centers_b = centers[b]
            c_norm = F.normalize(centers_b, dim=1)
            f_norm = F.normalize(feats_b, dim=1)
            sims = torch.matmul(f_norm, c_norm.t())
            sims[range(sims.shape[0]), labels_b] = -1.0
            violations = F.relu(sims - threshold)
            loss = loss + violations.sum()
            total += violations.numel()
        if total == 0:
            return loss
        return loss / (total + 1e-6)

    def _intra_class_absolute_loss(self, feats_flat: torch.Tensor, centers: torch.Tensor, one_hot: torch.Tensor, remove_max: bool = False, not_ignore_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, HW, D = feats_flat.shape
        _, _, C = one_hot.shape
        loss = feats_flat.new_tensor(0.0)
        total_count = 0
        for b in range(B):
            one_hot_b = one_hot[b]
            if one_hot_b.sum() == 0:
                continue
            feats_b = feats_flat[b]
            centers_b = centers[b]
            for cls in range(C):
                cls_mask = one_hot_b[:, cls].bool()
                if cls_mask.sum() == 0:
                    continue
                cls_feats = feats_b[cls_mask]
                center = centers_b[cls].unsqueeze(0)
                dists = (cls_feats - center).pow(2).sum(dim=1)
                if remove_max and dists.numel() > 1:
                    dists = dists.sort()[0][:-1]
                loss = loss + dists.sum()
                total_count += cls_feats.shape[0]
        if total_count == 0:
            return loss
        return loss / (total_count + 1e-6)

    def forward(self, features: torch.Tensor, label: Optional[torch.Tensor] = None, training: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        if training is None:
            training = self.train_mode
        device = self.device or features.device
        features = features.to(device)

        B, C_feat, H, W = features.shape

        # create proj if needed and register it
        self._create_proj_if_needed(C_feat)

        x = features
        if self.linear_conv is not None:
            if self.proj is not None:
                x = self.proj(x)
            else:
                x = self.linear_conv(x)

        y = x.clone()

        intra_loss_total = torch.tensor(0.0, device=device)
        inter_loss_total = torch.tensor(0.0, device=device)
        c2p_loss_total = torch.tensor(0.0, device=device)

        if training and label is not None:
            for rate in self.pooling_rates:
                x_sub = x
                if rate > 1:
                    x_sub = F.avg_pool2d(x_sub, kernel_size=rate, stride=rate, padding=0)

                feats_flat = _flatten_hw(x_sub)
                _, _, Hs, Ws = x_sub.shape
                if label.shape[1:] != (Hs, Ws):
                    label_resized = F.interpolate(label.unsqueeze(1).float(), size=(Hs, Ws), mode="nearest").squeeze(1).long()
                else:
                    label_resized = label
                one_hot, not_ignore_mask = _one_hot_labels(label_resized, self.num_class, ignore_index=self.ignore_label, device=device)
                class_sum, counts = self._compute_class_sum_and_counts(feats_flat, one_hot)
                if self.use_batch_class_center:
                    centers = self._compute_class_centers(class_sum, counts)
                    if self.use_last_class_center:
                        self._update_running_centers(centers, counts)
                        centers = self.last_class_center.repeat(B, 1, 1)
                else:
                    centers = self._compute_class_centers(class_sum, counts)

                if self.use_inter_class_loss and training:
                    if self.use_inter_c2c_loss:
                        inter_loss_total = inter_loss_total + self._inter_class_c2c_loss(centers, self.inter_c2c_loss_threshold)
                    if self.use_inter_c2p_loss:
                        c2p_loss_total = c2p_loss_total + self._pixel_inter_class_c2p_loss(feats_flat, centers, one_hot, self.inter_c2p_loss_threshold)

                if self.use_intra_class_loss:
                    intra_loss_total = intra_loss_total + self._intra_class_absolute_loss(feats_flat, centers, one_hot, remove_max=self.intra_class_loss_remove_max)

        inter_loss_total = inter_loss_total * self.inter_class_loss_rate
        intra_loss_total = intra_loss_total * self.intra_class_loss_rate
        total_loss = inter_loss_total + intra_loss_total + c2p_loss_total

        if self.apply_convs:
            y = self.end_conv(y)
            y = self.end_bn(y)
            y = F.relu(y)

        return {
            "loss": total_loss,
            "intra_loss": intra_loss_total.detach() if intra_loss_total is not None else torch.tensor(0.0, device=device),
            "inter_loss": inter_loss_total.detach() if inter_loss_total is not None else torch.tensor(0.0, device=device),
            "c2p_loss": c2p_loss_total.detach() if c2p_loss_total is not None else torch.tensor(0.0, device=device),
        }