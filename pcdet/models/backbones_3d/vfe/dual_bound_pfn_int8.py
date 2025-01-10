from typing import Optional

import torch
import torch.nn as nn

import brevitas
from brevitas.nn import QuantLinear, QuantIdentity
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerChannelFloat,
)
from brevitas.core.zero_point import ParameterFromRuntimeZeroPoint
from dependencies import this

import torch_scatter

from .vfe_template import VFETemplate


class AverageStatOp(brevitas.jit.ScriptModule):
    # based on brevitas.core.stats.AbsAve class
    __constants__ = ['stats_reduce_dim']
 
    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(AverageStatOp, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        if self.stats_reduce_dim is None:
            return torch.mean(x)
        else:
            return torch.mean(x, dim=self.stats_reduce_dim)
        
class _ScaleShiftZeroPoint_Fixed(brevitas.jit.ScriptModule):
    # based on brevitas.core.zero_point._ScaleShiftZeroPoint
    __constants__ = ['quantize_zero_point']

    def __init__(self, int_quant: torch.nn.Module, quantize_zero_point: bool) -> None:
        super(_ScaleShiftZeroPoint_Fixed, self).__init__()
        self.int_quant = int_quant
        self.quantize_zero_point = quantize_zero_point

    @brevitas.jit.script_method
    def forward(self, zero_point: torch.Tensor, scale: torch.Tensor, bit_width: torch.Tensor) -> torch.Tensor:
        # bugged line: min_int = self.int_quant.min_int(bit_width)
        min_int = 0
        if self.quantize_zero_point:
            out = self.int_quant.to_int(scale, min_int, bit_width, zero_point)
        else:
            out = zero_point / scale + min_int
        return out

# this is an ugly workaround for bugged stats collection for zero-points in Brevitas
import brevitas.core.zero_point as bczp
bczp._ScaleShiftZeroPoint = _ScaleShiftZeroPoint_Fixed 

class QuantDualBoundPFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm = True,
                 last_layer = False,
                 input_quant = None,
                 weight_quant = None,
                 output_quant = None):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 3
        else:
            out_channels = out_channels // 2

        self.pre_quant = QuantIdentity(
            act_quant = input_quant,
            return_quant_tensor = False
        )

        if self.use_norm:
            self.linear = QuantLinear(
                in_channels, 
                out_channels, 
                bias = False,
                input_quant = None,
                weight_quant = weight_quant,
                output_quant = output_quant
            )
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = QuantLinear(
                in_channels, 
                out_channels, 
                bias = True,
                input_quant = None,
                weight_quant = weight_quant,
                output_quant = output_quant
            )

    def forward(self, inputs, unq_inv):
        x = inputs
        x = self.pre_quant(x)
        x = self.linear(x)
        x = self.norm(x) if self.use_norm else x
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        x_min = torch_scatter.scatter_min(x, unq_inv, dim=0)[0]
        x_out = torch.cat([x_min, x_max], dim=1)

        if self.last_vfe:
            return x_out
        else:
            x_concatenated = torch.cat([x, x_out[unq_inv, :]], dim=1)
            return x_concatenated


class DualBoundPillarFeatureNetInt8(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.with_spherical_coords = self.model_cfg.WITH_SPHERICAL_COORDS
        self.with_log_features = self.model_cfg.WITH_LOG_FEATURES
        if self.use_absolute_xyz:
            num_point_features += 6

        if self.with_spherical_coords:
            num_point_features += 2

        if self.with_log_features:
            num_point_features += 2

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)


        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]

            # These classes need to be defined dynamically, as per-channel quantisation requires information about
            # number of filters and currently Brevitas does not support lazy initialisation of quantisation parameters
            class Int8ActPerChannelFloat(Int8ActPerTensorFloat):
                scaling_per_output_channel = True
                scaling_stats_permute_dims = (1, 0)
                per_channel_broadcastable_shape = (1, in_filters)
            
            class Int8ActPerChannelFloat_RuntimeStatsZP(Int8ActPerChannelFloat):
                zero_point_impl = ParameterFromRuntimeZeroPoint
                zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl
                zero_point_stats_impl = AverageStatOp 
                zero_point_shape = this.scaling_shape
                quantize_zero_point = False

            pfn_layers.append(
                QuantDualBoundPFNLayer(
                    in_filters, 
                    out_filters, 
                    self.use_norm,
                    last_layer = (i >= len(num_filters) - 2),
                    input_quant = Int8ActPerChannelFloat_RuntimeStatsZP,
                    weight_quant = Int8WeightPerChannelFloat,
                    output_quant = None,
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size[:2]).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def quantize_pts(self, pts: torch.Tensor) -> torch.Tensor:
        # Quantizes points so that they are represented by two 8-bit (affine transformed) integers
        num_dims = pts.shape[1]
        pc_range = self.point_cloud_range.reshape(2, 3)[:, :num_dims]
        pc_min = pc_range[0, :]
        pc_max = pc_range[1, :]
        pc_diameter = pc_max - pc_min

        major_divider = pc_diameter / 256
        minor_divider = major_divider / 256

        major_divider = major_divider.reshape(1, num_dims)
        minor_divider = minor_divider.reshape(1, num_dims)
        pc_min = pc_min.reshape(1, num_dims)
        
        pts_major_q = torch.floor_divide(pts - pc_min, major_divider) * major_divider + pc_min
        pts_minor_q = pts - pts_major_q

        pts_q = torch.cat((pts_minor_q, pts_major_q), dim = 1)
        return pts_q
    
    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        # Note: deleted "with distance"

        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
        mask = ((points_coords >= 0) & (
            points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
            points_coords[:, 0] * self.scale_y + \
            points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(
            merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - \
            (points_coords[:, 0].to(points_xyz.dtype)
             * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - \
            (points_coords[:, 1].to(points_xyz.dtype)
             * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [f_center]
        if self.use_absolute_xyz:
            features.append(
                self.quantize_pts(points[:, 1:4])
            )
        features.append(points[:, 4:]) # Note: intensity and timestamp should already be quantized

        if self.with_spherical_coords or self.with_log_features:
            rxy = torch.hypot(points[:, 1], points[:, 2])
        else:
            rxy = None

        if self.with_spherical_coords:
            yaw = torch.atan2(points[:, 2], points[:, 1]).unsqueeze(1)
            pitch = torch.atan2(points[:, 3], rxy).unsqueeze(1)
            features.append(yaw)
            features.append(pitch)

        if self.with_log_features:
            log_intensity = torch.log(points[:, 4] + 1e-8).unsqueeze(1)
            log_rxy = torch.log(rxy + 1e-8).unsqueeze(1)
            features.append(log_intensity)
            features.append(log_rxy)

        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        return batch_dict
