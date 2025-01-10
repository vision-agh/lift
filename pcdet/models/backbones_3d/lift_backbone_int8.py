import torch
import torch.nn as nn

from brevitas.nn import QuantIdentity
from brevitas.quant_tensor import QuantTensor
from brevitas.quant import (
    Uint8ActPerTensorFloat,
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
)
from spconv.pytorch.core import SparseConvTensor
import spconv.pytorch as spconv

from ...utils.spconv_brevitas import QuantSparseConvTensor
from ...utils.spconv_brevitas_rep_layers import (
    QuantRepSubMConv2d,
    QuantRepSparseConv2d,
    QuantRepSparseConv2dDownsampleX2
)


class QuantSparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, input_quant, weight_quant, act_quant, return_quant_tensor=False, act_quant2=None, return_quant_tensor2=False, indice_key=None, ):
        super(QuantSparseBasicBlock, self).__init__()
        if act_quant2 is None:
            act_quant2 = act_quant
        if return_quant_tensor2 is None:
            return_quant_tensor2 = return_quant_tensor

        self.conv1 = QuantRepSubMConv2d(
            inplanes, 
            planes, 
            indice_key, 
            input_quant = input_quant,
            weight_quant = weight_quant,
            act_quant = act_quant,
            return_quant_tensor = return_quant_tensor,
        )
        self.conv2 = QuantRepSubMConv2d(
            planes, 
            planes, 
            indice_key,
            input_quant = input_quant,
            weight_quant = weight_quant,
            act_quant = act_quant2,
            return_quant_tensor = return_quant_tensor2,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class LiFTBackboneInt8(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[[1, 0]]

        input_feats = 64

        class InputSparseFeaturesQuant(Int8ActPerTensorFloat):
            scaling_per_output_channel = True
            scaling_stats_permute_dims = (1, 0)
            per_channel_broadcastable_shape = (1, input_feats)

        self.quantize_input_feats = QuantIdentity(
            act_quant = InputSparseFeaturesQuant,
            return_quant_tensor = False,
        )

        self.bev_out_quantise = QuantIdentity(
            act_quant = Int8ActPerTensorFloat,
            return_quant_tensor = False,
        )

        input_quant = None
        weight_quant = Int8WeightPerTensorFloat
        act_quant = Uint8ActPerTensorFloat
        return_quant_tensor = False
        block_shared_kwargs = dict(
            input_quant = input_quant, 
            weight_quant = weight_quant, 
            act_quant = act_quant, 
            return_quant_tensor = return_quant_tensor
        )

        current_feats = input_feats # 64
        self.conv1 = spconv.SparseSequential(
            QuantRepSparseConv2dDownsampleX2(input_feats, input_feats, indice_key="spconv1", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res1", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res1", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res1", **block_shared_kwargs),
        )

        current_feats = current_feats # 64
        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            QuantRepSparseConv2dDownsampleX2(current_feats, current_feats, indice_key="spconv3", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res3", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res3", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res3", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res3", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res3", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res3", **block_shared_kwargs),
        )

        current_feats *= 2 #128
        self.conv3_5 = spconv.SparseSequential(
            QuantSparseBasicBlock(current_feats // 2, current_feats, indice_key="res3", return_quant_tensor2=False, **block_shared_kwargs), # idx 0
        )

        shared_act_quant = getattr(self.conv3_5, "0").conv2.act.act_quant

        current_feats *= 1 #128
        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            QuantRepSparseConv2dDownsampleX2(current_feats, current_feats, indice_key="spconv4", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res4", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res4", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res4", act_quant2=shared_act_quant, return_quant_tensor2=False, **block_shared_kwargs),
        )

        current_feats *= 1 #128
        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            QuantRepSparseConv2dDownsampleX2(current_feats, current_feats, indice_key="spconv5", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res5", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res5", **block_shared_kwargs),
            QuantSparseBasicBlock(current_feats, current_feats, indice_key="res5", act_quant2=shared_act_quant, return_quant_tensor2=False, **block_shared_kwargs),
        )

        self.conv_out = QuantRepSparseConv2d(
            current_feats, 
            current_feats,
            indice_key = "conv_out",
            input_quant = input_quant,
            weight_quant = weight_quant,
            act_quant = act_quant,
            return_quant_tensor = return_quant_tensor,
        )

        self.shared_conv = QuantRepSubMConv2d(
            current_feats, 
            current_feats,
            indice_key = "shared_conv",
            input_quant = input_quant,
            weight_quant = weight_quant,
            act_quant = act_quant,
            return_quant_tensor = return_quant_tensor,
        )
        
        self.num_point_features = current_feats

    def bev_out(self, x_conv):
        features_cat_quant = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        if isinstance(x_conv, QuantTensor):
            features_cat = features_cat_quant.value
        else:
            features_cat = features_cat_quant
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        features_unique_quant = self.bev_out_quantise(features_unique)

        SparseTensorCls = QuantSparseConvTensor if isinstance(x_conv, QuantTensor) else SparseConvTensor

        x_out = SparseTensorCls(
            features = features_unique_quant,
            indices = indices_unique,
            spatial_shape = x_conv.spatial_shape,
            batch_size = x_conv.batch_size
        )
        return x_out

    @staticmethod
    def make_sparse_tensor_copy(x):
        return SparseConvTensor(
            x.features + 0,
            x.indices + 0,
            x.spatial_shape,
            x.batch_size
        )
    
    def fuse_multiscale_features(self, x_conv_x1, x_conv_x2, x_conv_x4, subm_x1=True):

        num_voxels_x1 = x_conv_x1.features.shape[0]

        x1s = [x_conv_x1]

        x2s = []
        for off_x in range(2):
            for off_y in range(2):
                x = self.make_sparse_tensor_copy(x_conv_x2)
                x.indices[:, 1:] *= 2
                x.indices[:, 1] += off_x
                x.indices[:, 2] += off_y
                x2s.append(x)

        x4s = []
        for off_x in range(4):
            for off_y in range(4):
                x = self.make_sparse_tensor_copy(x_conv_x4)
                x.indices[:, 1:] *= 4
                x.indices[:, 1] += off_x
                x.indices[:, 2] += off_y
                x4s.append(x)

        xs = x1s + x2s + x4s
        features_cat = torch.cat([ x.features for x in xs ])
        indices_cat = torch.cat([ x.indices for x in xs ])

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique: torch.Tensor = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        if subm_x1:
            idx_unique_x1 = _inv[:num_voxels_x1]
            indices_unique_x1 = indices_unique[idx_unique_x1, :]
            features_unique_x1 = features_unique[idx_unique_x1, :]
        else:
            indices_unique_x1 = indices_unique
            features_unique_x1 = features_unique

        features_unique_x1_quant = self.bev_out_quantise(features_unique_x1)

        SparseTensorCls = QuantSparseConvTensor if isinstance(x_conv_x1, QuantTensor) else SparseConvTensor

        x_out = SparseTensorCls(
            features = features_unique_x1_quant,
            indices = indices_unique_x1,
            spatial_shape = x_conv_x1.spatial_shape,
            batch_size = x_conv_x1.batch_size
        )
        return x_out

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict["pillar_features"], batch_dict["pillar_coords"]
        batch_size = batch_dict["batch_size"]

        quant_pillar_features = self.quantize_input_feats(pillar_features)

        input_sp_tensor = SparseConvTensor(
            features = quant_pillar_features,
            indices = pillar_coords.int(),
            spatial_shape = self.sparse_shape,
            batch_size = batch_size
        )
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = x_conv1
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.conv3_5(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)

        out = self.fuse_multiscale_features(x_conv3, x_conv4, x_conv5, subm_x1=True)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        batch_dict.update({
            "encoded_spconv_tensor": out,
            "encoded_spconv_tensor_stride": 4
        })
        
        return batch_dict
