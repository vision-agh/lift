from typing import List, Union, Optional, Tuple

import torch
from torch import Tensor

from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_layer import (
    ActQuantType,
    BiasQuantType,
    WeightQuantType,
)
from brevitas.nn.mixin.base import _CachedIO
from brevitas.nn.utils import compute_channel_view_shape
from brevitas.quant_tensor import QuantTensor

from cumm import tensorview as tv
from spconv.core import ConvAlgo
from spconv.pytorch.core import SparseConvTensor
from spconv.pytorch.conv import SparseConvolution

def quant_scatter_nd(indices: Tensor, features: QuantTensor, shape: List[int]) -> QuantTensor:
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=features.value.dtype, device=features.value.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = features.value.view(*output_shape)

    if len(features.scale.shape) > 1:
        ret_scale = features.scale.reshape(
            features.scale.shape[0],
            1, 
            1,
            features.scale.shape[1],
        )
    else:
        ret_scale = features.scale

    qret = QuantTensor(
        ret,
        ret_scale,
        features.zero_point,
        features.bit_width,
        features.signed_t,
        features.training_t,
    )
    return qret

class QuantSparseConvTensor(SparseConvTensor):
    def __init__(self, features: QuantTensor, *args, **kwargs):
        if not hasattr(features, "ndim"):
            features.ndim = len(features.shape)
        super().__init__(features, *args, **kwargs)

    @classmethod
    def from_dense(cls, x: QuantTensor):
        raise NotImplementedError
    
    def dense(self, channels_first: bool = True) -> QuantTensor:
        output_shape = [self.batch_size] + list(
            self.spatial_shape) + [self.features.shape[1]]
        res = quant_scatter_nd(
            self.indices.to(self.features.device).long(), self.features,
            output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()
    
    def shadow_copy(self) -> "QuantSparseConvTensor":
        """create a new spconv tensor with all member unchanged"""
        tensor = QuantSparseConvTensor(self.features, self.indices,
                                  self.spatial_shape, self.batch_size,
                                  self.grid, self.voxel_num, self.indice_dict,
                                  self.benchmark)
        tensor.benchmark_record = self.benchmark_record
        tensor.thrust_allocator = self.thrust_allocator
        tensor._timer = self._timer
        tensor.force_algo = self.force_algo
        tensor.int8_scale = self.int8_scale

        if hasattr(self, "flops"):
            tensor.flops = self.flops
        return tensor
    
    def replace_feature(self, feature: QuantTensor) -> "QuantSparseConvTensor":
        """we need to replace x.features = F.relu(x.features) with x = x.replace_feature(F.relu(x.features))
        due to limit of torch.fx
        """
        # assert feature.shape[0] == self.indices.shape[0], "replaced num of features not equal to indices"
        new_spt = QuantSparseConvTensor(feature, self.indices, self.spatial_shape,
                                   self.batch_size, self.grid, self.voxel_num,
                                   self.indice_dict)
        new_spt.benchmark = self.benchmark
        new_spt.benchmark_record = self.benchmark_record
        new_spt.thrust_allocator = self.thrust_allocator
        new_spt._timer = self._timer
        new_spt.force_algo = self.force_algo
        new_spt.int8_scale = self.int8_scale

        if hasattr(self, "flops"):
            new_spt.flops = self.flops

        return new_spt

def replace_feature_quant2fp(inp: QuantSparseConvTensor, feature: Tensor) -> "SparseConvTensor":
    """we need to replace x.features = F.relu(x.features) with x = x.replace_feature(F.relu(x.features))
    due to limit of torch.fx
    """
    # assert feature.shape[0] == inp.indices.shape[0], "replaced num of features not equal to indices"
    new_spt = SparseConvTensor(feature, inp.indices, inp.spatial_shape,
                                inp.batch_size, inp.grid, inp.voxel_num,
                                inp.indice_dict)
    new_spt.benchmark = inp.benchmark
    new_spt.benchmark_record = inp.benchmark_record
    new_spt.thrust_allocator = inp.thrust_allocator
    new_spt._timer = inp._timer
    new_spt.force_algo = inp.force_algo
    new_spt.int8_scale = inp.int8_scale

    if hasattr(inp, "flops"):
        new_spt.flops = inp.flops

    return new_spt

def replace_feature_fp2quant(inp: SparseConvTensor, feature: QuantTensor) -> "QuantSparseConvTensor":
    """we need to replace x.features = F.relu(x.features) with x = x.replace_feature(F.relu(x.features))
    due to limit of torch.fx
    """
    # assert feature.shape[0] == inp.indices.shape[0], "replaced num of features not equal to indices"
    new_spt = QuantSparseConvTensor(feature, inp.indices, inp.spatial_shape,
                                inp.batch_size, inp.grid, inp.voxel_num,
                                inp.indice_dict)
    new_spt.benchmark = inp.benchmark
    new_spt.benchmark_record = inp.benchmark_record
    new_spt.thrust_allocator = inp.thrust_allocator
    new_spt._timer = inp._timer
    new_spt.force_algo = inp.force_algo
    new_spt.int8_scale = inp.int8_scale

    if hasattr(inp, "flops"):
        new_spt.flops = inp.flops

    return new_spt
    
def replace_feature(
        sparse_tensor: Union[SparseConvTensor, QuantSparseConvTensor],
        feature: Union[Tensor, QuantTensor]
        ):
    if isinstance(sparse_tensor, QuantSparseConvTensor) and not isinstance(feature, QuantTensor):
        ret = replace_feature_quant2fp(sparse_tensor, feature)
    elif not isinstance(sparse_tensor, QuantSparseConvTensor) and isinstance(feature, QuantTensor):
        ret = replace_feature_fp2quant(sparse_tensor, feature)
    else:
        ret = sparse_tensor.replace_feature(feature)
    return ret

class QuantSparseConvolution(QuantWBIOL, SparseConvolution):
    def __init__(self,
                 ndim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 subm: bool = False,
                 output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 transposed: bool = False,
                 inverse: bool = False,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None,
                 fp32_accum: Optional[bool] = None,
                 record_voxel_count: bool = False,
                 act_type: tv.gemm.Activation = tv.gemm.Activation.None_,
                 act_alpha: float = 0,
                 act_beta: float = 0,
                 large_kernel_fast_algo: bool = False,
                 name=None,
                 weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
                 bias_quant: Optional[BiasQuantType] = None,
                 input_quant: Optional[ActQuantType] = None,
                 output_quant: Optional[ActQuantType] = None,
                 return_quant_tensor: bool = False,
                 device=None,
                 dtype=None,
                 **kwargs):
        SparseConvolution.__init__(
            self,
            ndim = ndim,
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            subm = subm,
            output_padding = output_padding,
            transposed = transposed,
            inverse = inverse,
            indice_key = indice_key,
            algo = algo,
            fp32_accum = fp32_accum,
            record_voxel_count = record_voxel_count,
            act_type = act_type,
            act_alpha = act_alpha,
            act_beta = act_beta,
            large_kernel_fast_algo = large_kernel_fast_algo,
            name = name,
            device = device,
            dtype = dtype)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
    
    def forward(self, input: QuantSparseConvTensor):
        return self.forward_impl(input)

    def inner_forward_impl(self,
                           inp: SparseConvTensor,
                           weight: Tensor,
                           bias: Tensor) -> SparseConvTensor:
        add_input = None
        ret = self._conv_forward(self.training,
                                  inp,
                                  weight,
                                  bias,
                                  add_input,
                                  name = self.name,
                                  sparse_unique_name = self._sparse_unique_name,
                                  act_type = self.act_type,
                                  act_alpha = self.act_alpha,
                                  act_beta = self.act_beta)
        return ret
    
    @property
    def output_channel_dim(self):
        return 0

    def forward_impl(self, sparse_inp: QuantSparseConvTensor) -> QuantSparseConvTensor:
        output_scale = None
        output_bit_width = None
        output_zero_point = None
        output_signed = None

        inp = sparse_inp.features
        inp = self.unpack_input(inp)

        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(inp.value)
            self._set_global_is_quant_layer(False)
            return out

        quant_input = self.input_quant(inp)
        quant_weight = self.quant_weight(quant_input)

        if (self.return_quant_tensor or
            (self.is_bias_quant_enabled and
             (self.bias_quant.requires_input_scale or self.bias_quant.requires_input_bit_width))):
            if quant_input.bit_width is not None and quant_weight.bit_width is not None:
                output_bit_width = self.max_acc_bit_width(
                    quant_input.bit_width, quant_weight.bit_width)
            if quant_input.scale is not None and quant_weight.scale is not None:
                output_scale = self.quant_output_scale_impl(
                    inp, quant_input.scale, quant_weight.scale)
            if quant_input.signed is not None:
                output_signed = inp.signed or quant_weight.signed

        sparse_quant_input = replace_feature_quant2fp(sparse_inp, quant_input.value)
        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, output_scale, output_bit_width)
            if not self.training and self.cache_inference_quant_bias:
                self._cached_bias = _CachedIO(quant_bias.detach(), metadata_only=False)


            sparse_output_tensor = self.inner_forward_impl(
                sparse_quant_input, quant_weight.value, quant_bias.value)

            if (self.return_quant_tensor and output_scale is not None and
                (quant_bias.scale is None or
                 (quant_bias.scale is not None and
                  quant_bias.scale.data_ptr() != output_scale.data_ptr()))):
                output_scale_broadcast_shape = compute_channel_view_shape(inp, channel_dim=1)
                output_zero_point = -quant_bias.value.view(
                    output_scale_broadcast_shape) / output_scale

            if quant_bias.bit_width is not None and output_bit_width is not None:
                output_bit_width = torch.where(
                    quant_bias.bit_width > output_bit_width, quant_bias.bit_width, output_bit_width)
                output_bit_width = output_bit_width + 1
        else:
            sparse_output_tensor = self.inner_forward_impl(sparse_quant_input, quant_weight.value, None)
        output_tensor = sparse_output_tensor.features

        if self.return_quant_tensor and not self.is_output_quant_enabled:
            if (quant_input.zero_point is not None and quant_weight.zero_point is not None and
                ((quant_input.zero_point != 0.0).any() or (quant_weight.zero_point != 0.0).any())):
                raise RuntimeError("Computing zero point of output accumulator not supported yet.")
            elif quant_input.zero_point is not None and output_zero_point is None:
                output_zero_point = quant_input.zero_point

        quant_output = QuantTensor(
            value=output_tensor,
            scale=output_scale,
            zero_point=output_zero_point,
            bit_width=output_bit_width,
            signed=output_signed,
            training=self.training)
        quant_output = self.output_quant(quant_output)
        output = self.pack_output(quant_output)

        sparse_output = replace_feature_fp2quant(sparse_output_tensor, output)
        return sparse_output

class QuantSparseConv2d(QuantSparseConvolution):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                indice_key=None,
                algo: Optional[ConvAlgo] = None,
                fp32_accum: Optional[bool] = None,
                record_voxel_count: bool = False,
                large_kernel_fast_algo: bool = False,
                name=None,
                weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
                bias_quant: Optional[BiasQuantType] = None,
                input_quant: Optional[ActQuantType] = None,
                output_quant: Optional[ActQuantType] = None,
                return_quant_tensor: bool = False):
        ndim = 2
        super(QuantSparseConv2d, self).__init__(
            ndim,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
            weight_quant = weight_quant,
            bias_quant = bias_quant,
            input_quant = input_quant,
            output_quant = output_quant,
            return_quant_tensor = return_quant_tensor)

class QuantSubMConv2d(QuantSparseConvolution):
     def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                indice_key=None,
                algo: Optional[ConvAlgo] = None,
                fp32_accum: Optional[bool] = None,
                large_kernel_fast_algo: bool = False,
                name = None,
                weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
                bias_quant: Optional[BiasQuantType] = None,
                input_quant: Optional[ActQuantType] = None,
                output_quant: Optional[ActQuantType] = None,
                return_quant_tensor: bool = False):
        ndim = 2
        subm = True
        super(QuantSubMConv2d, self).__init__(
            ndim,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            subm,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
            weight_quant = weight_quant,
            bias_quant = bias_quant,
            input_quant = input_quant,
            output_quant = output_quant,
            return_quant_tensor = return_quant_tensor)