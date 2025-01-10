from typing import (
    Union,
    Optional
)
from functools import partial

from spconv.core import ConvAlgo

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from brevitas.nn import (
    QuantIdentity, 
    QuantReLU
)
from brevitas.quant_tensor import QuantTensor
from brevitas.nn.quant_layer import (
    ActQuantType,
    QuantNonLinearActLayer as QuantNLAL,
)
from brevitas.quant import (
    Uint8ActPerTensorFloat,
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
)
from spconv.pytorch.core import SparseConvTensor
import spconv.pytorch as spconv

from .spconv_brevitas import (
    QuantSparseConvTensor,
    QuantSparseConv2d,
    QuantSubMConv2d,
    replace_feature,
)
from .spconv_brevitas_rep_layers import (
    QuantRepSubMConv2d
)

class QuantLeakyReLU(QuantNLAL):
    def __init__(
            self,
            negative_slope: float = 1 / 8,
            act_quant: Optional[ActQuantType] = Int8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=partial(nn.LeakyReLU, negative_slope=negative_slope),
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)

class _QuantGaussian2xDownsampling3x3_Skip_V2(spconv.SparseModule):
    def __init__(self, in_channels: int, out_channels: int, indice_key: str, weight_quant, input_quant, output_quant, return_quant_tensor=False):
        super().__init__()

        # Constants
        # Antialiasing filter with sigma = 0.5
        # Note: sigma value taken from the scikit-image anti-aliasing resizing equation (s - 1) / 2
        filter_values = np.array(
            [
                [0.01134374, 0.08381951, 0.01134374],
                [0.08381951, 0.61934704, 0.08381951],
                [0.01134374, 0.08381951, 0.01134374],
            ],
            dtype = np.float32,
        )
        filter_values /= filter_values.sum()
        kw = 3
        kh = 3

        assert out_channels % in_channels == 0
        exp_ratio = out_channels // in_channels

        cin = in_channels
        cout = out_channels
        w = np.zeros((cout, cin, kw, kh), dtype = np.float32)
        for i in range(kw):
            for j in range(kh):
                for block_idx in range(exp_ratio):
                    w[cin * block_idx: cin * (block_idx + 1), :, i, j] = np.eye(cin) * filter_values[i, j]
        w = np.transpose(w, (0, 2, 3, 1)) # [Cout, Cin, Kw, Kh] --> [Cout, Kw, Kh, Cin]
        w = np.ascontiguousarray(w)
        w = torch.from_numpy(w)
        w = torch.nn.Parameter(w, requires_grad = False)

        self.dummy_cnv = QuantSparseConv2d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            bias = False,
            indice_key = indice_key,

            weight_quant = weight_quant,
            input_quant = input_quant,
            output_quant = output_quant,

            return_quant_tensor = return_quant_tensor,
        )
        self.dummy_cnv.weight = w

    def forward(self, x):
        out = self.dummy_cnv(x)
        return out

class _QuantDownsampling2x2_Skip(spconv.SparseModule):
    def __init__(self, in_channels: int, out_channels: int, indice_key: str, weight_quant, input_quant, output_quant, return_quant_tensor=False):
        super().__init__()

        # Constants
        filter_values = np.array(
            [
                [0.25, 0.25],
                [0.25, 0.25],
            ],
            dtype = np.float32,
        )
        filter_values /= filter_values.sum()
        kw = 2
        kh = 2

        assert out_channels % in_channels == 0
        exp_ratio = out_channels // in_channels

        cin = in_channels
        cout = out_channels
        w = np.zeros((cout, cin, kw, kh), dtype = np.float32)
        for i in range(kw):
            for j in range(kh):
                for block_idx in range(exp_ratio):
                    w[cin * block_idx: cin * (block_idx + 1), :, i, j] = np.eye(cin) * filter_values[i, j]
        w = np.transpose(w, (0, 2, 3, 1)) # [Cout, Cin, Kw, Kh] --> [Cout, Kw, Kh, Cin]
        w = np.ascontiguousarray(w)
        w = torch.from_numpy(w)
        w = torch.nn.Parameter(w, requires_grad = False)

        if 0:
            self.dummy_cnv = QuantSparseConv2d(
                in_channels,
                out_channels,
                kernel_size = 2,
                stride = 2,
                padding = 0,
                bias = False,
                indice_key = indice_key,

                weight_quant = weight_quant,
                input_quant = input_quant,
                output_quant = output_quant,

                return_quant_tensor = return_quant_tensor,
            )
        else:
            self.dummy_cnv = spconv.SparseConv2d(
                in_channels,
                out_channels,
                kernel_size = 2,
                stride = 2,
                padding = 0,
                bias = False,
                indice_key = indice_key,
            )
        self.dummy_cnv.weight = w

    def forward(self, x):
        out = self.dummy_cnv(x)
        return out

class PreDownsamplingAntialiasingFilter(spconv.SparseModule):
    def __init__(self, in_channels: int, out_channels: int, indice_key: str, weight_quant, input_quant, output_quant, act_quant, return_quant_tensor=False):
        super().__init__()

        # Constants
        filter_values = np.array(
            [
                [0.01134374, 0.08381951, 0.01134374],
                [0.08381951, 0.61934704, 0.08381951],
                [0.01134374, 0.08381951, 0.01134374],
            ],
            dtype = np.float32,
        )
        filter_values /= filter_values.sum()
        kw = 3
        kh = 3

        assert out_channels == in_channels

        cin = in_channels
        cout = out_channels
        w = np.zeros((cout, 1, kw, kh), dtype = np.float32)
        for i in range(kw):
            for j in range(kh):
                w[:, :, i, j] = filter_values[i, j]
        w = np.transpose(w, (0, 2, 3, 1)) # [Cout, Cin, Kw, Kh] --> [Cout, Kw, Kh, Cin]
        w = np.ascontiguousarray(w)
        w = torch.from_numpy(w)
        w = torch.nn.Parameter(w, requires_grad = False)

        self.dummy_cnv = QuantSparseConv2d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False,
            indice_key = indice_key,
            groups = in_channels,
            algo = ConvAlgo.Native,

            weight_quant = weight_quant,
            input_quant = input_quant,
            output_quant = output_quant,

            return_quant_tensor = return_quant_tensor,
        )
        self.dummy_cnv.weight = w

        self.act = QuantIdentity(act_quant=act_quant)

    def forward(self, x):
        out = self.dummy_cnv(x)
        out = replace_feature(out, self.act(out.features))
        return out

class QuantGaussian2xDownsampling3x3_V2(spconv.SparseModule):
    NORM_CLS = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
    # NORM_CLS = nn.BatchNorm1d

    def __init__(self, in_channels: int, out_channels: int, indice_key: str, weight_quant, input_quant, act_quant, return_quant_tensor=False):
        super().__init__()

        self.main_cnv = QuantSparseConv2d(
            in_channels, 
            out_channels, 
            kernel_size = 3, 
            stride = 2, 
            padding = 1, 
            bias = False,
            indice_key = indice_key,

            weight_quant = weight_quant,
            input_quant = input_quant,
            output_quant = None,

            return_quant_tensor = False,
        )
        self.skip_cnv = _QuantGaussian2xDownsampling3x3_Skip_V2(
            in_channels, 
            in_channels, 
            indice_key,

            weight_quant = self.main_cnv.weight_quant,
            input_quant = self.main_cnv.input_quant,
            output_quant = self.main_cnv.output_quant,

            return_quant_tensor = False,
        )

        self.cnv1x1 = QuantSubMConv2d(
            in_channels, 
            out_channels, 
            1, 
            stride = 1, 
            bias = False, 
            indice_key = indice_key,

            weight_quant = self.main_cnv.weight_quant,
            input_quant = self.main_cnv.input_quant,
            output_quant = self.main_cnv.output_quant,

            return_quant_tensor = False,
        )


        # self.skip_bn = self.NORM_CLS(out_channels)
        self.main_bn = self.NORM_CLS(out_channels)
        self.cnv_1x1_bn = self.NORM_CLS(out_channels)

        self.act = QuantReLU(
            act_quant = act_quant, 
            return_quant_tensor = return_quant_tensor
        )

    def forward(self, x):
        y_main = self.main_cnv(x)
        y_main = replace_feature(y_main, self.main_bn(y_main.features))

        y_skip = self.skip_cnv(x)

        y_1x1 = self.cnv1x1(y_skip)
        y_1x1 = replace_feature(y_1x1, self.cnv_1x1_bn(y_1x1.features))

        # y_skip = replace_feature(y_skip, self.skip_bn(y_skip.features))


        # out = replace_feature(y_main, y_main.features + y_skip.features + y_1x1.features)
        out = replace_feature(y_main, y_main.features + y_1x1.features)
        out = replace_feature(out, self.act(out.features))

        return out
    

class Downsampling_V3(spconv.SparseModule):
    NORM_CLS = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
    # NORM_CLS = nn.BatchNorm1d

    def __init__(self, in_channels: int, out_channels: int, indice_key: str, weight_quant, input_quant, act_quant, return_quant_tensor=False):
        super().__init__()

        self._use_expand_identity_branch = False

        self.antialias_filter = PreDownsamplingAntialiasingFilter(
            in_channels, 
            in_channels,
            indice_key=f"{indice_key}_pre_filter",
            weight_quant = weight_quant,
            act_quant = Int8ActPerTensorFloat,
            
            input_quant=None,
            output_quant=None,
            return_quant_tensor=False
        )

        self.main_cnv = QuantSparseConv2d(
            in_channels, 
            out_channels, 
            kernel_size = 2, 
            stride = 2, 
            padding = 0, 
            bias = False,
            indice_key = indice_key,

            weight_quant = weight_quant,
            input_quant = input_quant,
            output_quant = None,

            return_quant_tensor = False,
        )
        if self._use_expand_identity_branch:
            self.skip_cnv_expand = _QuantDownsampling2x2_Skip(
                in_channels,
                out_channels, 
                indice_key,

                weight_quant = self.main_cnv.weight_quant,
                input_quant = self.main_cnv.input_quant,
                output_quant = self.main_cnv.output_quant,

                return_quant_tensor = False,
            )
        self.skip_cnv_precnv1x1 = _QuantDownsampling2x2_Skip(
            in_channels,
            in_channels,
            indice_key,

            weight_quant = self.main_cnv.weight_quant,
            input_quant = self.main_cnv.input_quant,
            output_quant = self.main_cnv.output_quant,

            return_quant_tensor = False,
        )

        self.cnv1x1 = QuantSubMConv2d(
            in_channels, 
            out_channels, 
            1, 
            stride = 1, 
            bias = False, 
            indice_key = indice_key,

            weight_quant = self.main_cnv.weight_quant,
            input_quant = self.main_cnv.input_quant,
            output_quant = self.main_cnv.output_quant,

            return_quant_tensor = False,
        )


        if self._use_expand_identity_branch:
            self.skip_bn = self.NORM_CLS(out_channels)
        
        self.main_bn = self.NORM_CLS(out_channels)
        self.cnv_1x1_bn = self.NORM_CLS(out_channels)

        self.act = QuantReLU(
            act_quant = act_quant, 
            return_quant_tensor = return_quant_tensor
        )

    def forward(self, x):
        x = self.antialias_filter(x)

        y_main = self.main_cnv(x)
        y_main = replace_feature(y_main, self.main_bn(y_main.features))

        y_skip_pre1x1 = self.skip_cnv_precnv1x1(x)
        y_1x1 = self.cnv1x1(y_skip_pre1x1)
        y_1x1 = replace_feature(y_1x1, self.cnv_1x1_bn(y_1x1.features))

        if self._use_expand_identity_branch:
            y_skip = self.skip_cnv_expand(x)
            y_skip = replace_feature(y_skip, self.skip_bn(y_skip.features))
            y_side = replace_feature(y_skip, y_skip.features + y_1x1.features)
        else:
            y_side = y_1x1

        out = replace_feature(y_main, y_main.features + y_side.features)
        out = replace_feature(out, self.act(out.features))

        return out

class DummyIdentityCnv(spconv.SparseModule):
    def __init__(self, channels: int, indice_key: str):
        super().__init__()

        # Constants
        filter_values = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype = np.float32,
        )
        filter_values /= filter_values.sum()
        kw = 3
        kh = 3

        in_channels = channels
        out_channels = channels

        assert out_channels % in_channels == 0
        exp_ratio = out_channels // in_channels

        cin = in_channels
        cout = out_channels
        w = np.zeros((cout, cin, kw, kh), dtype = np.float32)
        for i in range(kw):
            for j in range(kh):
                for block_idx in range(exp_ratio):
                    w[cin * block_idx: cin * (block_idx + 1), :, i, j] = np.eye(cin) * filter_values[i, j]
        w = np.transpose(w, (0, 2, 3, 1)) # [Cout, Cin, Kw, Kh] --> [Cout, Kw, Kh, Cin]
        w = np.ascontiguousarray(w)
        w = torch.from_numpy(w)
        w = torch.nn.Parameter(w, requires_grad = False)

        self.dummy_cnv = QuantSparseConv2d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False,
            indice_key = indice_key,

            weight_quant = None,
            input_quant = None,
            output_quant = None,
            return_quant_tensor = None,
        )
        self.dummy_cnv.weight = w

    def forward(self, x):
        out = self.dummy_cnv(x)
        return out
