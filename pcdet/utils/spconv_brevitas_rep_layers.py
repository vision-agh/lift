from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import numpy as np

import spconv.pytorch as spconv

from brevitas.nn import (
    QuantIdentity, 
    QuantReLU
)

from .spconv_brevitas import (
    QuantSparseConv2d,
    QuantSubMConv2d,
    replace_feature,
)


class QuantRepSubMConv2d(spconv.SparseModule):
    NORM_CLS = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

    def __init__(self, in_channels: int, out_channels: int, indice_key: str, input_quant, weight_quant, act_quant, return_quant_tensor=False) -> None:
        super(QuantRepSubMConv2d, self).__init__()

        bias = False
        stride = 1

        self._has_identity_branch = (stride == 1) and (in_channels == out_channels)

        self.cnv3x3 = QuantSubMConv2d(
            in_channels, 
            out_channels, 
            3, 
            stride = stride, 
            padding = 1, 
            bias = bias, 
            indice_key = indice_key,

            weight_quant = weight_quant,
            input_quant = input_quant,
            output_quant = None,

            return_quant_tensor = False,
        )
        self.bn3x3 = self.NORM_CLS(out_channels)

        self.cnv1x1 = QuantSubMConv2d(
            in_channels, 
            out_channels, 
            1, 
            stride = stride, 
            bias = bias, 
            indice_key = indice_key,

            weight_quant = self.cnv3x3.weight_quant,
            input_quant = self.cnv3x3.input_quant,
            output_quant = self.cnv3x3.output_quant,

            return_quant_tensor = False,
        )
        self.bn1x1 = self.NORM_CLS(out_channels)

        self.identity = QuantIdentity(
            act_quant = self.cnv3x3.input_quant,
            return_quant_tensor = False,
        )

        self.act = QuantReLU(
            act_quant = act_quant,
            return_quant_tensor = return_quant_tensor,
        )

        if self._has_identity_branch:
            self.bn_identity = self.NORM_CLS(out_channels)
        else:
            self.bn_identity = None
    
    def forward(self, x):
        y_3x3 = self.cnv3x3(x)
        y_3x3 = replace_feature(y_3x3, self.bn3x3(y_3x3.features))

        y_1x1 = self.cnv1x1(x)
        y_1x1 = replace_feature(y_1x1, self.bn1x1(y_1x1.features))

        branches = [ y_3x3, y_1x1 ]

        if self._has_identity_branch:
            y_identity = replace_feature(x, self.identity(x.features))
            y_identity = replace_feature(y_identity, self.bn_identity(y_identity.features))
            branches.append(y_identity)

        feats_sum = 0
        for t in branches:
            feats_sum = feats_sum + t.features

        out = replace_feature(y_3x3, feats_sum)
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


class QuantRepSparseConv2d(spconv.SparseModule):
    NORM_CLS = nn.BatchNorm1d

    def __init__(self, in_channels: int, out_channels: int, indice_key: str, input_quant, weight_quant, act_quant, return_quant_tensor=False) -> None:
        super().__init__()

        bias = False
        stride = 1

        self._has_identity_branch = (stride == 1) and (in_channels == out_channels)

        self.cnv3x3 = QuantSparseConv2d(
            in_channels, 
            out_channels, 
            3, 
            stride = stride, 
            padding = 1, 
            bias = bias, 
            indice_key = f"{indice_key}_dilated",

            weight_quant = weight_quant,
            input_quant = input_quant,
            output_quant = None,

            return_quant_tensor = False,
        )
        self.bn3x3 = self.NORM_CLS(out_channels)

        self.cnv1x1 = QuantSubMConv2d(
            in_channels, 
            out_channels, 
            1, 
            stride = stride, 
            bias = bias, 
            indice_key = indice_key,

            weight_quant = self.cnv3x3.weight_quant,
            input_quant = self.cnv3x3.input_quant,
            output_quant = self.cnv3x3.output_quant,

            return_quant_tensor = False,
        )
        self.bn1x1 = self.NORM_CLS(out_channels)



        self.dummy_dilation_cnv = DummyIdentityCnv(
            out_channels,
            indice_key = f"{indice_key}_dilated"
        )

        self.act = QuantReLU(
            act_quant = act_quant,
        )

        if self._has_identity_branch:
            self.bn_identity = self.NORM_CLS(out_channels)
            self.identity = QuantIdentity(
                act_quant = self.cnv3x3.input_quant,
                return_quant_tensor = False,
            )
        else:
            self.bn_identity = None
            self.identity = None
    
    def forward(self, x):
        y_3x3 = self.cnv3x3(x)
        y_3x3 = replace_feature(y_3x3, self.bn3x3(y_3x3.features))

        x_dilated = self.dummy_dilation_cnv(x)

        y_1x1 = self.cnv1x1(x_dilated)
        y_1x1 = replace_feature(y_1x1, self.bn1x1(y_1x1.features))

        subm_branch = y_1x1

        if self._has_identity_branch:
            y_identity = replace_feature(x_dilated, self.identity(x_dilated.features))
            y_identity = replace_feature(y_identity, self.bn_identity(y_identity.features))

            subm_branch = replace_feature(subm_branch, y_identity.features + subm_branch.features)

        out = replace_feature(y_3x3, y_3x3.features + subm_branch.features)
        out = replace_feature(out, self.act(out.features))

        return out


class QuantGaussianDownsampleX2(spconv.SparseModule):
    def __init__(self, in_channels: int, out_channels: int, indice_key: str, weight_quant, input_quant, output_quant, return_quant_tensor=False):
        super().__init__()

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


class QuantRepSparseConv2dDownsampleX2(spconv.SparseModule):
    NORM_CLS = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

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
        self.skip_cnv = QuantGaussianDownsampleX2(
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

        out = replace_feature(y_main, y_main.features + y_1x1.features)
        out = replace_feature(out, self.act(out.features))

        return out
    