from typing import Tuple, List, Union
from collections import OrderedDict
import numpy as np
import torch
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.conv import SparseConvolution
# from det3d.models.necks.aspp import ASPPNeck

def _count_mac_ops_conv_dense(tensor_w, tensor_h, kh, kw, cin, cout, stride_x, stride_y, padding_same):
    if padding_same:
        pad_left = kw // 2
        pad_right = kw - pad_left - 1
        pad_up = kh // 2
        pad_down = kh - pad_up - 1
        tensor_w = tensor_w + pad_left + pad_right
        tensor_h = tensor_h + pad_up + pad_down

    wout = (tensor_w - kw) // stride_x + 1
    hout = (tensor_h - kh) // stride_y + 1

    macs = wout * hout * kw * kh * cin * cout
    return macs

def _count_mac_ops_sparse(coords, tensor_w, tensor_h, kh, kw, cin, cout, stride_x, stride_y, is_submainfold):
    # Remember to handle coords shift due to padding or no-padding.
    # If padding is not handled here (padding may have been added at earlier stage), we must also change 
    # this condition: out_coord_x < 0 or out_coord_y < 0 or out_coord_x >= tensor_w or out_coord_y >= tensor_h
    # to this: out_coord_x < pad_x or out_coord_y < pad_y or out_coord_x >= tensor_w - pad_x or out_coord_y >= tensor_h - pad_y
    # So that we don't produce at output out of range pixels.

    # Here, however we assumed "virtually" zero-padded tensor --> so that the padding coords are outside [0, tensor_w-1] X [0, tensor_h-1] range (minus coords or greater-equal to tensor_w, tensor_h).

    # We assume that in strided conv, the reference active pixel is at (0, 0) location.

    assert kw % 2 == 1 and kh % 2 == 1 # We assume here only uneven kernel size
    assert not (is_submainfold and (stride_x != 1 or stride_y != 1)) # Submainfold convs are well defined only for stride == 1

    input_coords = coords
    if is_submainfold:
        output_coords = [ (int(c[0]), int(c[1])) for c in input_coords ]
    else:
        output_coords = None

    mac_inc = cin * cout
    macs = 0

    off_x_center = kw // 2
    off_y_center = kh // 2
    for input_coord in input_coords:
        coord_x = int(input_coord[0])
        coord_y = int(input_coord[1])
        for ky_pos in range(kh):
            off_y = ky_pos - off_y_center
            for kx_pos in range(kw):
                off_x = kx_pos - off_x_center
                out_coord_x = coord_x + off_x
                out_coord_y = coord_y + off_y
                if out_coord_x < 0 or out_coord_y < 0 or out_coord_x >= tensor_w or out_coord_y >= tensor_h:
                    # Handle out of bounds output pixels
                    continue
                else:
                    if out_coord_x % stride_x == 0 and out_coord_y % stride_y == 0:
                        out_coord = (out_coord_x, out_coord_y)

                        if is_submainfold:
                            if out_coord in output_coords:
                                macs += mac_inc
                            else:
                                continue
                        else:
                            macs += mac_inc
                    else:
                        # Handle output pixels due to striding
                        continue
    return macs

def _count_mac_ops_sparse_fast(coords: np.ndarray, tensor_w, tensor_h, kh, kw, cin, cout, stride_x, stride_y, is_submainfold):
    # Remember to handle coords shift due to padding or no-padding.
    # If padding is not handled here (padding may have been added at earlier stage), we must also change 
    # this condition: out_coord_x < 0 or out_coord_y < 0 or out_coord_x >= tensor_w or out_coord_y >= tensor_h
    # to this: out_coord_x < pad_x or out_coord_y < pad_y or out_coord_x >= tensor_w - pad_x or out_coord_y >= tensor_h - pad_y
    # So that we don't produce at output out of range pixels.

    # Here, however we assumed "virtually" zero-padded tensor --> so that the padding coords are outside [0, tensor_w-1] X [0, tensor_h-1] range (minus coords or greater-equal to tensor_w, tensor_h).

    # We assume that in strided conv, the reference active pixel is at (0, 0) location.

    assert kw % 2 == 1 and kh % 2 == 1 # We assume here only uneven kernel size
    assert not (is_submainfold and (stride_x != 1 or stride_y != 1)) # Submainfold convs are well defined only for stride == 1

    input_coords = coords

    off_x_center = kw // 2
    off_y_center = kh // 2

    coords_y = input_coords[:, 0]
    coords_x = input_coords[:, 1]

    offsets_x = np.arange(kw) - off_x_center
    offsets_y = np.arange(kh) - off_y_center
    off_xx, off_yy = np.meshgrid(offsets_x, offsets_y)
    out_coords_x = coords_x[:, np.newaxis] + off_xx.reshape(1, kh * kw)
    out_coords_y = coords_y[:, np.newaxis] + off_yy.reshape(1, kh * kw)

    out_coords_x = out_coords_x.flatten()
    out_coords_y = out_coords_y.flatten()

    out_coords_yx = np.stack((out_coords_y, out_coords_x), axis=1)
    out_coords_yx_values, out_coords_yx_counts = np.unique(
        out_coords_yx,
        axis = 0,
        return_counts = True,
    )

    valid_coords = np.logical_and.reduce(
        (
            out_coords_yx_values[:, 0] >= 0,
            out_coords_yx_values[:, 0] < tensor_h,
            out_coords_yx_values[:, 1] >= 0,
            out_coords_yx_values[:, 1] < tensor_w,
            out_coords_yx_values[:, 0] % stride_y == 0,
            out_coords_yx_values[:, 1] % stride_x == 0,
        )
    )
    if is_submainfold:
        comparison = out_coords_yx_values[:, np.newaxis, :] == input_coords[np.newaxis, :, :]
        comparison = comparison.all(-1).any(1)
        valid_coords = np.logical_and(valid_coords, comparison)

    out_coords_yx_values = out_coords_yx_values[valid_coords, :]
    out_coords_yx_counts = out_coords_yx_counts[valid_coords]

    macs = np.sum(out_coords_yx_counts) * cin * cout 
    return macs

def _count_mac_ops_sparse_fast_torch(coords: torch.Tensor, tensor_w, tensor_h, kh, kw, cin, cout, stride_x, stride_y, is_submainfold):
    # Remember to handle coords shift due to padding or no-padding.
    # If padding is not handled here (padding may have been added at earlier stage), we must also change 
    # this condition: out_coord_x < 0 or out_coord_y < 0 or out_coord_x >= tensor_w or out_coord_y >= tensor_h
    # to this: out_coord_x < pad_x or out_coord_y < pad_y or out_coord_x >= tensor_w - pad_x or out_coord_y >= tensor_h - pad_y
    # So that we don't produce at output out of range pixels.

    # Here, however we assumed "virtually" zero-padded tensor --> so that the padding coords are outside [0, tensor_w-1] X [0, tensor_h-1] range (minus coords or greater-equal to tensor_w, tensor_h).

    # We assume that in strided conv, the reference active pixel is at (0, 0) location.

    # assert kw % 2 == 1 and kh % 2 == 1 # We assume here only uneven kernel size
    assert not (is_submainfold and (stride_x != 1 or stride_y != 1)) # Submainfold convs are well defined only for stride == 1

    input_coords = coords

    off_x_center = kw // 2
    off_y_center = kh // 2

    device = input_coords.device
    dtype =  input_coords.dtype

    coords_y = input_coords[:, 0]
    coords_x = input_coords[:, 1]

    offsets_x = torch.arange(kw, device=device, dtype=dtype) - off_x_center
    offsets_y = torch.arange(kh, device=device, dtype=dtype) - off_y_center
    off_xx, off_yy = torch.meshgrid(offsets_x, offsets_y)
    out_coords_x = coords_x[:, None] + off_xx.reshape(1, kh * kw)
    out_coords_y = coords_y[:, None] + off_yy.reshape(1, kh * kw)

    out_coords_x = out_coords_x.flatten()
    out_coords_y = out_coords_y.flatten()


    out_coords_yx = torch.stack((out_coords_y, out_coords_x), dim=1)


    out_coords_yx_values, out_coords_yx_counts = torch.unique(
        out_coords_yx,
        dim = 0,
        return_counts = True,
    )

    # Uncomment in case of problems with CUDA out of memory
    # del out_coords_x
    # del out_coords_y
    # del out_coords_yx
    # del off_xx
    # del off_yy
    # del offsets_x
    # del offsets_y
    # torch.cuda.empty_cache()

    valid_coords = torch.all(
        torch.stack(
            (
                out_coords_yx_values[:, 0] >= 0,
                out_coords_yx_values[:, 0] < tensor_h,
                out_coords_yx_values[:, 1] >= 0,
                out_coords_yx_values[:, 1] < tensor_w,
                out_coords_yx_values[:, 0] % stride_y == 0,
                out_coords_yx_values[:, 1] % stride_x == 0,
            ),
            dim = 0,
        ),
        dim = 0,
    )
    if is_submainfold:
        # Note: solution from https://stackoverflow.com/a/67113105 tried to allocate almost 12GiB of CUDA memory, way too much!

        # Based on https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/10
        # Note: both input_coords and out_coords_yx_values have unique coordinates.
        # So, if we concat both arrays, then non-unique coordinates (with count > 1) are the intersection of the two arrays. 
        # But then, we have to project the intersection on the original concatenated array.
        # So if x_unique[inverse_indexes] == x, so we can project counts by inverse indexes.
        # Then we just slice the array and we are at home.
        num_out_coords = out_coords_yx_values.shape[0]
        coords_cat = torch.cat((out_coords_yx_values, input_coords), dim=0)
        _, inv_idxes, counts = torch.unique(coords_cat, dim=0, return_inverse=True, return_counts=True)
        counts_cat = counts[inv_idxes]
        counts_cat = counts_cat[:num_out_coords]
        valid_coords = torch.logical_and(valid_coords, counts_cat > 1)

    out_coords_yx_counts = out_coords_yx_counts[valid_coords]

    macs = torch.sum(out_coords_yx_counts) * cin * cout 
    macs = float(macs.detach().cpu())
    return macs

def _count_mac_ops_sparse_naive_fast_torch(coords: torch.Tensor, tensor_w, tensor_h, kh, kw, cin, cout, stride_x, stride_y, is_submainfold):
    # Remember to handle coords shift due to padding or no-padding.
    # If padding is not handled here (padding may have been added at earlier stage), we must also change 
    # this condition: out_coord_x < 0 or out_coord_y < 0 or out_coord_x >= tensor_w or out_coord_y >= tensor_h
    # to this: out_coord_x < pad_x or out_coord_y < pad_y or out_coord_x >= tensor_w - pad_x or out_coord_y >= tensor_h - pad_y
    # So that we don't produce at output out of range pixels.

    # Here, however we assumed "virtually" zero-padded tensor --> so that the padding coords are outside [0, tensor_w-1] X [0, tensor_h-1] range (minus coords or greater-equal to tensor_w, tensor_h).

    # We assume that in strided conv, the reference active pixel is at (0, 0) location.

    # assert kw % 2 == 1 and kh % 2 == 1 # We assume here only uneven kernel size
    assert not (is_submainfold and (stride_x != 1 or stride_y != 1)) # Submainfold convs are well defined only for stride == 1

    input_coords = coords

    off_x_center = kw // 2
    off_y_center = kh // 2

    device = input_coords.device
    dtype =  input_coords.dtype

    coords_y = input_coords[:, 0]
    coords_x = input_coords[:, 1]

    offsets_x = torch.arange(kw, device=device, dtype=dtype) - off_x_center
    offsets_y = torch.arange(kh, device=device, dtype=dtype) - off_y_center
    off_xx, off_yy = torch.meshgrid(offsets_x, offsets_y)
    out_coords_x = coords_x[:, None] + off_xx.reshape(1, kh * kw)
    out_coords_y = coords_y[:, None] + off_yy.reshape(1, kh * kw)

    out_coords_x = out_coords_x.flatten()
    out_coords_y = out_coords_y.flatten()


    out_coords_yx = torch.stack((out_coords_y, out_coords_x), dim=1)


    out_coords_yx_values, out_coords_yx_counts = torch.unique(
        out_coords_yx,
        dim = 0,
        return_counts = True,
    )

    # Uncomment in case of problems with CUDA out of memory
    # del out_coords_x
    # del out_coords_y
    # del out_coords_yx
    # del off_xx
    # del off_yy
    # del offsets_x
    # del offsets_y
    # torch.cuda.empty_cache()

    valid_coords = torch.all(
        torch.stack(
            (
                out_coords_yx_values[:, 0] >= 0,
                out_coords_yx_values[:, 0] < tensor_h,
                out_coords_yx_values[:, 1] >= 0,
                out_coords_yx_values[:, 1] < tensor_w,
                out_coords_yx_values[:, 0] % stride_y == 0,
                out_coords_yx_values[:, 1] % stride_x == 0,
            ),
            dim = 0,
        ),
        dim = 0,
    )
    if is_submainfold:
        # Note: solution from https://stackoverflow.com/a/67113105 tried to allocate almost 12GiB of CUDA memory, way too much!

        # Based on https://discuss.pytorch.org/t/intersection-between-to-vectors-tensors/50364/10
        # Note: both input_coords and out_coords_yx_values have unique coordinates.
        # So, if we concat both arrays, then non-unique coordinates (with count > 1) are the intersection of the two arrays. 
        # But then, we have to project the intersection on the original concatenated array.
        # So if x_unique[inverse_indexes] == x, so we can project counts by inverse indexes.
        # Then we just slice the array and we are at home.
        num_out_coords = out_coords_yx_values.shape[0]
        coords_cat = torch.cat((out_coords_yx_values, input_coords), dim=0)
        _, inv_idxes, counts = torch.unique(coords_cat, dim=0, return_inverse=True, return_counts=True)
        counts_cat = counts[inv_idxes]
        counts_cat = counts_cat[:num_out_coords]
        valid_coords = torch.logical_and(valid_coords, counts_cat > 1)

    out_coords_yx_counts = out_coords_yx_counts[valid_coords]

    macs = out_coords_yx_counts.shape[0] * kh * kw * cin * cout 
    macs = float(macs)
    return macs

def _format_arr_stats(arr):
    if len(arr) > 0:
        min_ = np.min(arr)
        mean_ = np.mean(arr)
        median_ = np.median(arr)
        max_ = np.max(arr)
        str_ = f'min: {min_:.2f}, mean: {mean_:.2f}, median: {median_:.2f}, max: {max_:.2f}'
    else:
        str_ = "No stats, 0 ops."
    return str_

class MacsCounterBase(object):
    def _register_measurement(self, name: str, dense_ops: float, sparse_ops: float, sparse_naive_ops: float, sparse_ops_kd: float):
        raise NotImplementedError

class LayerMacCounterBase(object):
    def __init__(self, name: str, parent: MacsCounterBase):
        self._name = name
        self._parent = parent

    def _count_dense_ops(self, module, input) -> float:
        raise NotImplementedError

    def _count_sparse_ops(self, module, input) -> float:
        raise NotImplementedError
    
    def _count_sparse_naive_ops(self, module, input) -> float:
        raise NotImplementedError
    
    def _count_sparse_sparse_ops_kd(self, module, input, output) -> float:
        raise NotImplementedError

    def __call__(self, module, input, output) -> None:
        dense_ops = self._count_dense_ops(module, input)
        sparse_ops = self._count_sparse_ops(module, input)
        sparse_naive_ops = self._count_sparse_naive_ops(module, input)
        sparse_ops_kd = self._count_sparse_sparse_ops_kd(module, input, output)
        self._parent._register_measurement(self._name, dense_ops, sparse_ops, sparse_naive_ops, sparse_ops_kd)
        return None
    
class DenseLayerMacCounterBase(LayerMacCounterBase):
    def _count_sparse_ops(self, module, input) -> float:
        return self._count_dense_ops(module, input)
    
    def _count_sparse_naive_ops(self, module, input) -> float:
        return self._count_dense_ops(module, input)
    
    def _count_sparse_sparse_ops_kd(self, module, input, output) -> float:
        return self._count_dense_ops(module, input)
    
class SparseConvMacCounter(LayerMacCounterBase):
    def _count_dense_ops(self, module: SparseConvolution, input: Tuple[SparseConvTensor]) -> float:
        x = input[0]
        cout, kh, kw, cin = module.weight_shape
        stride = module.stride
        tensor_h, tensor_w = x.spatial_shape

        stride_y, stride_x = stride
        ops = _count_mac_ops_conv_dense(
            tensor_w,
            tensor_h,
            kh,
            kw,
            cin,
            cout,
            stride_x,
            stride_y,
            padding_same=True,
        )
        return ops

    def _count_sparse_ops(self, module: SparseConvolution, input: Tuple[SparseConvTensor]) -> float:
        x = input[0]

        coords = x.indices[:, 1:] #x.indices.shape == [14336, 3]

        is_submainfold = module.subm
        cout, kh, kw, cin = module.weight_shape
        stride = module.stride
        tensor_h, tensor_w = x.spatial_shape

        stride_y, stride_x = stride

        ops = _count_mac_ops_sparse_fast_torch(
            coords,
            tensor_w,
            tensor_h,
            kh,
            kw,
            cin,
            cout,
            stride_x,
            stride_y,
            is_submainfold
        )
        return ops
    
    def _count_sparse_naive_ops(self, module: SparseConvolution, input: Tuple[SparseConvTensor]) -> float:
        x = input[0]

        coords = x.indices[:, 1:] #x.indices.shape == [14336, 3]

        is_submainfold = module.subm
        cout, kh, kw, cin = module.weight_shape
        stride = module.stride
        tensor_h, tensor_w = x.spatial_shape

        stride_y, stride_x = stride

        ops = _count_mac_ops_sparse_naive_fast_torch(
            coords,
            tensor_w,
            tensor_h,
            kh,
            kw,
            cin,
            cout,
            stride_x,
            stride_y,
            is_submainfold
        )
        return ops
    
    def _count_sparse_sparse_ops_kd(self, module, input, output) -> float:
        if hasattr(output, 'flops'):
            ret = float(output.flops)
        else:
            ret = self._count_sparse_ops(module, input)
        return ret
    
        
class DenseConvMacCounter(DenseLayerMacCounterBase):
    def _count_dense_ops(self, module: torch.nn.Conv2d, input: Tuple[torch.Tensor]) -> float:
        x = input[0]
        cout, cin, kh, kw = module.weight.shape
        _, _, tensor_h, tensor_w = x.shape
        if hasattr(module, 'stride'):
            stride = module.stride
        else:
            stride = (1, 1)
        
        cout = int(cout)
        cin = int(cin)
        kh = int(kh)
        kw = int(kw)
        tensor_h = int(tensor_h)
        tensor_w = int(tensor_w)
        stride_y, stride_x = stride
        
        ops = _count_mac_ops_conv_dense(
            tensor_w,
            tensor_h,
            kh,
            kw,
            cin,
            cout,
            stride_x,
            stride_y,
            padding_same=True,
        )
        return ops
        
# class ASPPNeckMacCounter(DenseConvMacCounter):
#     def _count_dense_ops(self, module: ASPPNeck, input: Tuple[torch.Tensor]) -> float:
#         one_conv_ops = super()._count_dense_ops(module, input)
#         aspp_ops = 4 * one_conv_ops
#         return aspp_ops

class DenseLinearMacCounter(DenseLayerMacCounterBase):
    def _count_dense_ops(self, module: torch.nn.Linear, input: Tuple[torch.Tensor]) -> float:
        x = input[0]

        num_vox = x.shape[0]
        cout, cin = module.weight.shape

        num_vox = int(num_vox)
        cout = int(cout)
        cin = int(cin)
        ops = num_vox * cin * cout
        return ops

    
class MacsCounter(MacsCounterBase):
    def __init__(self):
        self._dense_ops = OrderedDict()
        self._sparse_ops = OrderedDict()
        self._sparse_naive_ops = OrderedDict()
        self._sparse_ops_kd = OrderedDict()
        self._sparsity = OrderedDict()
        self._sparsity_naive = OrderedDict()
        self._naive_overhead = OrderedDict()
        self._total_dense_ops = []
        self._total_sparse_ops = []
        self._total_sparse_naive_ops = []
        self._total_sparse_ops_kd = []
        self._total_sparsity = []
        self._total_sparsity_naive = []
        self._total_naive_overhead = []
        self._current_dense_ops = 0
        self._current_sparse_ops = 0
        self._current_sparse_naive_ops = 0
        self._current_sparse_ops_kd = 0

    def _model_forward_pre_hook(self, module, input) -> None:
        self._current_dense_ops = 0
        self._current_sparse_ops = 0
        self._current_sparse_naive_ops = 0
        self._current_sparse_ops_kd = 0
        return None
    
    def _model_forward_hook(self, module, input, output) -> None:
        self._total_dense_ops.append(self._current_dense_ops)
        self._total_sparse_ops.append(self._current_sparse_ops)
        self._total_sparse_naive_ops.append(self._current_sparse_naive_ops)
        self._total_sparse_ops_kd.append(self._current_sparse_ops_kd)
        self._total_sparsity.append(1 - self._current_sparse_ops / self._current_dense_ops)
        self._total_sparsity_naive.append(1 - self._current_sparse_naive_ops / self._current_dense_ops)
        self._total_naive_overhead.append(self._current_sparse_naive_ops / self._current_sparse_ops)
        return None

    def hook_model(self, model: torch.nn.Module, verbose: bool = False) -> None:
        model.register_forward_pre_hook(self._model_forward_pre_hook)
        model.register_forward_hook(self._model_forward_hook)

        for name, module in model.named_modules():
            if isinstance(module, SparseConvolution):
                if verbose:
                    print('Registering mac counter hook for sparse convolution:', name)
                module.register_forward_hook(
                    SparseConvMacCounter(name, self)
                )
            elif isinstance(module, torch.nn.modules.conv._ConvNd):
                if verbose:
                    print('Registering mac counter hook for dense convolution:', name)
                module.register_forward_hook(
                    DenseConvMacCounter(name, self)
                )
            elif isinstance(module, torch.nn.Linear):
                if verbose:
                    print('Registering mac counter hook for dense linear:', name)
                module.register_forward_hook(
                    DenseLinearMacCounter(name, self)
                )
            # elif isinstance(module, ASPPNeck):
            #     if verbose:
            #         print('Registering mac counter hook for ASPPNeck:', name)
            #     module.register_forward_pre_hook(
            #         ASPPNeckMacCounter(name, self)
            #     )

    def _register_measurement(self, name: str, dense_ops: float, sparse_ops: float, sparse_naive_ops: float, sparse_ops_kd: float) -> None:
        if name not in self._dense_ops:
            self._dense_ops[name] = []
        if name not in self._sparse_ops:
            self._sparse_ops[name] = []
        if name not in self._sparse_naive_ops:
            self._sparse_naive_ops[name] = []
        if name not in self._sparse_ops_kd:
            self._sparse_ops_kd[name] = []
        if name not in self._sparsity:
            self._sparsity[name] = []
        if name not in self._sparsity_naive:
            self._sparsity_naive[name] = []
        if name not in self._naive_overhead:
            self._naive_overhead[name] = []
        
        self._dense_ops[name].append(dense_ops)
        self._sparse_ops[name].append(sparse_ops)
        self._sparse_naive_ops[name].append(sparse_naive_ops)
        self._sparse_ops_kd[name].append(sparse_ops_kd)
        self._sparsity[name].append(1 - sparse_ops / dense_ops)
        self._sparsity_naive[name].append(1 - sparse_naive_ops / dense_ops)
        self._naive_overhead[name].append(sparse_naive_ops / sparse_ops)

        self._current_dense_ops += dense_ops
        self._current_sparse_ops += sparse_ops
        self._current_sparse_naive_ops += sparse_naive_ops
        self._current_sparse_ops_kd += sparse_ops_kd
        return None

    def print_summary_per_layer(self):
        TAB = 4 * ' '
        for name in self._dense_ops.keys():
            print(f'\n{name}')
            print(f'{TAB}Dense ops summary : {_format_arr_stats(self._dense_ops[name])}')
            print(f'{TAB}Sparse ops summary: {_format_arr_stats(self._sparse_ops[name])}')
            print(f'{TAB}Snaive ops summary: {_format_arr_stats(self._sparse_naive_ops[name])}')
            print(f'{TAB}Sparse kd summary : {_format_arr_stats(self._sparse_ops_kd[name])}')
            print(f'{TAB}Sparsity summary  : {_format_arr_stats(self._sparsity[name])}')
            print(f'{TAB}Snaivety summary  : {_format_arr_stats(self._sparsity_naive[name])}')
            print(f'{TAB}Naive overhead    : {_format_arr_stats(self._naive_overhead[name])}')

    def print_total_summary(self):
        print('')
        print(f'Number of runs: {len(self._total_dense_ops)}')
        print(f'Total dense ops summary : {_format_arr_stats(self._total_dense_ops)}')
        print(f'Total sparse ops summary: {_format_arr_stats(self._total_sparse_ops)}')
        print(f'Total snaive ops summary: {_format_arr_stats(self._total_sparse_naive_ops)}')
        print(f'Total sparse kd summary : {_format_arr_stats(self._total_sparse_ops_kd)}')
        print(f'Total sparsity summary  : {_format_arr_stats(self._total_sparsity)}')
        print(f'Total snaivety summary  : {_format_arr_stats(self._total_sparsity_naive)}')
        print(f'Total naive overhead    : {_format_arr_stats(self._total_naive_overhead)}')

    def print_summary(self):
        self.print_summary_per_layer()
        self.print_total_summary()
