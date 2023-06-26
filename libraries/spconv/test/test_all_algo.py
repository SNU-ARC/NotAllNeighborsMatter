# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test all gemm/conv kernels.
We can't test all kernels in network because auto-tuner will only find one best kernel.
"""


import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import sys
import time
from pathlib import Path
from cumm.gemm.algospec.core import GemmAlgo, ShuffleStrideType

import numpy as np
import pccm
import torch
import torch.nn.functional as F
from spconv.test_utils import TestCase
from cumm import tensorview as tv
from cumm.conv.bases import NCHW, NHWC, ConvIterAlgo, ConvOpType
import os
from cumm.gemm.codeops import div_up
from spconv.core import AlgoHint, ConvAlgo
from spconv.pytorch.conv import expand_nd
from spconv.pytorch import ops
from spconv.algo import CONV, GEMM, BestAlgoByProfile, BestConvAlgoByProfile
from spconv.pytorch.cppcore import get_current_stream, torch_tensor_to_tv
from spconv.test_utils import generate_sparse_data, params_grid
import tqdm 

# TODO remove or release this when tf32 op is ready
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

NUMPY_DTYPE_TO_TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int8: torch.int8,

}

class SparseConvTester:
    def __init__(self, algo: ConvAlgo, subm: bool, shape: List[int], bs: int, dtype: np.dtype, N: int, K: int, C: int, 
        ksize: int, stride: int, padding: int, dilation: int) -> None:
        ndim = 3
        self.shape = shape 
        self.bs = bs 
        self.dtype = dtype 
        self.dtype_th = NUMPY_DTYPE_TO_TORCH[dtype]
        self.K = K 
        self.C = C 
        self.ksize = expand_nd(ndim, ksize) 
        self.stride = expand_nd(ndim, stride) 
        self.padding = expand_nd(ndim, padding) 
        self.dilation = expand_nd(ndim, dilation) 
        self.N = N
        self.device = torch.device("cuda:0")
        op = expand_nd(ndim, 0)
        self.kv: int = np.prod(self.ksize)
        self.num_split = 1 if algo == ConvAlgo.MaskImplicitGemm else 2

        sparse_dict = generate_sparse_data(shape, [N] * bs, C)

        voxels_np = np.ascontiguousarray(sparse_dict["features"]).astype(
            np.float32)
        indices_np = np.ascontiguousarray(
            sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)
        indices_th = torch.from_numpy(indices_np).to(self.device)

        out_inds, pair_ref, indice_num_per_loc = ops.get_indice_pairs(
            indices_th, 1, shape, ConvAlgo.Native, self.ksize, self.stride, self.padding,
            self.dilation, op, subm)
        self.indice_num_per_loc_np = indice_num_per_loc.cpu().numpy()
        self.indice_pairs_np = pair_ref.cpu().numpy()
        self.pair_native = pair_ref
        self.indice_num_per_loc = indice_num_per_loc
        if algo == ConvAlgo.Native:
            self.out_inds: torch.Tensor = out_inds
            self.num_inds_per_loc: torch.Tensor = indice_num_per_loc
            self.pair_fwd : torch.Tensor = torch.Tensor()
            self.pair_bwd: torch.Tensor = torch.Tensor()
            self.pair_mask_fwd_splits: List[torch.Tensor] = []
            self.pair_mask_bwd_splits: List[torch.Tensor] = []
            self.mask_argsort_fwd_splits: List[torch.Tensor] = []
            self.mask_argsort_bwd_splits: List[torch.Tensor] = []
            self.masks = np.array([])
        else:
            res = ops.get_indice_pairs_implicit_gemm(indices_th, bs, shape,
                                                    algo, self.ksize, self.stride, self.padding,
                                                    self.dilation, op, subm=subm)
            
            self.out_inds = res[0]
            self.num_inds_per_loc = res[1]
            self.pair_fwd = res[2]
            self.pair_bwd = res[3]
            self.pair_mask_fwd_splits = res[4]
            self.pair_mask_bwd_splits = res[5]
            self.mask_argsort_fwd_splits = res[6]
            self.mask_argsort_bwd_splits = res[7]
            self.masks = res[8]
        self.voxels_np = voxels_np
        self.indices_np = indices_np

        self.subm = subm
        if dtype == np.int8:
            self.inp = np.random.randint(-2, 2, size=[voxels_np.shape[0],
                                                    C]).astype(np.int8)
            self.weight = np.random.randint(-2, 2, size=[K, *self.ksize,
                                                    C]).astype(np.int8)
            self.output = np.random.randint(-2, 2, size=[
                self.out_inds.shape[0], K
            ]).astype(dtype)
        else:
            self.inp = np.random.uniform(-1, 1, size=[
                voxels_np.shape[0], C
            ]).astype(dtype)
            self.weight = np.random.uniform(-1, 1, size=[K, *self.ksize, C]).astype(dtype)
            self.output = np.random.uniform(-1, 1, size=[
                self.out_inds.shape[0], K
            ]).astype(dtype)
        self.weight_ref = self.weight.transpose(1, 2, 3, 0, 4)
        self.weight_ref = np.ascontiguousarray(self.weight_ref).reshape(-1, K, C)

        self.out_ref, self.din_ref, self.dw_ref = self._get_ref_output()
        self.dw_ref = np.ascontiguousarray(self.dw_ref.transpose(1, 0, 2).reshape(K, *self.ksize, C))

    def _get_ref_output(self):
        output_ref = np.zeros_like(self.output, dtype=np.float32)
        dinput_ref = np.zeros_like(self.inp, dtype=np.float32)
        dw_ref = np.zeros_like(self.weight_ref,
                                dtype=np.float32)  # KV, K, C

        for filter_offset in range(self.kv):
            if self.subm and filter_offset > self.kv // 2:
                nhot = self.indice_num_per_loc_np[self.kv - 1 - filter_offset]
            elif self.subm and filter_offset == self.kv // 2:
                nhot = self.voxels_np.shape[0]
            else:
                nhot = self.indice_num_per_loc_np[filter_offset]

            i_inds = self.indice_pairs_np[0][filter_offset][:nhot]
            o_inds = self.indice_pairs_np[1][filter_offset][:nhot]
            a = self.inp[i_inds]
            cc = a.astype(
                np.float32) @ self.weight_ref[filter_offset].T.astype(
                    np.float32)
            output_ref[o_inds] += cc
            a = self.output[o_inds]
            # NK @ KC
            cc = a.astype(
                np.float32) @ self.weight_ref[filter_offset].astype(
                    np.float32)
            dinput_ref[i_inds] += cc
            out_gather = self.output[o_inds]  # [N, K]
            inp_gather = self.inp[i_inds]  # [N, C]
            # KN @ NC
            dw_res = out_gather.astype(
                np.float32).T @ inp_gather.astype(np.float32)
            dw_ref[filter_offset] = dw_res
        return output_ref, dinput_ref, dw_ref

    def get_operands(self, op_type: ConvOpType):
        zeros_func = tv.zeros if not self.subm else tv.empty
        if op_type == ConvOpType.kBackwardInput:
            inp_tv = zeros_func(list(self.inp.shape), self.dtype, 0)
        else:
            inp_tv = tv.from_numpy(self.inp).cuda()
        if op_type == ConvOpType.kBackwardWeight:
            weight_tv = zeros_func(list(self.weight.shape), self.dtype, 0)
        else:
            weight_tv = tv.from_numpy(self.weight).cuda()
        if op_type == ConvOpType.kForward:
            output_tv = zeros_func(list(self.output.shape), self.dtype, 0)
        else:
            output_tv = tv.from_numpy(self.output).cuda()
        return inp_tv, weight_tv, output_tv

    def get_operands_torch(self, op_type: ConvOpType):
        zeros_func = torch.zeros if not self.subm else torch.empty
        if op_type == ConvOpType.kBackwardInput:
            inp_tv = zeros_func(list(self.inp.shape), dtype=self.dtype_th, device=self.device)
        else:
            inp_tv = torch.from_numpy(self.inp).cuda()
        if op_type == ConvOpType.kBackwardWeight:
            weight_tv = zeros_func(list(self.weight.shape), dtype=self.dtype_th, device=self.device)
        else:
            weight_tv = torch.from_numpy(self.weight).cuda()
        if op_type == ConvOpType.kForward:
            output_tv = zeros_func(list(self.output.shape), dtype=self.dtype_th, device=self.device)
        else:
            output_tv = torch.from_numpy(self.output).cuda()
        return inp_tv, weight_tv, output_tv

def _test_impgemm_conv_cuda(subm: bool):
    ndim = 3
    dtype_to_tol = {
        np.float32: (1e-4, 1e-4),
        np.float16: (1e-2, 1e-2),
        np.int8: (1e-4, 1e-4),
    }
    device = torch.device("cuda:0")
    shapes = [[17, 18, 19]]
    batchsizes = [1]
    dtypes = [np.float16]
    test_case = TestCase()
    in_channels = [32]
    out_channels = [64]
    if subm:
        ksizes = [3]
        strides = [1]
        paddings = [0]
        dilations = [1]
    else:
        ksizes = [2, 3]
        strides = [2]
        paddings = [0]
        dilations = [1]
    algos = [
        ConvAlgo.MaskImplicitGemm,
    ]
    arch = torch.cuda.get_device_capability()

    for shape, bs, C, K, k, s, p, d, algo, dtype in tqdm.tqdm(params_grid(
            shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations, algos, dtypes)):
        shape_prod = np.prod(shape)
        num_batch = 1500
        #num_batch = np.random.randint(int(0.2 * shape_prod), int(0.7 * shape_prod))
        #C = np.random.randint(int(0.3 * C), int(0.7 * C))
        #K = np.random.randint(int(0.3 * K), int(0.7 * K))

        tester = SparseConvTester(algo, subm, shape, bs, dtype, num_batch, K, C, k, s, p, d)
        atol, rtol = dtype_to_tol[dtype]
        mask_width_to_mask_out_fwd: Dict[int, torch.Tensor] = {}
        mask_width_to_mask_out_bwd: Dict[int, torch.Tensor] = {}

        op_types = [ConvOpType.kForward]
        spk = 1
        for op_type in op_types:
            inp_tv, weight_tv, output_tv = tester.get_operands(op_type)
            avail_desps = CONV.get_all_available(inp_tv, weight_tv, output_tv, NHWC, NHWC, NHWC, arch, op_type, -1)
            for desp in avail_desps:
                if not subm:
                    if op_type == ConvOpType.kForward:
                        output_tv.zero_()
                    else:
                        inp_tv.zero_()

                # this algo must success
                mask_width = desp.tile_shape[0]
                # if mask_width != 32:
                #     continue
                if mask_width not in mask_width_to_mask_out_fwd:
                    mask_width_to_mask_out_fwd[mask_width] = torch.zeros([2, div_up(tester.out_inds.shape[0], mask_width)],
                                      dtype=torch.int32,
                                      device=tester.device)
                mask_output_fwd = mask_width_to_mask_out_fwd[mask_width]

                if subm:
                    if desp.op_type == ConvOpType.kForward.value:
                        indice_pairs = tester.pair_fwd
                    elif desp.op_type == ConvOpType.kBackwardInput.value:
                        indice_pairs = tester.pair_bwd
                    else:
                        indice_pairs = tester.pair_fwd
                    mask_output = mask_output_fwd
                    # print([bin(x.item()) for x in masks])
                    for j in range(tester.num_split):
                        beta = 1 if j == 1 else 0
                        mask_filter = tester.masks[j].item()

                        reverse_mask = False
                        if desp.op_type == ConvOpType.kBackwardWeight.value:
                            mask_op = mask_output[j]
                        else:
                            mask_op = tester.pair_mask_fwd_splits[j]
                        if desp.op_type == ConvOpType.kBackwardInput.value:
                            reverse_mask = True
                        mask_output_run = torch_tensor_to_tv(mask_output[j], dtype=tv.uint32)
                        if desp.op_type == ConvOpType.kBackwardWeight.value:
                            mask_output_run = tv.Tensor()
                        CONV.run_with_tuned_result(
                            BestConvAlgoByProfile(desp, spk),
                            desp.op_type,
                            inp_tv,
                            weight_tv,
                            output_tv,
                            torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                            torch_tensor_to_tv(tester.mask_argsort_fwd_splits[j]),
                            mask_output_run,
                            torch_tensor_to_tv(indice_pairs),
                            reverse_mask,
                            mask_filter=mask_filter,
                            mask_width=mask_width,
                            beta=beta,
                            verbose=False,
                        )
                else:
                    if mask_width not in mask_width_to_mask_out_bwd:
                        mask_width_to_mask_out_bwd[mask_width] = torch.zeros([2, div_up(tester.indices_np.shape[0], mask_width)],
                                        dtype=torch.int32,
                                        device=tester.device)
                    mask_output_bwd = mask_width_to_mask_out_bwd[mask_width]

                    if desp.op_type == ConvOpType.kForward.value:
                        indice_pairs = tester.pair_fwd  # inp -> out
                        mask_ops = tester.pair_mask_fwd_splits
                        mask_argsorts = tester.mask_argsort_fwd_splits
                        mask_output = mask_output_fwd
                    elif desp.op_type == ConvOpType.kBackwardInput.value:
                        indice_pairs = tester.pair_bwd  # out -> inp
                        mask_ops = tester.pair_mask_bwd_splits
                        mask_argsorts = tester.mask_argsort_bwd_splits
                        mask_output = mask_output_bwd
                    else:
                        indice_pairs = tester.pair_fwd  # inp -> out
                        mask_ops = tester.pair_mask_fwd_splits
                        mask_argsorts = tester.mask_argsort_fwd_splits
                        mask_output = mask_output_fwd

                    for j in range(tester.num_split):
                        beta = 1 if j == 1 else 0
                        mask_filter = tester.masks[j].item()
                        reverse_mask = False
                        if desp.op_type == ConvOpType.kBackwardWeight.value:
                            mask_op = mask_output[j]
                        else:
                            mask_op = mask_ops[j]

                        CONV.run_with_tuned_result(
                            BestConvAlgoByProfile(desp, spk),
                            desp.op_type,
                            inp_tv,
                            weight_tv,
                            output_tv,
                            torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                            torch_tensor_to_tv(mask_argsorts[j]),
                            torch_tensor_to_tv(mask_output[j], dtype=tv.uint32),
                            torch_tensor_to_tv(indice_pairs),
                            reverse_mask,
                            mask_filter=mask_filter,
                            mask_width=mask_width,
                            beta=beta,
                            verbose=False,
                        )
                out_ref = tester.out_ref
                din_ref = tester.din_ref
                dw_ref = tester.dw_ref
                if op_type == ConvOpType.kForward:
                    out_my = output_tv.cpu().numpy()
                    if dtype != np.float16:
                        test_case.assertAllClose(out_ref, out_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(out_ref.reshape(-1) - out_my.reshape(-1))
                        if (error_norm > 5):
                            print(f"{desp}, Error={error_norm}")
                        assert error_norm < 10
                    # print(desp, )
                else:
                    din_my = inp_tv.cpu().numpy()
                    if dtype != np.float16:
                        test_case.assertAllClose(din_ref, din_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(din_ref.reshape(-1) - din_my.reshape(-1))
                        assert error_norm < 10, f"{desp}, {error_norm}, {k}, {s}, {p}, {d}"
        inp_tv, weight_tv, output_tv = tester.get_operands(ConvOpType.kBackwardWeight)

        for spk in [1, 4, 16, 64]:
            for mask_width, mask_output in mask_width_to_mask_out_fwd.items():
                avail_desps = CONV.get_all_available(inp_tv, weight_tv, output_tv, NHWC, NHWC, NHWC, arch, ConvOpType.kBackwardWeight, mask_width)
                for desp in avail_desps:
                    weight_tv.zero_()
                    if subm:
                        indice_pairs = tester.pair_fwd
                        for j in range(tester.num_split):
                            beta = 0
                            mask_filter = tester.masks[j].item()
                            mask_op = mask_output[j]
                            mask_op_tv = torch_tensor_to_tv(mask_op, dtype=tv.uint32)
                            # mask_op_np = mask_op_tv.cpu().numpy()
                            # bit_ref = np.bitwise_or.reduce(mask_op_np, axis=0)
                            # bit_my = mask_filter
                            CONV.run_with_tuned_result(
                                BestConvAlgoByProfile(desp, spk),
                                desp.op_type,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                mask_op_tv,
                                torch_tensor_to_tv(tester.mask_argsort_fwd_splits[j]),
                                tv.Tensor(),
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask=False,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                            )
                    else:
                        indice_pairs = tester.pair_fwd  # inp -> out
                        mask_ops = tester.pair_mask_fwd_splits
                        mask_argsorts = tester.mask_argsort_fwd_splits
                        for j in range(tester.num_split):
                            # beta = 1 if j == 1 else 0
                            beta = 0
                            mask_filter = tester.masks[j].item()
                            reverse_mask = False
                            mask_op = mask_output[j]

                            CONV.run_with_tuned_result(
                                BestConvAlgoByProfile(desp, spk),
                                desp.op_type,
                                inp_tv,
                                weight_tv,
                                output_tv,
                                torch_tensor_to_tv(mask_op, dtype=tv.uint32),
                                torch_tensor_to_tv(mask_argsorts[j]),
                                torch_tensor_to_tv(mask_output[j], dtype=tv.uint32),
                                torch_tensor_to_tv(indice_pairs),
                                reverse_mask,
                                mask_filter=mask_filter,
                                mask_width=mask_width,
                                beta=beta,
                                verbose=False,
                            )
                    dw_ref = tester.dw_ref
                    dw_my = weight_tv.cpu().numpy()
                    if dtype != np.float16:
                        # print(desp, spk, K, C, mask_width, algo)
                        test_case.assertAllClose(dw_ref, dw_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(dw_ref.reshape(-1) - dw_my.reshape(-1))
                        # print(desp, error_norm)
                        if (error_norm > 5):
                            print(f"{desp}, Error={error_norm}")
                        assert error_norm < 10

def _test_native_conv_cuda(subm: bool):
    ndim = 3
    dtype_to_tol = {
        np.float32: (1e-4, 1e-4),
        np.float16: (1e-2, 1e-2),
        np.int8: (1e-4, 1e-4),
    }
    device = torch.device("cuda:0")
    shapes = [[17, 18, 19]]
    batchsizes = [1]
    dtypes = [np.float16]
    test_case = TestCase()
    in_channels = [32]
    out_channels = [64]
    if subm:
        ksizes = [3]
        strides = [1]
        paddings = [0]
        dilations = [1]
    else:
        ksizes = [2, 3]
        strides = [2]
        paddings = [0]
        dilations = [1]
    arch = torch.cuda.get_device_capability()
    stream = get_current_stream()
    for shape, bs, C, K, k, s, p, d, dtype in tqdm.tqdm(params_grid(
            shapes, batchsizes, in_channels, out_channels, ksizes,
            strides, paddings, dilations, dtypes)):
        tester = SparseConvTester(ConvAlgo.Native, subm, shape, bs, dtype, 1500, K, C, k, s, p, d)
        atol, rtol = dtype_to_tol[dtype]

        kv_center = tester.kv // 2
        kv = tester.kv
        pair_in = torch_tensor_to_tv(tester.pair_native)[0]
        pair_out = torch_tensor_to_tv(tester.pair_native)[1]

        op_types = [ConvOpType.kForward]
        indice_pair_num_cpu = tester.indice_num_per_loc_np
        spk = 1

        out_ref = tester.out_ref
        din_ref = tester.din_ref
        dw_ref = tester.dw_ref.reshape(K, -1, C)

        for op_type in op_types:
            inp_th, weight_th, output_th = tester.get_operands_torch(op_type)
            weight_th = weight_th.view(K, -1, C)
            inp_tv = torch_tensor_to_tv(inp_th)
            weight_tv = torch_tensor_to_tv(weight_th)
            output_tv = torch_tensor_to_tv(output_th)

            if op_type == ConvOpType.kForward:
                a = inp_tv
                c = output_tv
                b = weight_tv.select(1, tester.kv // 2)


                avail_desps = GEMM.get_all_available(a, b, c, False, True, False, arch, ShuffleStrideType.ShuffleAC)
                for desp in avail_desps:
                    if subm:
                        torch.mm(inp_th, weight_th[:, tester.kv // 2].T, out=output_th)
                    else:
                        output_tv.zero_()
                    inited = subm
                    for i, nhot in enumerate(indice_pair_num_cpu):
                        if subm and i == kv_center:
                            continue
                        if subm and i > kv_center:
                            nhot = indice_pair_num_cpu[kv - i - 1]
                        if nhot <= 0:
                            continue
                        inp_indices = pair_in[i].slice_first_axis(0, nhot)
                        out_indices = pair_out[i].slice_first_axis(0, nhot)
                        b = weight_tv.select(1, i)
                        # inp @ filter.T, NC @ KC
                        beta = 1.0 if inited else 0.0
                        GEMM.run_with_tuned_result(
                            BestAlgoByProfile(desp, 1),
                            a,
                            b,
                            c,
                            False,
                            True,
                            False,
                            arch=arch,
                            stream=stream,
                            shuffle_type=ShuffleStrideType.ShuffleAC,
                            a_inds=inp_indices,
                            c_inds=out_indices,
                            hint=AlgoHint.Fowrard.value,
                            alpha=1.0,
                            beta=beta)
                        inited = True
                    out_my = output_tv.cpu().numpy()
                    if dtype != np.float16:
                        # error_norm = np.linalg.norm(out_ref.reshape(-1) - out_my.reshape(-1))
                        # assert error_norm < 1
                        # print(desp, K, C, k, error_norm)

                        test_case.assertAllClose(out_ref, out_my, atol=atol, rtol=rtol)
                    else:
                        error_norm = np.linalg.norm(out_ref.reshape(-1) - out_my.reshape(-1))
                        assert error_norm < 10

            elif op_type == ConvOpType.kBackwardInput:
                a = output_tv
                b = weight_tv.select(1, tester.kv // 2)
                c = inp_tv
                avail_desps = GEMM.get_all_available(a, b, c, False, False, False, arch, ShuffleStrideType.ShuffleAC)
                for desp in avail_desps:
                    if subm:
                        torch.mm(output_th, weight_th[:, tester.kv // 2], out=inp_th)
                    else:
                        inp_tv.zero_()
                    inited = subm
                    for i, nhot in enumerate(indice_pair_num_cpu):
                        if subm and i == kv_center:
                            continue
                        if subm and i > kv_center:
                            nhot = indice_pair_num_cpu[kv - i - 1]
                        if nhot <= 0:
                            continue
                        inp_indices = pair_in[i].slice_first_axis(0, nhot)
                        out_indices = pair_out[i].slice_first_axis(0, nhot)
                        b = weight_tv.select(1, i)
                        # inp @ filter.T, NC @ KC
                        beta = 1.0 if inited else 0.0
                        GEMM.run_with_tuned_result(
                            BestAlgoByProfile(desp, 1),
                            a,
                            b,
                            c,
                            False,
                            False,
                            False,
                            arch=arch,
                            stream=stream,
                            shuffle_type=ShuffleStrideType.ShuffleAC,
                            a_inds=out_indices,
                            c_inds=inp_indices,
                            hint=AlgoHint.Fowrard.value,
                            alpha=1.0,
                            beta=beta)
                        inited = True
                    din_my = inp_tv.cpu().numpy()
                    if dtype != np.float16:
                        # error_norm = np.linalg.norm(din_ref.reshape(-1) - din_my.reshape(-1))
                        # print(desp, K, C, k, error_norm)
                        test_case.assertAllClose(din_ref, din_my, atol=atol, rtol=rtol)
                        # assert error_norm < 1

                    else:
                        error_norm = np.linalg.norm(din_ref.reshape(-1) - din_my.reshape(-1))
                        assert error_norm < 10

            else:
                a = output_tv
                b = inp_tv
                c = weight_tv.select(1, tester.kv // 2)
                avail_desps = GEMM.get_all_available(a, b, c, True, False, False, arch, ShuffleStrideType.ShuffleAB)
                for desp in avail_desps:
                    inited = subm
                    weight_tv.zero_()
                    if subm:
                        torch.mm(output_th.T, inp_th, out=weight_th[:, kv_center])

                    for i, nhot in enumerate(indice_pair_num_cpu):
                        if subm and i == kv_center:
                            continue
                        if subm and i > kv_center:
                            nhot = indice_pair_num_cpu[kv - i - 1]
                        if nhot <= 0:
                            continue
                        beta = 1.0 if inited else 0.0
                        inp_indices = pair_in[i].slice_first_axis(0, nhot)
                        out_indices = pair_out[i].slice_first_axis(0, nhot)
                        a_inds = out_indices
                        b_inds = inp_indices

                        GEMM.run_with_tuned_result(BestAlgoByProfile(desp, 32),
                                                a,
                                                b,
                                                weight_tv.select(1, i),
                                                True,
                                                False,
                                                False,
                                                arch=arch,
                                                stream=stream,
                                                shuffle_type=ShuffleStrideType.ShuffleAB,
                                                a_inds=a_inds,
                                                b_inds=b_inds,
                                                hint=AlgoHint.BackwardWeight.value,
                                                alpha=1.0,
                                                beta=beta)
                    dw_my = weight_tv.cpu().numpy()
                    if dtype != np.float16:
                        error_norm = np.linalg.norm(dw_ref.reshape(-1) - dw_my.reshape(-1))
                        assert error_norm < 1

                        # test_case.assertAllClose(dw_ref, dw_my, atol=atol, rtol=rtol)
                        # print(desp, error_norm)

                    else:
                        error_norm = np.linalg.norm(dw_ref.reshape(-1) - dw_my.reshape(-1))
                        # print(desp, error_norm)
                        assert error_norm < 10


def test_all_algo_unit():
    _test_impgemm_conv_cuda(True)
    #_test_impgemm_conv_cuda(False)
    #_test_native_conv_cuda(True)
    #_test_native_conv_cuda(False)


if __name__ == "__main__":
    test_all_algo_unit()
