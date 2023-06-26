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

import asyncio
import sys
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pccm
from ccimport import compat
from pccm.core import CodeFormatter

# from myclang import clangformat
from cumm import cudasim, dtypes
from cumm import tensorview as tv
from cumm.common import GemmBasic, TensorView, TensorViewKernel
from cumm.constants import CUTLASS_MODE
from cumm.conv import kernel
from cumm.gemm.kernel import GemmKernel
from cumm.conv.bases import (NCHW, NHWC, ConvEnum, ConvIterAlgo, ConvLayout,
                             ConvLayoutType, ConvMode, ConvOpType)
from cumm.conv.params import (ConvProblem, ConvProblemCommon, gemm_abc_012_to_iwo,
                              conv_iwo_012_to_abc, get_gemm_trans_abc)
from cumm.core_cc.csrc.arrayref import ArrayPtr
from cumm.gemm.algospec import GemmAlgo
from cumm.gemm.algospec.core import TensorOpParams
from cumm.gemm.core.metaarray import MetaArray
from cumm.gemm.main import GemmAlgoParams, GemmAlgoDesp, GemmMainUnitTest, GemmParams
from cumm.gemm import codeops 
import os



class ConvAlgoDesp(GemmAlgoDesp):
    def __init__(self):
        super().__init__()
        self.add_dependency(kernel.ConvUtils)
        self.add_pybind_member("ndim", "int")
        self.add_pybind_member("op_type", "int")
        self.add_pybind_member("iter_algo", "int")
        self.add_pybind_member("layout_i, layout_w, layout_o", "int")
        self.add_pybind_member("interleave_i, interleave_w, interleave_o", "int")
        self.add_pybind_member("mask_sparse", "bool", "false")
        self.add_pybind_member("increment_k_first", "bool", "false")

        self.add_member("conv2gemm_inds", "std::array<int, 3>")
        self.add_member("gemm2conv_inds", "std::array<int, 3>")

    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.FunctionCode()
        code.arg("ndim", "int")
        code.arg("op_type", "int")
        code.ctor_init("GemmAlgoDesp", "")
        code.ctor_init("ndim", "ndim")
        code.ctor_init("op_type", f"op_type")
        code.ctor_init("iter_algo", f"{ConvIterAlgo.Optimized.value}")
        code.ctor_init("layout_i", f"{ConvLayoutType.ChannelLast.value}")
        code.ctor_init("layout_w", f"{ConvLayoutType.ChannelLast.value}")
        code.ctor_init("layout_o", f"{ConvLayoutType.ChannelLast.value}")
        code.ctor_init("interleave_i", f"1")
        code.ctor_init("interleave_w", f"1")
        code.ctor_init("interleave_o", f"1")
        code.ctor_init("conv2gemm_inds", f"conv_iwo_012_to_abc(op_type)")
        code.ctor_init("gemm2conv_inds", f"gemm_abc_012_to_iwo(op_type)")
        return code

    @pccm.pybind.mark
    @pccm.member_function(name="__repr__")
    def repr(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        check_valid();
        std::stringstream ss;
        ss << GemmAlgoDesp::__repr__();
        ss << "_C" << ndim << "_" << op_type << iter_algo;
        std::string layout_i_str = layout_i == {ConvLayoutType.ChannelFirst.value} ? "F" : "L";
        std::string layout_w_str = layout_w == {ConvLayoutType.ChannelFirst.value} ? "F" : "L";
        std::string layout_o_str = layout_o == {ConvLayoutType.ChannelFirst.value} ? "F" : "L";
        if (interleave_i > 1){{
            layout_i_str += std::to_string(interleave_i);
        }}
        if (interleave_w > 1){{
            layout_w_str += std::to_string(interleave_w);
        }}
        if (interleave_o > 1){{
            layout_o_str += std::to_string(interleave_o);
        }}

        ss << layout_i_str << layout_w_str << layout_o_str;
        if (mask_sparse){{
            ss << "_" << increment_k_first ? "SF" : "SK";
        }}
        return ss.str();
        """)
        return code.ret("std::string")

    @pccm.pybind.mark
    @pccm.static_function
    def conv_iwo_012_to_abc(self):
        code = pccm.FunctionCode()
        code.arg("op_type", "int")
        code.raw(f"""
        if (op_type == {ConvOpType.kForward.value}){{
            return {{0, 1, 2}};
        }}
        if (op_type == {ConvOpType.kBackwardInput.value}){{
            return {{2, 1, 0}};
        }}
        if (op_type == {ConvOpType.kBackwardWeight.value}){{
            return {{1, 2, 0}};
        }}
        TV_THROW_RT_ERR("unknown op type",op_type);
        """)
        return code.ret("std::array<int, 3>")


    @pccm.pybind.mark
    @pccm.static_function
    def gemm_abc_012_to_iwo(self):
        code = pccm.FunctionCode()
        code.arg("op_type", "int")
        code.raw(f"""
        if (op_type == {ConvOpType.kForward.value}){{
            return {{0, 1, 2}};
        }}
        if (op_type == {ConvOpType.kBackwardInput.value}){{
            return {{2, 1, 0}};
        }}
        if (op_type == {ConvOpType.kBackwardWeight.value}){{
            return {{2, 0, 1}};
        }}
        TV_THROW_RT_ERR("unknown op type",op_type);
        """)
        return code.ret("std::array<int, 3>")


    @pccm.pybind.mark_prop_getter(prop_name="dtype_input")
    @pccm.member_function
    def dtype_input(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        std::array<int, 3> dtypes{{dtype_a, dtype_b, dtype_c}};
        return dtypes[conv2gemm_inds[0]];
        """)
        return code.ret("int")

    @pccm.pybind.mark_prop_getter(prop_name="dtype_weight")
    @pccm.member_function
    def dtype_weight(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        std::array<int, 3> dtypes{{dtype_a, dtype_b, dtype_c}};
        return dtypes[conv2gemm_inds[1]];
        """)
        return code.ret("int")
    
    @pccm.pybind.mark_prop_getter(prop_name="dtype_output")
    @pccm.member_function
    def dtype_output(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        std::array<int, 3> dtypes{{dtype_a, dtype_b, dtype_c}};
        return dtypes[conv2gemm_inds[2]];
        """)
        return code.ret("int")

    @pccm.pybind.mark 
    @pccm.member_function
    def supported(self):
        code = pccm.FunctionCode()
        code.arg("m,n,k, C, K, mask_width", "int")
        code.raw(f"""
        bool res = GemmAlgoDesp::supported(m, n, k);
        if (mask_sparse){{
            if (op_type == {ConvOpType.kForward.value}){{
                // NC -> NRSC @ KRSC
                res &= C % element_per_access_a == 0;
            }}else if (op_type == {ConvOpType.kBackwardInput.value}){{
                // NK -> NRSK @ KRSC -> RSKC
                res &= K % element_per_access_a == 0;
            }}else{{
                // NK @ NC -> NRSC
                // we must ensure every k iteration only have one mask (from forward),
                res &= mask_width % tile_shape[2] == 0;
                res &= K % element_per_access_a == 0;
                res &= C % element_per_access_b == 0;
            }}
        }}
        return res;
        """)
        return code.ret("bool")

    @pccm.pybind.mark 
    @pccm.member_function
    def query_conv_workspace_size(self):
        code = pccm.FunctionCode()
        code.arg("m,n,k,split_k_slices,kv", "int")
        code.raw(f"""
        if (!mask_sparse){{
            return query_workspace_size(m, n, k, split_k_slices);
        }}
        auto logical_tile_count = ConvUtils::get_spconv_logical_tile_count(m, n, k, 
            tile_shape[0], tile_shape[1], split_k_slices, kv, op_type);
        int workspace_size = 0;
        if (split_k_slices > 1){{
            if (split_k_serial()){{
                workspace_size = sizeof(int) * logical_tile_count[0] * logical_tile_count[1];
            }} else if (split_k_parallel()){{
                workspace_size = tv::detail::sizeof_dtype(tv::DType(dacc)) * m * n * logical_tile_count[2];
            }} else{{
                TV_THROW_INVALID_ARG("not impemented");
            }}
        }}
        return workspace_size;
        """)
        return code.ret("int")

    @pccm.pybind.mark 
    @pccm.member_function
    def supported_ldx_conv(self):
        code = pccm.FunctionCode()
        code.arg("ldi, ldw, ldo", "int")
        code.raw(f"""
        bool res = true;
        std::array<int, 3> epas{{element_per_access_a, element_per_access_b, element_per_access_c}};
        int epa_i = epas[conv2gemm_inds[0]];
        int epa_w = epas[conv2gemm_inds[1]];
        int epa_o = epas[conv2gemm_inds[2]];

        if (epa_i > 0){{
            res &= ldi % epa_i == 0;
        }}
        if (epa_w > 0){{
            res &= ldw % epa_w == 0;
        }}
        if (epa_o > 0){{
            res &= ldo % epa_o == 0;
        }}
        return res;
        """)
        return code.ret("bool")

class ConvParams(pccm.Class, pccm.pybind.PybindClassMixin):
    def __init__(self):
        super().__init__()
        self.add_dependency(ConvAlgoDesp)
        self.add_pybind_member("conv_algo_desp", "ConvAlgoDesp")
        self.add_pybind_member("input,weight,output", "tv::Tensor", pyanno="cumm.tensorview.Tensor")
        self.add_pybind_member("split_k_slices", "int", "1")
        self.add_pybind_member("padding,stride,dilation", "std::vector<int>")
        self.add_pybind_member("alpha,beta", "float")
        self.add_pybind_member("mask_width", "int")
        self.add_pybind_member("mask_filter", "uint32_t")
        self.add_pybind_member("reverse_mask", "bool")
        self.add_pybind_member("verbose", "bool")
        self.add_pybind_member("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)", pyanno="cumm.tensorview.CUDAKernelTimer")

        self.add_pybind_member("is_train", "bool")

        self.add_pybind_member("workspace",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("mask",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("mask_argsort",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("indices",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("mask_output",
                               "tv::Tensor",
                               "tv::Tensor()",
                               pyanno="cumm.tensorview.Tensor = Tensor()")
        self.add_pybind_member("stream", "std::uintptr_t", "0", pyanno="int")

    @pccm.pybind.mark
    @pccm.constructor
    def default_ctor(self):
        code = pccm.FunctionCode()
        code.arg("ndim", "int")
        code.arg("op_type", "int")
        code.arg("timer", "tv::CUDAKernelTimer", "tv::CUDAKernelTimer(false)", pyanno="cumm.tensorview.CUDAKernelTimer = CUDAKernelTimer(False)")

        code.ctor_init("conv_algo_desp", "ndim, op_type")
        code.ctor_init("input", "tv::Tensor()")
        code.ctor_init("weight", "tv::Tensor()")
        code.ctor_init("output", "tv::Tensor()")
        code.ctor_init("padding", "std::vector<int>()")
        code.ctor_init("stride", "std::vector<int>()")
        code.ctor_init("dilation", "std::vector<int>()")
        code.ctor_init("alpha", "1.0")
        code.ctor_init("beta", "0.0")
        code.ctor_init("mask_width", "-1")
        code.ctor_init("mask_filter", "0xffffffff")
        code.ctor_init("reverse_mask", "false")
        code.ctor_init("verbose", "false")
        code.ctor_init("timer", "timer")

        code.ctor_init("is_train", "false")

        return code


def seq(*vals):
    return np.array([*vals], dtype=np.int64)


class ConvAlgoParams(GemmAlgoParams):
    def __init__(self,
                 ndim: int,
                 op_type: ConvOpType,
                 iter_algo: ConvIterAlgo,
                 ts: Tuple[int, int, int],
                 wts: Tuple[int, int, int],
                 num_stage: int,
                 dtype_shorts: str,
                 layout_desp_input: ConvLayout,
                 layout_desp_weight: ConvLayout,
                 layout_desp_output: ConvLayout,
                 algo: GemmAlgo,
                 tensorop: Optional[TensorOpParams] = None,
                 splitk_serial: bool = False,
                 splitk_parallel: bool = False,
                 mask_sparse: bool = False,
                 increment_k_first: bool = False,
                 access_per_vector: int = 1):
        trans_a, trans_b, trans_c = get_gemm_trans_abc(op_type)
        super().__init__(ts, wts, num_stage, dtype_shorts, trans_a, trans_b, 
            trans_c, algo, tensorop, splitk_serial, splitk_parallel, access_per_vector=access_per_vector)
        self.ndim = ndim
        self.op_type = op_type
        self.iter_algo = iter_algo
        self.mask_sparse = mask_sparse
        self.increment_k_first = increment_k_first

        indices = conv_iwo_012_to_abc(op_type)
        dtypes_abc = [self.dtype_a, self.dtype_b, self.dtype_c]
        self.dtype_input = dtypes_abc[indices[0]]
        self.dtype_weight = dtypes_abc[indices[1]]
        self.dtype_output = dtypes_abc[indices[2]]

        self.layout_desp_input = layout_desp_input
        self.layout_desp_weight = layout_desp_weight
        self.layout_desp_output = layout_desp_output

    def skipped(self):
        if self.op_type != ConvOpType.kForward and self.dtype_a.itemsize(
            ) == 1:
            return True
            
        return super().skipped()

    def get_algo_name(self):
        res = super().get_algo_name()
        res += f"_C{self.ndim}{self.op_type.value}{self.iter_algo.value}"
        res += f"{self.layout_desp_input}{self.layout_desp_weight}{self.layout_desp_output}"
        if self.mask_sparse:
            res += "_SF" if not self.increment_k_first else "_SK"
        return res


def gen_gemm_params(op_types: List[ConvOpType],
                    ts,
                    wts,
                    ndim: int,
                    iter_algo: ConvIterAlgo,
                    stage: int,
                    dtypes_string: Union[str, List[str]],
                    li: ConvLayout,
                    lw: ConvLayout,
                    lo: ConvLayout,
                    algo: GemmAlgo,
                    tensorop: Optional[TensorOpParams],
                    splitk_serial: bool = False,
                    splitk_parallel: bool = False,
                    mask_sparse: bool = False,
                    increment_k_first: bool = False,
                    access_per_vector: int = 1):
    res = []
    if not isinstance(dtypes_string, list):
        dtypes_string = [dtypes_string]
    for dts in dtypes_string:
        for op_type in op_types:
            if op_type == ConvOpType.kBackwardWeight:
                p = ConvAlgoParams(ndim,
                                op_type,
                                iter_algo,
                                ts,
                                wts,
                                stage,
                                dts,
                                li,
                                lw,
                                lo,
                                algo,
                                tensorop,
                                True,
                                splitk_parallel,
                                mask_sparse,
                                increment_k_first,
                                access_per_vector)
            else:
                p = ConvAlgoParams(ndim, op_type, iter_algo, ts, wts, stage,
                                dts, li, lw, lo, algo, tensorop,
                                splitk_serial, splitk_parallel, mask_sparse,
                                increment_k_first,
                                access_per_vector)

            if not p.skipped():
                res.append(p)
    return res

ConvFwdAndBwdInput = [ConvOpType.kForward, ConvOpType.kBackwardInput]
ConvBwdWeight = [ConvOpType.kBackwardWeight]
ConvAllOp = [ConvOpType.kForward, ConvOpType.kBackwardInput, ConvOpType.kBackwardWeight]

def gen_spwgrad_params(ts,
                       wts,
                       ndim: int,
                       iter_algo: ConvIterAlgo,
                       stage: int,
                       dtypes_string: str,
                       li: ConvLayout,
                       lw: ConvLayout,
                       lo: ConvLayout,
                       algo: GemmAlgo,
                       tensorop: Optional[TensorOpParams],
                       splitk_serial: bool = False,
                       splitk_parallel: bool = False,
                       mask_sparse: bool = False,
                       increment_k_first: bool = False,
                       access_per_vector: int = 1):
    p = ConvAlgoParams(ndim,
                       ConvOpType.kBackwardWeight,
                       iter_algo,
                       ts,
                       wts,
                       stage,
                       dtypes_string,
                       li,
                       lw,
                       lo,
                       algo,
                       tensorop,
                       True,
                       splitk_parallel,
                       mask_sparse,
                       increment_k_first,
                       access_per_vector)
    return [p]


def gen_gemm_kernels(params: ConvAlgoParams):
    return kernel.ConvKernel(params.ndim,
                             params.op_type,
                             params.iter_algo,
                             params.ts,
                             params.wts,
                             params.num_stage,
                             dtype_a=params.dtype_a,
                             dtype_b=params.dtype_b,
                             dtype_c=params.dtype_c,
                             dtype_acc=params.dtype_acc,
                             dtype_comp=params.dtype_comp,
                             layout_desp_input=params.layout_desp_input,
                             layout_desp_output=params.layout_desp_output,
                             layout_desp_weight=params.layout_desp_weight,
                             algo=params.algo,
                             tensorop=params.tensorop,
                             splitk_serial=params.splitk_serial,
                             splitk_parallel=params.splitk_parallel,
                             mask_sparse=params.mask_sparse,
                             increment_k_first=params.increment_k_first,
                             access_per_vector=params.access_per_vector)

SHUFFLE_SIMT_PARAMS = []

SHUFFLE_VOLTA_PARAMS = []
SHUFFLE_TURING_PARAMS = []
class ConvMainUnitTest(pccm.Class):
    def __init__(self, conv_params: Optional[List[ConvAlgoParams]] = None):
        super().__init__()
        self.add_dependency(TensorView, GemmBasic, ConvEnum, ConvParams, ConvProblemCommon)
        # unit test params: [ts, wts, stage, dtypes, trans, algo, tensorop]
        if conv_params is None:
            is_debug = os.getenv("CUMM_DEBUG", None)
            if is_debug is not None and is_debug == "1":
                simt_params: List[ConvAlgoParams] = [
                    # *gen_gemm_params((64, 128, 32), (32, 64, 32), 2, ConvIterAlgo.Optimized, 2, "s8,s8,s32,s32,s32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.SimtDP4A, None),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8))),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None),
                    
                    # *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True),
                    # *gen_gemm_params(ConvBwdWeight, (128, 128, 8), (32, 64, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 64, 32), (32, 32, 16), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 64), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 64), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 64), (32, 32, 64), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    *gen_gemm_params(ConvFwdAndBwdInput, (32, 128, 64), (32, 64, 64), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),

                    *gen_gemm_params(ConvBwdWeight, (128, 128, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                        NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 128, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (128, 64, 32), (64, 32, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 64, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (128, 256, 32), (64, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 64, 32), (32, 32, 16), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 32, 64), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 64), (32, 64, 64), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Turing, TensorOpParams((16, 8, 8)), mask_sparse=True, increment_k_first=True, access_per_vector=1),

                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 128, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Volta, TensorOpParams((8, 8, 4)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvFwdAndBwdInput, (64, 64, 32), (32, 32, 32), 3, ConvIterAlgo.Optimized, 2, ["f16,f16,f16,f32,f32", "f16,f16,f16,f16,f16"],
                    #     NHWC, NHWC, NHWC, GemmAlgo.Volta, TensorOpParams((8, 8, 4)), mask_sparse=True, increment_k_first=True, access_per_vector=1),
                    # *gen_gemm_params(ConvBwdWeight, (64, 256, 32), (32, 64, 32), 3, ConvIterAlgo.Optimized, 2, "f16,f16,f16,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Volta, TensorOpParams((8, 8, 4)), mask_sparse=True, increment_k_first=True, access_per_vector=1),

                    # *gen_spwgrad_params((128, 128, 8), (32, 64, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True, mask_width=32),
                    # *gen_gemm_params((32, 128, 16), (32, 32, 8), 3, ConvIterAlgo.Optimized, 2, "f32,f32,f32,f32,f32",
                    #     NHWC, NHWC, NHWC, GemmAlgo.Simt, None, mask_sparse=True, increment_k_first=True, mask_width=32),
                    # *gen_gemm_params(
                    #     (32, 128, 16), (32, 32, 8), 2, ConvIterAlgo.Optimized, 2,
                    #     "f32,f32,f32,f32,f32", NHWC, NHWC, NHWC, GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 32, 8), (8, 16, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((8, 64, 8), (8, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((16, 32, 8), (16, 16, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((64, 128, 8), (32, 32, 8), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f16,f16", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 8), (32, 64, 8), 2, "f16,f16,f16,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f32,f32,f32,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params_rowmajor_c((128, 128, 16), (32, 64, 16), 2, "f16,f16,f16,f32,f32", GemmAlgo.Simt, None),
                    # *gen_gemm_params((128, 128, 32), (32, 64, 32), 2, "s8,s8,s32,s32,s32", GemmAlgo.SimtDP4A, None),
                ]  # type: List[ConvAlgoParams]
                volta_params: List[ConvAlgoParams] = [
                ]
                turing_params: List[ConvAlgoParams] = [
                ]
            else:
                simt_params: List[ConvAlgoParams] = [
                    *SHUFFLE_SIMT_PARAMS,
                ] 
                volta_params: List[ConvAlgoParams] = [
                    *SHUFFLE_VOLTA_PARAMS,
                ]
                turing_params: List[ConvAlgoParams] = [
                    *SHUFFLE_TURING_PARAMS,
                ]
            self.all_params = simt_params + volta_params + turing_params
            self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]

        else:
            assert len(conv_params) > 0
            self.all_params = conv_params
            self.all_kernels = [gen_gemm_kernels(p) for p in self.all_params]
        self.ker_names = [k.get_algo_name() for k in self.all_kernels]
        assert len(set(self.ker_names)) == len(self.ker_names), "kernel must unique"

    @staticmethod
    def _get_layout_types(ker: kernel.ConvKernel):
        p = ker.problem
        return (p.layout_desp_input.layout_type, p.layout_desp_weight.layout_type, p.layout_desp_output.layout_type)
    
    @staticmethod
    def _get_layout_interleaves(ker: kernel.ConvKernel):
        p = ker.problem
        return (p.layout_desp_input.interleave, p.layout_desp_weight.interleave, p.layout_desp_output.interleave)

    @staticmethod
    def _get_sparse_params(ker: kernel.ConvKernel):
        return (ker.mask_sparse, ker.increment_k_first)


    @staticmethod
    def conv_select_helper_stage1(kernels: List[kernel.ConvKernel], code: pccm.FunctionCode):
        ndim_op_iter_to_kers = codeops.group_by(
            lambda x: (x.problem.ndim, x.problem.op_type, x.iter_algo), kernels)
        for ndim_op_iter, ndim_op_iter_kers in ndim_op_iter_to_kers.items():
            if_tests = [
                f"algo_desp.ndim == {ndim_op_iter[0]}",
                f"algo_desp.op_type == {ndim_op_iter[1].value}",
                f"algo_desp.iter_algo == {ndim_op_iter[2].value}",
            ]
            with code.if_(" && ".join(if_tests)):
                li_lw_lo_to_kers = codeops.group_by(
                    ConvMainUnitTest._get_layout_types, ndim_op_iter_kers)
                for li_lw_lo, lilwlo_kers in li_lw_lo_to_kers.items():
                    if_tests = [
                        f"algo_desp.layout_i == {li_lw_lo[0].value}",
                        f"algo_desp.layout_w == {li_lw_lo[1].value}",
                        f"algo_desp.layout_o == {li_lw_lo[2].value}",
                    ]
                    with code.if_(" && ".join(if_tests)):
                        lii_lwi_loi_to_kers = codeops.group_by(
                            ConvMainUnitTest._get_layout_interleaves, lilwlo_kers)
                        for liilwiloi, liilwiloi_kers in lii_lwi_loi_to_kers.items():
                            if_tests = [
                                f"algo_desp.interleave_i == {liilwiloi[0]}",
                                f"algo_desp.interleave_w == {liilwiloi[1]}",
                                f"algo_desp.interleave_o == {liilwiloi[2]}",
                            ]
                            with code.if_(" && ".join(if_tests)):
                                ms_ikf_mw_to_kers = codeops.group_by(
                                    ConvMainUnitTest._get_sparse_params, liilwiloi_kers)
                                for ms_ikf_mw, ms_ikf_mw_kers in ms_ikf_mw_to_kers.items():
                                    assert len(ms_ikf_mw_kers) == 1, "find multiple kernels for one configuration"
                                    if_tests = [
                                        f"algo_desp.mask_sparse == {pccm.boolean(ms_ikf_mw[0])}",
                                        f"algo_desp.increment_k_first == {pccm.boolean(ms_ikf_mw[1])}",
                                    ]
                                    with code.if_(" && ".join(if_tests)):
                                        yield ms_ikf_mw_kers


    @staticmethod
    def conv_select_helper(kernels: List[Union[kernel.ConvKernel]], code: pccm.FunctionCode):
        for kers in GemmMainUnitTest.matmul_select_helper_stage2(kernels, code, False, False):
            yield from ConvMainUnitTest.conv_select_helper_stage1(kers, code)

    @pccm.pybind.mark
    @pccm.static_function
    def extract_mnk(self):
        code = pccm.FunctionCode()
        code.arg("op_type", "int")
        code.arg("N, C, K", "int")
        code.arg("kernel_volume, in_prod, out_prod", "int")
        code.arg("mask_sparse", "bool")
        code.raw(f"""
        auto op_type_enum = static_cast<ConvEnum::OpType>(op_type);
        auto res = ConvProblemCommon::implicit_gemm_mnk(op_type_enum, N, C, K, 
            kernel_volume, in_prod, out_prod, mask_sparse);
        return {{res[0], res[1], res[2]}};
        """)
        return code.ret("std::array<int, 3>")

    @pccm.pybind.mark
    @pccm.cuda.static_function
    def implicit_gemm2(self):
        code = pccm.FunctionCode()
        for p, ker in zip(self.all_params, self.all_kernels):
            code.add_param_class("cp" + ker.get_algo_name(), ker.gemm_params,
                                 "ConvParams" + ker.get_algo_name())
            code.add_param_class(ker.get_algo_name(), ker,
                                 "Conv" + ker.get_algo_name())
        code.arg("params",
                 "ConvParams",
                 pyanno="ConvParams")
        code.raw(f"""
        // auto rtxtimer = tv::CPUTimer<>();
        int groups = 1;
        bool found = false;
        auto& algo_desp = params.conv_algo_desp;
        auto dacc = tv::DType(algo_desp.dacc);
        auto dcomp = tv::DType(algo_desp.dcomp);
        ConvEnum::IterAlgo iter_algo = static_cast<ConvEnum::IterAlgo>(algo_desp.iter_algo);
        ConvEnum::OpType op_type = static_cast<ConvEnum::OpType>(algo_desp.op_type);
        // ConvEnum::LayoutType i_ltype = static_cast<ConvEnum::LayoutType>(algo_desp.layout_i);
        // ConvEnum::LayoutType w_ltype = static_cast<ConvEnum::LayoutType>(algo_desp.layout_w);
        // ConvEnum::LayoutType o_ltype = static_cast<ConvEnum::LayoutType>(algo_desp.layout_o);

        int split_k_slices = params.split_k_slices;
        auto workspace = params.workspace;
        auto input = params.input;
        auto weight = params.weight;
        auto output = params.output;

        auto indices = params.indices;
        auto mask = params.mask;
        auto mask_argsort = params.mask_argsort;
        auto mask_output = params.mask_output;
        auto padding = params.padding;
        auto stride = params.stride;
        auto dilation = params.dilation;
        auto mask_width = params.mask_width;
        auto evtimer = params.timer;
        auto is_train = params.is_train;

        """)

        for kers in self.conv_select_helper(self.all_kernels, code):
            ker = kers[0]
            p = ker.problem
            param_type_str = "ConvParams" + ker.get_algo_name()
            indices = conv_iwo_012_to_abc(ker.problem.op_type)
            inv_indices = gemm_abc_012_to_iwo(ker.problem.op_type)
            dtypes_abc = [ker.dtype_a, ker.dtype_b, ker.dtype_c]
            dtypes_iwo = [dtypes_abc[i] for i in indices]
            param_cls_name = "ConvParams" + ker.get_algo_name()
            param_cls_ns = "cp" + ker.get_algo_name()

            if not ker.support_splitk():
                code.raw(f"""
                TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                """)
            # TODO if input is NCxHWx
            # TODO if input weight and output have different layout
            dim_start = 2 if p.layout_desp_weight.is_channel_first() else 1
            io_ndim = 2 if p.mask_sparse else p.ndim + 2
            weight_ndim = 3 if p.mask_sparse else p.ndim + 2
            abc_names = ["a", "b", "c"]
            abc_names = [abc_names[i] for i in indices]
            input_names = ["input", "weight", "output"]
            input_names = [input_names[i] for i in inv_indices]

            code.raw(f"""
            // {ker.get_algo_name()}
            found = true;
            TV_ASSERT_RT_ERR(input.ndim() == {io_ndim}, "error");
            TV_ASSERT_RT_ERR(weight.ndim() == {weight_ndim}, "error");
            TV_ASSERT_RT_ERR(output.ndim() == {io_ndim}, "error");
            int N = input.dim(0);
            int C = {'input.dim(1)' if p.layout_desp_input.is_channel_first() else f'input.dim({io_ndim - 1})'};
            int K = weight.dim(0);
            int K2 = {'output.dim(1)' if p.layout_desp_output.is_channel_first() else f'output.dim({io_ndim - 1})'};
            TV_ASSERT_RT_ERR(K2 == K, "error");
            """)
            if p.mask_sparse:
                if p.op_type == ConvOpType.kBackwardWeight:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(mask_width > 0 && mask_width % {ker.tile_shape[2]} == 0, "error");
                    """)
                code.raw(f"""
                TV_ASSERT_RT_ERR(!indices.empty(), "error");
                TV_ASSERT_RT_ERR(!mask.empty(), "error");
                TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                int kernel_volume = weight.dim({dim_start});
                tv::check_shape(indices, {{kernel_volume, -1}});
                N = indices.dim(1);

                {param_cls_ns}::ConvProblem problem(N, C, K, kernel_volume, 
                    ConvEnum::Mode::kCrossCorrelation, split_k_slices, groups);
                """)
                if p.op_type == ConvOpType.kBackwardWeight:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                    TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * {ker.dtype_b.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    """)
                elif p.op_type == ConvOpType.kForward:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                    TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    """)
                else:
                    code.raw(f"""
                    TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * {ker.dtype_a.bitsize()} / 8 < std::numeric_limits<int32_t>::max(), 
                        "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                    TV_ASSERT_RT_ERR(N == input.dim(0), "error");
                    """)
            else:
                code.raw(f"""
                tv::array<int, {p.ndim}> input_dims, output_dims;
                tv::array<int, {p.ndim}> ksize;
                TV_ASSERT_RT_ERR({p.ndim} == padding.size() && {p.ndim} == stride.size() && {p.ndim} == dilation.size(), "error");
                for (int i = {dim_start}; i < {dim_start + p.ndim}; ++i){{
                    ksize[i - {dim_start}] = weight.dim(i);
                    input_dims[i - {dim_start}] = input.dim(i);
                    output_dims[i - {dim_start}] = output.dim(i);
                }}
                int kernel_volume = tv::arrayops::prod(ksize);
                tv::array<int, {p.ndim}> padding_arr{{{code.unpack([f"padding[{i}]" for i in range(p.ndim)])}}};
                tv::array<int, {p.ndim}> stride_arr{{{code.unpack([f"stride[{i}]" for i in range(p.ndim)])}}};
                tv::array<int, {p.ndim}> dilation_arr{{{code.unpack([f"dilation[{i}]" for i in range(p.ndim)])}}};
                auto output_dims_check_again = {param_cls_ns}::ConvProblem::calc_output_dims(input_dims, ksize, padding_arr, stride_arr, dilation_arr);
                for (int i = 0; i < {p.ndim}; ++i){{
                    TV_ASSERT_RT_ERR(output_dims_check_again[i] == output_dims[i], "error");
                }}
                {param_cls_ns}::ConvProblem problem(N, C, K, input_dims, output_dims, ksize, padding_arr, stride_arr, dilation_arr, 
                    ConvEnum::Mode::kCrossCorrelation, split_k_slices, groups);
                """)
            code.raw(f"""
            auto mnk = problem.implicit_gemm_mnk(op_type);

            TV_ASSERT_RT_ERR(algo_desp.supported(mnk[0], mnk[1], mnk[2], C, K, mask_width), "error");

            int workspace_size = algo_desp.query_conv_workspace_size(mnk[0], mnk[1], mnk[2], split_k_slices, kernel_volume);

            auto ctx = tv::Context();
            ctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(params.stream));
            if (workspace_size > 0){{
                if (!workspace.empty()){{
                    workspace.zero_(ctx);
                    TV_ASSERT_RT_ERR(workspace.nbytes() >= workspace_size, 
                        "workspace at least", workspace_size, "bytes.");
                }}else{{
                    workspace = tv::empty({{workspace_size}}, tv::uint8, 0);
                    workspace.zero_(ctx);
                }}
            }}
            """)
            input_names = ["input", "weight", "output"]
            input_names = [input_names[i] for i in inv_indices]
            code.raw(f"""
            auto a_ten = {input_names[0]};
            auto b_ten = {input_names[1]};
            auto c_ten = {input_names[2]};
            """)
            if p.mask_sparse:
                mask_out_ptr = "mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), "
                if p.op_type != ConvOpType.kForward:
                    mask_out_ptr = ""
                
                mask_width_str = "mask_width,"
                if p.op_type != ConvOpType.kBackwardWeight:
                    mask_width_str = ""

                code.raw(f"""
                {param_type_str} ker_params(
                    problem, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                    c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(),
                    mask.data_ptr<uint32_t>(), mask_argsort.data_ptr<int32_t>(),
                    indices.data_ptr<int32_t>(), {mask_out_ptr} params.mask_filter, 
                    params.reverse_mask, {mask_width_str} is_train,
                    {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta){", split_k_slices, workspace.raw_data()" if ker.support_splitk() else ""});
                """)
            else:
                code.raw(f"""
                {param_type_str} ker_params(
                    problem, a_ten.data_ptr<{ker.dtype_a}>(), b_ten.data_ptr<{ker.dtype_b}>(),
                    c_ten.data_ptr<{ker.dtype_c}>(), c_ten.data_ptr<{ker.dtype_c}>(), 
                    {ker.dtype_comp}(params.alpha), {ker.dtype_comp}(params.beta){", split_k_slices, workspace.raw_data()" if ker.support_splitk() else ""});
                """)
            if p.op_type == ConvOpType.kBackwardWeight:
                code.raw(f"""
                int num_reduced_mask = tv::div_up(ker_params.problem.N, ker_params.mask_width);
                TV_ASSERT_RT_ERR(mask.dim(0) >= num_reduced_mask, "error");
                """)
            code.raw(f"""
            tv::cuda::Launch launcher(ker_params.grid_dims, dim3({ker.num_threads}),
                                        {ker.smem_size}, reinterpret_cast<cudaStream_t>(params.stream));
            cudaError_t result;
            if ({ker.smem_size} >= (48 << 10)) {{
                result = cudaFuncSetAttribute({ker.get_algo_name()}::conv_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                {ker.smem_size});
                TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                result = cudaFuncSetAttribute(
                    {ker.get_algo_name()}::conv_kernel,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
            }}
            """)
            # if cudasim.enable_debug():
            code.raw(f"""
            auto timer = tv::CUDATimer(params.verbose);
            // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
            {{
                tv::CUDAKernelTimerGuard timerguard(\"{ker.get_algo_name()}\", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                launcher({ker.get_algo_name()}::conv_kernel, ker_params);
            }}
            """)
            if p.mask_sparse:
                code.raw(f"""
                TV_CHECK_CUDA_ERR_V2("{ker.get_algo_name()}", "error with params", input.shape(), weight.shape(), output.shape(), 
                    indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                """)
            else:
                code.raw(f"""
                TV_CHECK_CUDA_ERR_V2("{ker.get_algo_name()}", "error with params", input.shape(), weight.shape(), output.shape());
                """)

            # if cudasim.enable_debug():
            code.raw(f"""
            if (params.verbose){{
                cudaFuncAttributes attr;
                checkCudaErrors(
                    cudaFuncGetAttributes(&attr, {ker.get_algo_name()}::conv_kernel));
                tv::ssprint("{ker.get_algo_name()} kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
            }}

            """)
            code.raw(f"return;")
        code.raw("""
        if (!found){
            TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", algo_desp.__repr__(), tv::dtype_str(input.dtype()), 
                tv::dtype_str(weight.dtype()), tv::dtype_str(output.dtype()), tv::dtype_str(dacc), 
                tv::dtype_str(dcomp));
        }
        """)
        return code
    
    @pccm.pybind.mark
    @pccm.static_function
    def get_all_conv_algo_desp(self):
        code = pccm.FunctionCode()
        code.raw(f"""
        std::vector<ConvAlgoDesp> desps;
        """)
        for ker in self.all_kernels:
            code.raw("{")
            code.raw(f"""
            ConvAlgoDesp desp({ker.problem.ndim}, {ker.problem.op_type.value});
            desp.dtype_a = {ker.dtype_a.tv_dtype};
            desp.dtype_b = {ker.dtype_b.tv_dtype};
            desp.dtype_c = {ker.dtype_c.tv_dtype};
            desp.dacc = {ker.dtype_acc.tv_dtype};
            desp.dcomp = {ker.dtype_comp.tv_dtype};

            desp.trans_a_set({pccm.boolean(ker.trans_a)});
            desp.trans_b_set({pccm.boolean(ker.trans_b)});
            desp.trans_c_set({pccm.boolean(ker.trans_c)});
            desp.tile_shape = {{{ker.tile_shape[0]}, {ker.tile_shape[1]}, {ker.tile_shape[2]}}};
            desp.warp_tile_shape = {{{ker.warp_tile_shape[0]}, {ker.warp_tile_shape[1]}, {ker.warp_tile_shape[2]}}};
            """)
            if ker.tensorop is not None:
                code.raw(
                    f"desp.tensorop = {{{ker.tensorop[0]}, {ker.tensorop[1]}, {ker.tensorop[2]}}};"
                )
            else:
                code.raw(f"desp.tensorop = {{-1, -1, -1}};")
            code.raw(f"""
            desp.num_stage = {ker.num_stage};
            desp.algo = "{ker.algo.value}";
            desp.split_k_serial_set({pccm.boolean(ker.splitk_serial)});
            desp.split_k_parallel_set({pccm.boolean(ker.splitk_parallel)});
            desp.shuffle_type = "{ker.shuffle_stride.value}";
            desp.element_per_access_a = {ker.input_spec.input_iter_a.element_per_acc};
            desp.element_per_access_b = {ker.input_spec.input_iter_b.element_per_acc};
            desp.element_per_access_c = {ker.output_spec.out_iter.element_per_acc};
            desp.access_per_vector = {ker.access_per_vector};

            // Conv attrs
            desp.ndim = {ker.problem.ndim};
            desp.op_type = {ker.problem.op_type.value};
            desp.iter_algo = {ker.iter_algo.value};
            desp.layout_i = {ker.problem.layout_desp_input.layout_type.value};
            desp.layout_w = {ker.problem.layout_desp_weight.layout_type.value};
            desp.layout_o = {ker.problem.layout_desp_output.layout_type.value};
            desp.interleave_i = {ker.problem.layout_desp_input.interleave};
            desp.interleave_w = {ker.problem.layout_desp_weight.interleave};
            desp.interleave_o = {ker.problem.layout_desp_output.interleave};
            desp.mask_sparse = {pccm.boolean(ker.mask_sparse)};
            desp.increment_k_first = {pccm.boolean(ker.increment_k_first)};
            desps.push_back(desp);
            """)
            code.raw("}")
        code.raw(f"""
        return desps;
        """)
        return code.ret("std::vector<ConvAlgoDesp>", pyanno="List[ConvAlgoDesp]")

    # @lineprof.lineprof_wrapper_cpp
    def implicit_gemm_python(self,
                             input_: np.ndarray,
                             weight: np.ndarray,
                             output: np.ndarray,
                             input_meta: np.ndarray,
                             weight_meta: np.ndarray,
                             output_meta: np.ndarray,
                             padding: List[int],
                             stride: List[int],
                             dilation: List[int],
                             ndim: int,
                             iter_algo: ConvIterAlgo,
                             op_type: ConvOpType,
                             i_ltype: ConvLayoutType,
                             w_ltype: ConvLayoutType,
                             o_ltype: ConvLayoutType,
                             ts: np.ndarray,
                             wts: np.ndarray,
                             num_stage: int,
                             dacc: dtypes.DType,
                             dcomp: dtypes.DType,
                             algo: str,
                             tensorop: np.ndarray,
                             i_interleave: int = 1,
                             w_interleave: int = 1,
                             o_interleave: int = 1):
        found = False
        for p, ker in zip(self.all_params, self.all_kernels):
            indices = conv_iwo_012_to_abc(p.op_type)
            inv_indices = gemm_abc_012_to_iwo(p.op_type)
            dtypes_abc = [p.dtype_a, p.dtype_b, p.dtype_c]
            dtypes_iwo = [dtypes_abc[i] for i in indices]

            if_tests = [
                dtypes_iwo[0].npdtype() == input_.dtype,
                dtypes_iwo[1].npdtype() == weight.dtype,
                dtypes_iwo[2].npdtype() == output.dtype,
                p.layout_desp_input.layout_type == i_ltype,
                p.layout_desp_weight.layout_type == w_ltype,
                p.layout_desp_output.layout_type == o_ltype,
                p.layout_desp_input.interleave == i_interleave,
                p.layout_desp_weight.interleave == w_interleave,
                p.layout_desp_output.interleave == o_interleave,
                p.ts[0] == ts[0] and p.ts[1] == ts[1] and p.ts[2] == ts[2],
                p.wts[0] == wts[0] and p.wts[1] == wts[1]
                and p.wts[2] == wts[2],
                p.num_stage == num_stage,
                p.dtype_acc == dacc,
                p.dtype_comp == dcomp,
                algo == p.algo.value,
            ]
            if all(if_tests):
                found = True
                assert input_.ndim == p.ndim + 2
                assert weight.ndim == p.ndim + 2
                assert output.ndim == p.ndim + 2
                N = input_.shape[0]
                if p.layout_desp_input.is_channel_first():
                    C = input_.shape[1]
                else:
                    C = input_.shape[p.ndim + 1]
                K = weight.shape[0]
                if p.layout_desp_output.is_channel_first():
                    K2 = output.shape[1]
                else:
                    K2 = output.shape[p.ndim + 1]
                assert K == K2
                ksize = [0] * p.ndim
                input_dims = [0] * p.ndim
                output_dims = [0] * p.ndim
                dim_start = 2 if p.layout_desp_weight.is_channel_first() else 1
                for i in range(dim_start, dim_start + p.ndim):
                    ksize[i - dim_start] = weight.shape[i]
                    input_dims[i - dim_start] = input_.shape[i]
                    output_dims[i - dim_start] = output.shape[i]

                output_dims_check_again = ConvProblem.calc_output_dims_python(
                    input_dims, ksize, padding, stride, dilation)
                assert output_dims_check_again == output_dims
                problem = ker.problem.python_ctor(N, C, K, input_dims,
                                                  output_dims, ksize, padding,
                                                  stride, dilation,
                                                  ConvMode.kCrossCorrelation,
                                                  1, 1)
                print(problem.N_, problem.C_, problem.K_, problem.output_dims_)
                inputs = [input_, weight, output]
                input_metas = [input_meta, weight_meta, output_meta]
                input_abcs = [inputs[i] for i in inv_indices]
                input_meta_abcs = [input_metas[i] for i in inv_indices]

                a_ten = input_abcs[0]
                b_ten = input_abcs[1]
                c_ten = input_abcs[2]
                a_meta_ten = input_meta_abcs[0]
                b_meta_ten = input_meta_abcs[1]

                if cudasim.enable_debug():
                    a_ptr = ArrayPtr(p.dtype_a.tv_dtype,
                                     a_ten.size,
                                     external_data=tv.from_numpy(a_ten),
                                     meta_data=tv.from_numpy(a_meta_ten))
                    b_ptr = ArrayPtr(p.dtype_b.tv_dtype,
                                     b_ten.size,
                                     external_data=tv.from_numpy(b_ten),
                                     meta_data=tv.from_numpy(b_meta_ten))
                else:
                    a_ptr = ArrayPtr(p.dtype_a.tv_dtype,
                                     a_ten.size,
                                     external_data=tv.from_numpy(a_ten),
                                     meta_data=tv.Tensor())
                    b_ptr = ArrayPtr(p.dtype_b.tv_dtype,
                                     b_ten.size,
                                     external_data=tv.from_numpy(b_ten),
                                     meta_data=tv.Tensor())

                c_ptr = ArrayPtr(p.dtype_c.tv_dtype,
                                 c_ten.size,
                                 external_data=tv.from_numpy(c_ten))
                params = ker.gemm_params.python_ctor(problem, a_ptr, b_ptr,
                                                     c_ptr, c_ptr, 1.0, 0.0)
                func = partial(ker.conv_kernel_python, params=params)
                blocks = params.grid_dims
                threads = cudasim.Dim3(ker.num_threads, 1, 1)
                return asyncio.run(
                    cudasim.kernel_launch(func, blocks, threads,
                                          ker.smem_size)), blocks, threads
        raise NotImplementedError
