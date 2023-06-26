import collections
from enum import Enum
import torch.nn as nn
import torch

import MinkowskiEngine as ME

import time
def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()

class NormType(Enum):
  BATCH_NORM = 0
  INSTANCE_NORM = 1
  INSTANCE_BATCH_NORM = 2

def get_relu(inplace=True, backend=None):
  if backend == 'torchsparse':
      import torchsparse.nn as spnn
      return spnn.ReLU(inplace)
  elif backend == 'mink':
      import MinkowskiEngine as ME
      return ME.MinkowskiReLU(inplace)
  elif backend == 'spconv':
      return nn.ReLU(inplace)
  else:
      raise Exception(f"{backend} backend not supported")

        
def get_norm(norm_type, n_channels, D, bn_momentum=0.1, eps=1e-5, backend=None):
  if backend == 'torchsparse':
      import torchsparse.nn as spnn
      return spnn.BatchNorm(num_features=n_channels, eps=eps, momentum=bn_momentum)
  elif backend == 'mink':
      import MinkowskiEngine as ME
      return ME.MinkowskiBatchNorm(num_features=n_channels, eps=eps, momentum=bn_momentum)
  elif backend == 'spconv':
      return nn.BatchNorm1d(num_features=n_channels, eps=eps, momentum=bn_momentum)
  else:
      raise Exception(f"{backend} backend not supported")
  
class ConvType(Enum):
  """
  Define the kernel region type
  """
  HYPER_CUBE = 0, 'HYPER_CUBE'
  SPATIAL_HYPER_CUBE = 1, 'SPATIAL_HYPER_CUBE'
  SPATIO_TEMPORAL_HYPER_CUBE = 2, 'SPATIO_TEMPORAL_HYPER_CUBE'
  HYPER_CROSS = 3, 'HYPER_CROSS'
  SPATIAL_HYPER_CROSS = 4, 'SPATIAL_HYPER_CROSS'
  SPATIO_TEMPORAL_HYPER_CROSS = 5, 'SPATIO_TEMPORAL_HYPER_CROSS'
  # SPATIAL_HYPER_CUBE_TEMPORAL_HYPER_CROSS = 6, 'SPATIAL_HYPER_CUBE_TEMPORAL_HYPER_CROSS '

  def __new__(cls, value, name):
    member = object.__new__(cls)
    member._value_ = value
    member.fullname = name
    return member

  def __int__(self):
    return self.value


# Covert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPER_CUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPER_CUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPER_CUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPER_CROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPER_CROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPER_CROSS: ME.RegionType.HYPER_CROSS,
    # ConvType.SPATIAL_HYPER_CUBE_TEMPORAL_HYPER_CROSS: ME.RegionType.HYBRID
}

# int_to_region_type = {m.value: m for m in ME.RegionType}
int_to_region_type = {m: ME.RegionType(m) for m in range(3)}


def convert_region_type(region_type):
  """
  Convert the integer region_type to the corresponding RegionType enum object.
  """
  return int_to_region_type[region_type]


def convert_conv_type(conv_type, kernel_size, D):
  assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
  region_type = conv_to_region_type[conv_type]
  axis_types = None
  if conv_type == ConvType.SPATIAL_HYPER_CUBE:
    # No temporal convolution
    if isinstance(kernel_size, collections.Sequence):
      kernel_size = kernel_size[:3]
    else:
      kernel_size = [
          kernel_size,
      ] * 3
    if D == 4:
      kernel_size.append(1)
  elif conv_type == ConvType.SPATIO_TEMPORAL_HYPER_CUBE:
    # conv_type conversion already handled
    assert D == 4
  elif conv_type == ConvType.HYPER_CUBE:
    # conv_type conversion already handled
    pass
  elif conv_type == ConvType.SPATIAL_HYPER_CROSS:
    if isinstance(kernel_size, collections.Sequence):
      kernel_size = kernel_size[:3]
    else:
      kernel_size = [
          kernel_size,
      ] * 3
    if D == 4:
      kernel_size.append(1)
  elif conv_type == ConvType.HYPER_CROSS:
    # conv_type conversion already handled
    pass
  elif conv_type == ConvType.SPATIO_TEMPORAL_HYPER_CROSS:
    # conv_type conversion already handled
    assert D == 4
  # elif conv_type == ConvType.SPATIAL_HYPER_CUBE_TEMPORAL_HYPER_CROSS:
  #   # Define the CUBIC conv kernel for spatial dims and CROSS conv for temp dim
  #   axis_types = [
  #       ME.RegionType.HYPER_CUBE,
  #   ] * 3
  #   if D == 4:
  #     axis_types.append(ME.RegionType.HYPER_CROSS)
  return region_type, axis_types, kernel_size


def conv(in_planes,
         out_planes,
         kernel_size,
         stride=1,
         dilation=1,
         bias=False,
         D=-1,
         padding=0,
         indice_key=None,
         prune_edge=-1,
         backend=None,
         selected_tune_res=None,
         bitcount_export=False):
  if backend == 'torchsparse':
    import torchsparse.nn as spnn
    return spnn.Conv3d(in_planes, out_planes, kernel_size, stride, dilation, bias, transposed=False, prune_edge=prune_edge)
  elif backend == 'mink':
    import MinkowskiEngine as ME
    return ME.MinkowskiConvolution(in_planes, out_planes, kernel_size, stride, dilation, bias, dimension=D, prune_edge=prune_edge)
  elif backend == 'spconv':
    import spconv.pytorch as spconv
    if stride == 1:
        return spconv.SubMConv3d(in_planes, out_planes, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, prune_edge=prune_edge, bitcount_export=bitcount_export, selected_tune_res=selected_tune_res)#, use_hash='hash' in kmap_mode)
    else:
        return spconv.SparseConv3d(in_planes, out_planes, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, prune_edge=prune_edge, bitcount_export=bitcount_export, selected_tune_res=selected_tune_res)#, use_hash='hash' in kmap_mode)
  else:
      raise Exception(f"{backend} backend not supported")


def conv_tr(in_planes,
            out_planes,
            kernel_size,
            upsample_stride=1,
            dilation=1,
            bias=False,
            D=-1,
            indice_key=None,
            prune_edge=-1,
            backend=None,
            selected_tune_res=None,
            bitcount_export=False):

  if backend == 'torchsparse':
    import torchsparse.nn as spnn
    return spnn.Conv3d(in_planes, out_planes, kernel_size, upsample_stride, dilation, bias, transposed=True, prune_edge=prune_edge)
  elif backend == 'mink':
    import MinkowskiEngine as ME
    return ME.MinkowskiConvolutionTranspose(in_planes, out_planes, kernel_size, upsample_stride, dilation, bias, dimension=D, prune_edge=prune_edge)
  elif backend == 'spconv':
    import spconv.pytorch as spconv
    return spconv.SparseInverseConv3d(in_planes, out_planes, kernel_size, indice_key=indice_key, prune_edge=prune_edge, bias=bias, bitcount_export=bitcount_export, selected_tune_res=selected_tune_res)
  else:
      raise Exception(f"{backend} backend not supported")


def avg_pool(kernel_size,
             stride=1,
             dilation=1,
             conv_type=ConvType.HYPER_CUBE,
             in_coords_key=None,
             D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)

  return ME.MinkowskiAvgPooling(
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      kernel_generator=kernel_generator,
      dimension=D)


def avg_unpool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPER_CUBE, D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)

  return ME.MinkowskiAvgUnpooling(
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      kernel_generator=kernel_generator,
      dimension=D)


def sum_pool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPER_CUBE, D=-1):
  assert D > 0, 'Dimension must be a positive integer'
  region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
      kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D)

  return ME.MinkowskiSumPooling(
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      kernel_generator=kernel_generator,
      dimension=D)
