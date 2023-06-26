import torch.nn as nn

from models.modules.common import ConvType, NormType, get_norm, conv, get_relu, cuda_time

from MinkowskiEngine import MinkowskiReLU

class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3,
               indice_key=None,
               weight_mask=None,
               prune_edge=-1,
               backend=None,
               selected_tune_res=None,
               bitcount_export=False,
               layer_id=-1,
               subm_time=None):
    super(BasicBlockBase, self).__init__()
    self.backend = backend
    self.subm_time = subm_time
    self.conv1 = conv(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, D=D, indice_key=indice_key, prune_edge=weight_mask[str(layer_id)]['conv1'][prune_edge] if weight_mask is not None else -1, bitcount_export=bitcount_export, selected_tune_res=selected_tune_res[str(layer_id)]['conv1']['kernel'], backend=backend)
    self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum, backend=backend)
    self.conv2 = conv(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        bias=False,
        D=D,
        indice_key=indice_key,
        prune_edge=weight_mask[str(layer_id)]['conv2'][prune_edge] if weight_mask is not None else -1,
        bitcount_export=bitcount_export,
        selected_tune_res=selected_tune_res[str(layer_id)]['conv2']['kernel'],
        backend=backend)
    self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum, backend=backend)
    self.relu = get_relu(backend=backend)
    self.downsample = downsample

  def forward(self, x):
    if self.backend == 'spconv':
      residual = x

      out = self.conv1(x)
      out = out.replace_feature(self.norm1(out.features))
      out = out.replace_feature(self.relu(out.features))

      out = self.conv2(out)
      out = out.replace_feature(self.norm2(out.features))

      if self.downsample is not None:
        residual = self.downsample(x)

      # out += residual
      out = out.replace_feature(out.features+residual.features)
      out = out.replace_feature(self.relu(out.features))

      return out
    else:
      residual = x

      out = self.conv1(x)
      out = self.norm1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.norm2(out)

      if self.downsample is not None:
        residual = self.downsample(x)

      out += residual
      out = self.relu(out)

      return out


class BasicBlock(BasicBlockBase):
  NORM_TYPE = NormType.BATCH_NORM


class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = NormType.INSTANCE_NORM


class BasicBlockINBN(BasicBlockBase):
  NORM_TYPE = NormType.INSTANCE_BATCH_NORM


class BottleneckBase(nn.Module):
  expansion = 4
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPER_CUBE,
               bn_momentum=0.1,
               D=3):
    super(BottleneckBase, self).__init__()
    self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

    self.conv2 = conv(
        planes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

    self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
    self.norm3 = get_norm(self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum)

    self.relu = MinkowskiReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.norm2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.norm3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(BottleneckBase):
  NORM_TYPE = NormType.BATCH_NORM


class BottleneckIN(BottleneckBase):
  NORM_TYPE = NormType.INSTANCE_NORM


class BottleneckINBN(BottleneckBase):
  NORM_TYPE = NormType.INSTANCE_BATCH_NORM