from models.resnet import ResNetBase, get_norm
from models.modules.common import ConvType, NormType, conv, conv_tr, get_relu, cuda_time
from models.modules.resnet_block import BasicBlock, Bottleneck

import MinkowskiEngine.MinkowskiOps as me
import torchsparse
import torch
import os
import pickle

class Res16UNetBase(ResNetBase):
  BLOCK = None
  PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  INIT_DIM = 32
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.HYPER_CUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    self.backend = config.backend
    print("\033[104m[NANM] Backend is set as", self.backend, "\033[0m")
    super(Res16UNetBase, self).__init__(in_channels, out_channels, config, D)

  def network_initialization(self, in_channels, out_channels, config, D):
    self.selected_tune_res = dict()

    pickle_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "../pickles/")

    if config.fp16:
        filename = pickle_dir+'saved_tune_res/'+config.dataset+'_'+config.model+'_gv100_fp16'+'.pickle'
    else:
        filename = pickle_dir+'saved_tune_res/'+config.dataset+'_'+config.model+'_gv100.pickle'

    if os.path.isfile(filename):
      with open(filename, 'rb') as handle:
        self.selected_tune_res = pickle.load(handle)
    else:
      print("Please generate ", filename)
      exit(0)

    weight_masks = dict()
    filename = pickle_dir+"weight_masks/"+config.dataset+'_'+config.model+'_weight_mask_trainset.pickle'
    if os.path.isfile(filename):
      with open(filename, 'rb') as handle:
        weight_masks = pickle.load(handle)
        print(weight_masks)
    else:
      print("Please generate ", filename)
      exit(0)

    # Setup net_metadata
    dilations = self.DILATIONS
    bn_momentum = config.bn_momentum

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    # Output of the first conv concated to conv6
    self.inplanes = self.INIT_DIM
    self.conv0p1s1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        stride=1,
        dilation=1,
        D=D,
        indice_key="res0",
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['conv0p1s1']['kernel'],
        backend=self.backend)

    self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum, backend=self.backend)

    self.conv1p1s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        D=D,
        indice_key="down1",
        padding=1,
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['conv1p1s2']['kernel'],
        backend=self.backend)
    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum, backend=self.backend)
    self.block1 = self._make_layer(
        self.BLOCK,
        self.PLANES[0],
        self.LAYERS[0],
        dilation=dilations[0],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res1",
        weight_mask=weight_masks['block1'] if 'block1' in weight_masks else None,
        prune_edge=config.prune_edge['0'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block1'],
        backend=self.backend)

    self.conv2p2s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        D=D,
        indice_key="down2",
        padding=1,
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['conv2p2s2']['kernel'],
        backend=self.backend)
    self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum, backend=self.backend)
    self.block2 = self._make_layer(
        self.BLOCK,
        self.PLANES[1],
        self.LAYERS[1],
        dilation=dilations[1],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res2",
        weight_mask=weight_masks['block2'] if 'block2' in weight_masks else None,
        prune_edge=config.prune_edge['1'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block2'],
        backend=self.backend)

    self.conv3p4s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        D=D,
        indice_key="down3",
        padding=1,
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['conv3p4s2']['kernel'],
        backend=self.backend)
    self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum, backend=self.backend)
    self.block3 = self._make_layer(
        self.BLOCK,
        self.PLANES[2],
        self.LAYERS[2],
        dilation=dilations[2],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res3",
        weight_mask=weight_masks['block3'] if 'block3' in weight_masks else None,
        prune_edge=config.prune_edge['2'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block3'],
        backend=self.backend)

    self.conv4p8s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        D=D,
        indice_key="down4",
        padding=1,
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['conv4p8s2']['kernel'],
        backend=self.backend)
    self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum, backend=self.backend)
    self.block4 = self._make_layer(
        self.BLOCK,
        self.PLANES[3],
        self.LAYERS[3],
        dilation=dilations[3],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res4",
        weight_mask=weight_masks['block4'] if 'block4' in weight_masks else None,
        prune_edge=config.prune_edge['3'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block4'],
        backend=self.backend)
    self.convtr4p16s2 = conv_tr(
        self.inplanes,
        self.PLANES[4],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        D=D,
        indice_key="down4",
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['convtr4p16s2']['kernel'],
        backend=self.backend)
    self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum, backend=self.backend)

    self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
    self.block5 = self._make_layer(
        self.BLOCK,
        self.PLANES[4],
        self.LAYERS[4],
        dilation=dilations[4],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res3",
        weight_mask=weight_masks['block5'] if 'block5' in weight_masks else None,
        prune_edge=config.prune_edge['4'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block5'],
        backend=self.backend)
    self.convtr5p8s2 = conv_tr(
        self.inplanes,
        self.PLANES[5],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        D=D,
        indice_key="down3",
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['convtr5p8s2']['kernel'],
        backend=self.backend)
    self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum, backend=self.backend)

    self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
    self.block6 = self._make_layer(
        self.BLOCK,
        self.PLANES[5],
        self.LAYERS[5],
        dilation=dilations[5],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res2",
        weight_mask=weight_masks['block6'] if 'block6' in weight_masks else None,
        prune_edge=config.prune_edge['5'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block6'],
        backend=self.backend)
    self.convtr6p4s2 = conv_tr(
        self.inplanes,
        self.PLANES[6],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        D=D,
        indice_key="down2",
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['convtr6p4s2']['kernel'],
        backend=self.backend)
    self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum, backend=self.backend)

    self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
    self.block7 = self._make_layer(
        self.BLOCK,
        self.PLANES[6],
        self.LAYERS[6],
        dilation=dilations[6],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res1",
        weight_mask=weight_masks['block7'] if 'block7' in weight_masks else None,
        prune_edge=config.prune_edge['6'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block7'],
        backend=self.backend)
    self.convtr7p2s2 = conv_tr(
        self.inplanes,
        self.PLANES[7],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        D=D,
        indice_key="down1",
        prune_edge=-1,
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['convtr7p2s2']['kernel'],
        backend=self.backend)
    self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum, backend=self.backend)

    self.inplanes = self.PLANES[7] + self.INIT_DIM
    self.block8 = self._make_layer(
        self.BLOCK,
        self.PLANES[7],
        self.LAYERS[7],
        dilation=dilations[7],
        norm_type=self.NORM_TYPE,
        bn_momentum=bn_momentum,
        indice_key="res0",
        weight_mask=weight_masks['block8'] if 'block7' in weight_masks else None,
        prune_edge=config.prune_edge['7'],
        bitcount_export=config.bitcount_export,
        selected_tune_res = self.selected_tune_res['block8'],
        backend=self.backend)

    self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D, backend=self.backend)
    
    self.relu =  get_relu(True, self.backend)

  def forward(self, x):
    if self.backend == 'spconv':
      out = self.conv0p1s1(x)
      out = out.replace_feature(self.bn0(out.features))
      out_p1 = out.replace_feature(self.relu(out.features))

      out = self.conv1p1s2(out_p1)
      out = out.replace_feature(self.bn1(out.features))
      out = out.replace_feature(self.relu(out.features))
      out_b1p2 = self.block1(out)

      out = self.conv2p2s2(out_b1p2)
      out = out.replace_feature(self.bn2(out.features))
      out = out.replace_feature(self.relu(out.features))
      out_b2p4 = self.block2(out)

      out = self.conv3p4s2(out_b2p4)
      out = out.replace_feature(self.bn3(out.features))
      out = out.replace_feature(self.relu(out.features))
      out_b3p8 = self.block3(out)

      # pixel_dist=16
      out = self.conv4p8s2(out_b3p8)
      out = out.replace_feature(self.bn4(out.features))
      out = out.replace_feature(self.relu(out.features))
      out = self.block4(out)

      # pixel_dist=8
      out = self.convtr4p16s2(out)
      out = out.replace_feature(self.bntr4(out.features))
      out = out.replace_feature(self.relu(out.features))

      out = out.replace_feature(torch.cat((out.features, out_b3p8.features), dim=-1))
      out = self.block5(out)

      # pixel_dist=4
      out = self.convtr5p8s2(out)
      out = out.replace_feature(self.bntr5(out.features))
      out = out.replace_feature(self.relu(out.features))

      out = out.replace_feature(torch.cat((out.features, out_b2p4.features), dim=-1))
      out = self.block6(out)

      # pixel_dist=2
      out = self.convtr6p4s2(out)
      out = out.replace_feature(self.bntr6(out.features))
      out = out.replace_feature(self.relu(out.features))

      out = out.replace_feature(torch.cat((out.features, out_b1p2.features), dim=-1))
      out = self.block7(out)

      # pixel_dist=1
      out = self.convtr7p2s2(out)
      out = out.replace_feature(self.bntr7(out.features))
      out = out.replace_feature(self.relu(out.features))

      out = out.replace_feature(torch.cat((out.features, out_p1.features), dim=-1))
      out = self.block8(out)
      return self.final(out)
    
    elif self.backend == 'mink':
      out = self.conv0p1s1(x)
      out = self.bn0(out)
      out_p1 = self.relu(out)

      out = self.conv1p1s2(out_p1)
      out = self.bn1(out)
      out = self.relu(out)
      out_b1p2 = self.block1(out)

      out = self.conv2p2s2(out_b1p2)
      out = self.bn2(out)
      out = self.relu(out)
      out_b2p4 = self.block2(out)

      out = self.conv3p4s2(out_b2p4)
      out = self.bn3(out)
      out = self.relu(out)
      out_b3p8 = self.block3(out)

      # pixel_dist=16
      out = self.conv4p8s2(out_b3p8)
      out = self.bn4(out)
      out = self.relu(out)
      out = self.block4(out)

      # pixel_dist=8
      out = self.convtr4p16s2(out)
      out = self.bntr4(out)
      out = self.relu(out)

      out = me.cat(out, out_b3p8)
      out = self.block5(out)

      # pixel_dist=4
      out = self.convtr5p8s2(out)
      out = self.bntr5(out)
      out = self.relu(out)

      out = me.cat(out, out_b2p4)
      out = self.block6(out)

      # pixel_dist=2
      out = self.convtr6p4s2(out)
      out = self.bntr6(out)
      out = self.relu(out)

      out = me.cat(out, out_b1p2)
      out = self.block7(out)

      # pixel_dist=1
      out = self.convtr7p2s2(out)
      out = self.bntr7(out)
      out = self.relu(out)

      out = me.cat(out, out_p1)
      out = self.block8(out)

      return self.final(out)

    elif self.backend == 'torchsparse':
      out = self.conv0p1s1(x)
      out = self.bn0(out)
      out_p1 = self.relu(out)

      out = self.conv1p1s2(out_p1)
      out = self.bn1(out)
      out = self.relu(out)
      out_b1p2 = self.block1(out)

      out = self.conv2p2s2(out_b1p2)
      out = self.bn2(out)
      out = self.relu(out)
      out_b2p4 = self.block2(out)

      out = self.conv3p4s2(out_b2p4)
      out = self.bn3(out)
      out = self.relu(out)
      out_b3p8 = self.block3(out)

      # pixel_dist=16
      out = self.conv4p8s2(out_b3p8)
      out = self.bn4(out)
      out = self.relu(out)
      out = self.block4(out)

      # pixel_dist=8
      out = self.convtr4p16s2(out)
      out = self.bntr4(out)
      out = self.relu(out)

      out = torchsparse.cat((out, out_b3p8))
      out = self.block5(out)

      # pixel_dist=4
      out = self.convtr5p8s2(out)
      out = self.bntr5(out)
      out = self.relu(out)

      out = torchsparse.cat((out, out_b2p4))
      out = self.block6(out)

      # pixel_dist=2
      out = self.convtr6p4s2(out)
      out = self.bntr6(out)
      out = self.relu(out)

      out = torchsparse.cat((out, out_b1p2))
      out = self.block7(out)

      # pixel_dist=1
      out = self.convtr7p2s2(out)
      out = self.bntr7(out)
      out = self.relu(out)

      out = torchsparse.cat((out, out_p1))
      out = self.block8(out)

      return self.final(out)


class Res16UNet14(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

class Res16UNet34(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
  LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
  PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class STRes16UNetBase(Res16UNetBase):
  CONV_TYPE = ConvType.SPATIO_TEMPORAL_HYPER_CROSS

  def __init__(self, in_channels, out_channels, config, D=4, **kwargs):
    super(STRes16UNetBase, self).__init__(in_channels, out_channels, config, D, **kwargs)


class STRes16UNet14(STRes16UNetBase, Res16UNet14):
  pass


class STRes16UNet14A(STRes16UNetBase, Res16UNet14A):
  pass


class STRes16UNet18(STRes16UNetBase, Res16UNet18):
  pass


class STRes16UNet34(STRes16UNetBase, Res16UNet34):
  pass


class STRes16UNet50(STRes16UNetBase, Res16UNet50):
  pass


class STRes16UNet101(STRes16UNetBase, Res16UNet101):
  pass


class STRes16UNet18A(STRes16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class STResTesseract16UNetBase(STRes16UNetBase):
  CONV_TYPE = ConvType.HYPER_CUBE


class STResTesseract16UNet18A(STRes16UNet18A, STResTesseract16UNetBase):
  pass
