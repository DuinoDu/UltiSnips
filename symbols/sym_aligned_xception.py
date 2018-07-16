import logging


import mxnet as mx

fix_gamma = False
eps = 2e-5
bn_mom = 0.9
mirroring_level = 0
use_global_stats = False


def Separable_conv2d(data,
                     in_channels,
                     out_channels,
                     kernel,
                     pad,
                     stride=(1, 1),
                     dilate=(1, 1),
                     bias=False,
                     bn_out=False,
                     act_out=False,
                     name=None,
                     workspace=512,
                     use_deformable=False):
    # depthwise
    pad = list(pad)
    if dilate[0] > 1:
        assert use_deformable
        pad[0] = ((kernel[0] - 1) * dilate[0] + 1) // 2
    if dilate[1] > 1:
        assert use_deformable
        pad[1] = ((kernel[1] - 1) * dilate[1] + 1) // 2
    if use_deformable:
        assert kernel[0] == kernel[1]
        assert pad[0] == pad[1]
        assert stride[0] == stride[1]
        assert dilate[0] == dilate[1]
        assert dilate[0] > 1
        from sym_common import deformable_conv
        dw_out = deformable_conv(data=data,
                                 num_deformable_group=4,
                                 num_filter=in_channels,
                                 kernel=kernel[0],
                                 pad=pad[0],
                                 stride=stride[0],
                                 dilate=dilate[0],
                                 no_bias=False if bias else True,
                                 num_group=in_channels,
                                 name=name + '_conv2d_depthwise')
    else:
        dw_out = mx.sym.Convolution(data=data,
                                    num_filter=in_channels,
                                    kernel=kernel,
                                    pad=pad,
                                    stride=stride,
                                    dilate=dilate,
                                    no_bias=False if bias else True,
                                    num_group=in_channels,
                                    workspace=workspace,
                                    name=name + '_conv2d_depthwise')
    if bn_out:
        dw_out = mx.sym.BatchNorm(data=dw_out,
                                  fix_gamma=fix_gamma,
                                  eps=eps,
                                  momentum=bn_mom,
                                  use_global_stats=use_global_stats,
                                  attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                                  if mirroring_level >= 2 else {},
                                  name=name + '_conv2d_depthwise_bn')
    if act_out:
        dw_out = mx.sym.Activation(data=dw_out,
                                   act_type='relu',
                                   name=name + '_conv2d_depthwise_relu')
        if mirroring_level >= 1:
            dw_out._set_attr(force_mirroring='True')
    #pointwise
    pw_out = mx.sym.Convolution(data=dw_out,
                                num_filter=out_channels,
                                kernel=(1, 1),
                                stride=(1, 1),
                                pad=(0, 0),
                                num_group=1,
                                no_bias=False if bias else True,
                                workspace=workspace,
                                name=name + '_conv2d_pointwise')
    return pw_out


def xception_residual_norm(data,
                           in_channels,
                           out_channels,
                           kernel=(3, 3),
                           pad=(1, 1),
                           stride=(1, 1),
                           bias=False,
                           bypass_type='norm',  # 'bypass_type: norm or separable'
                           name=None,
                           workspace=512):
    assert stride[0] == stride[1]
    assert stride[0] == 1

    sep1 = mx.sym.Activation(data=data,
                             act_type='relu',
                             name=name + '_sep1_relu')
    if mirroring_level >= 1:
        sep1._set_attr(force_mirroring='True')
    sep1 = Separable_conv2d(data=sep1,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel=kernel,
                            pad=pad,
                            stride=stride,
                            bias=bias,
                            bn_out=True,
                            act_out=False,
                            name=name + '_sep1_conv')
    sep1 = mx.sym.BatchNorm(data=sep1,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            name=name + '_sep1_bn')

    sep2 = mx.sym.Activation(data=sep1,
                             act_type='relu',
                             name=name + '_sep2_relu')
    if mirroring_level >= 1:
        sep2._set_attr(force_mirroring='True')
    sep2 = Separable_conv2d(data=sep2,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel=kernel,
                            pad=pad,
                            stride=stride,
                            bias=bias,
                            bn_out=True,
                            act_out=False,
                            name=name + '_sep2_conv')
    sep2 = mx.sym.BatchNorm(data=sep2,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            name=name + '_sep2_bn')

    sep3 = mx.sym.Activation(data=sep2,
                             act_type='relu',
                             name=name + '_sep3_relu')
    if mirroring_level >= 1:
        sep3._set_attr(force_mirroring='True')
    sep3 = Separable_conv2d(data=sep3,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel=kernel,
                            pad=pad,
                            stride=stride,
                            bias=bias,
                            bn_out=True,
                            act_out=False,
                            name=name + '_sep3_conv')
    sep3 = mx.sym.BatchNorm(data=sep3,
                            fix_gamma=fix_gamma,
                            eps=eps,
                            momentum=bn_mom,
                            use_global_stats=use_global_stats,
                            attr={'force_mirroring': 'True', 'cudnn_off': 'True'}
                            if mirroring_level >= 2 else {},
                            name=name + '_sep3_bn')

    if in_channels == out_channels:
        short_cut = data
    else:
        if bypass_type == 'norm':
            short_cut = mx.sym.Convolution(data=data,
                                           num_filter=out_channels,
                                           kernel=(1, 1),
                                           stride=(1, 1),
                                           pad=(0, 0),
                                           num_group=1,
                                           no_bias=False if bias else True,
                                           workspace=workspace,
                                           name=name + '_conv2d_bypass')
            short_cut = mx.sym.BatchNorm(data=short_cut,
                                         fix_gamma=fix_gamma,
                                         eps=eps,
                                         momentum=bn_mom,
                                         use_global_stats=