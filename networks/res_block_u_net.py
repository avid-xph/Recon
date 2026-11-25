import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, DepthwiseConv2D, MaxPooling2D, UpSampling2D, Add, \
    BatchNormalization, Input, Activation, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply
from tensorflow.keras.models import Model


def _channel_attention_block(x, kernel_size=3, name="attention_blk"):
    attention_conv = Conv1D(filters=1, kernel_size=kernel_size, strides=1,
                            use_bias=False, padding="same", name="%s_attn_conv" % name)
    #定义了一个一维卷积层 attention_conv，这个卷积层对每个通道的特征进行处理，生成注意力权重。
    #它的输出是一个形状为 (batch_size, channels, 1) 的张量，表示每个通道的重要性。batch_size是同时输入U-Net模型的图像数量，channels是图像的通道数（例如RGB图像有3个通道）。
    #kernel_size 参数定义了卷积核的大小，这里默认为3。它代表了在计算某一通道注意力权重时考虑的邻域范围。比方说，kernel_size = 5意味着在计算第N个图像通道的注意力权重时，会考虑第N-2到N+2共5个通道的信息。
    avg_pool = attention_conv(
        tf.expand_dims(
            GlobalAveragePooling2D(name="%s_attn_avg_pool" % name)(x),#对输入张量x进行全局平均池化，得到每个通道的平均值，形状为 (batch_size, channels)。张量x的形状通常是 (batch_size, height, width, channels)，表示一批图像数据。
            axis=-1#作用是在最后一个维度上增加一个新的维度，使得张量的形状从 (batch_size, channels) 变为 (batch_size, channels, 1)。相当于把二维数组里的每一个位置由一个数变成一个一维数组，数组里只有一个数。
        )
    )  # shape=(b, c, 1)
    max_pool = attention_conv(
        tf.expand_dims(
            GlobalMaxPooling2D(name="%s_attn_max_pool" % name)(x),#对输入张量x进行全局最大池化，得到每个通道的最大值，形状为 (batch_size, channels)。
            axis=-1#作用是在最后一个维度上增加一个新的维度，使得张量的形状从 (batch_size, channels) 变为 (batch_size, channels, 1)。相当于把二维数组里的每一个位置由一个数变成一个一维数组，数组里只有一个数（如果最后一个维度的数值是10，那么表明那个一维数组里有10个数）。
        
        )
    )  # shape=(b, c, 1)
    attention_feature = Activation('sigmoid', name="%s_attn_sigmoid" % name)(avg_pool + max_pool)  # shape=(b, c, 1)，这里将平均池化和最大池化的结果相加，并通过sigmoid激活函数生成注意力权重，范围在0到1之间。相当于一张通道重要性评分表.
    attention_feature = tf.expand_dims(attention_feature, axis=-1)  # shape=(b, c, 1, 1)，维度扩张
    attention_feature = tf.transpose(attention_feature, perm=[0, 2, 3, 1])  # shape=(b, 1, 1, c)
    return attention_feature * x #这里会将注意力权重应用到输入张量x上，调整每个通道的特征表示，从而增强重要特征，抑制不重要特征。这里会自动把（b,1,1,c）广播成（b,h,w,c），然后和x做逐元素相乘.


def _spectral_spatial_fusion_block(x, reduction_ratio=4, name="spectral_spatial_blk"):
    """
    Lightweight spectral-spatial fusion block for hyperspectral reconstruction.
    - Spectral branch: 1x1 conv bottleneck to mix channel dependencies.
    - Spatial branch: depthwise 3x3 conv to provide local spatial gating.
    """
    channel_cnt = x.shape[-1]
    assert channel_cnt is not None, "Channel dimension must be known for spectral-spatial fusion."
    hidden_channel = max(int(channel_cnt) // reduction_ratio, 8)

    spectral = Conv2D(hidden_channel, kernel_size=1, padding='same', activation='gelu',
                      name="%s_spec_conv1" % name)(x)
    spectral = Conv2D(int(channel_cnt), kernel_size=1, padding='same',
                      name="%s_spec_conv2" % name)(spectral)

    spatial = DepthwiseConv2D(kernel_size=3, padding='same', name="%s_spatial_dw" % name)(x)
    spatial = Activation('sigmoid', name="%s_spatial_sigmoid" % name)(spatial)
    spatial_weighted = Multiply(name="%s_spatial_weighted" % name)([x, spatial])

    return Add(name="%s_fused_out" % name)([x, spectral, spatial_weighted])


def get_res_block_u_net(filter_root, depth, output_channel=31, input_size=(256, 256, 3), activation='elu',
                        batch_norm=True, batch_norm_after_activation=False, final_activation='sigmoid', net_num=1,
                        extra_upsampling=False, remove_first_long_connection=False, channel_attention=False,
                        spectral_spatial_fusion=False, spectral_fusion_reduction=4,
                        kernel_initializer='glorot_uniform', final_kernel_initializer='glorot_uniform'):
    """
    Our implementation of Res-UNet

    Args:
        filter_root (int): The filter number in first layer.
        depth (int): Depth of UNet. i.e. the count of down/upsampling layers. the `filter_root` and image height and width must be divisible by `2^depth`.每次下采样和上采样都会使图像的高度和宽度减半或加倍，因此初始图像的高度和宽度必须能够被2的depth次方整除，以确保在经过多次下采样后，图像尺寸仍然是整数。
        output_channel (int, optional): Final output channel number.
        input_size (tuple, optional): Input image resolution. Default is (512, 512, 3).
        activation (str, optional): Activation used in layers. Default is 'elu'.
        batch_norm (bool, optional): Whether to add BatchNormalization in each layer. Default is True.让每一层的输出具有标准正态分布（均值为0，标准差为1），从而加速训练过程并提高模型的稳定性。
        batch_norm_after_activation: The location of BatchNormalization. Default is before the activation (False).
        final_activation (str, optional): Activation of the last layer. Default is 'sigmoid'；
        net_num (int, optional)：Sub-network count. Default is 1 (only one network).
        extra_upsampling: Whether to add an extra upsampling. This operation can make the output image double the size of the input. Default is False.
        remove_first_long_connection: Whether to discard the long residual connection of the outermost layer of ResUNet. Default is False.
        channel_attention: Whether to use channel attention mechanism on long residual connections. Default is False.
        spectral_spatial_fusion: Whether to enable lightweight spectral-spatial fusion blocks. Default is False.
        spectral_fusion_reduction: Channel reduction ratio in spectral branch. Default is 4.
        kernel_initializer: Kernel initializer of convolution layers. Default is 'glorot_uniform'.
        final_kernel_initializer: Kernel initializer of the last layer. Default is 'glorot_uniform'.
    Returns:
        The Res-UNet Keras model.
    """
    assert net_num >= 1, "There should be at least one network."
    print("Network Args:")

    print("activation=", activation)
    print("final_activation=", final_activation)
    print("kernel_initializer=", kernel_initializer)
    print("final_kernel_initializer=", final_kernel_initializer)

    inputs = Input(input_size)
    x = inputs

    # Dictionary for long connections
    long_connection_store = {}

    if net_num == 1 and extra_upsampling:
        x = UpSampling2D(name="first_bilinear_upsampling", interpolation="bilinear")(x)

    for cur_net_num in range(net_num):
        # Downsampling layer (via maxpooling)
        for i in range(depth):
            out_channel = 2 ** i * filter_root
            # Residual/Skip connection
            inner_block_res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                                     kernel_initializer=kernel_initializer,
                                     name="net{}_identity{}_1".format(cur_net_num, i))(x)
            # First Conv2D Block with Conv2D, BN and activation
            conv1 = Conv2D(out_channel, kernel_size=7 if i == 0 else 3, padding='same',
                           kernel_initializer=kernel_initializer,
                           name="net{}_conv{}_1".format(cur_net_num, i))(x)
            if batch_norm and not batch_norm_after_activation:
                conv1 = BatchNormalization(name="net{}_bn{}_1".format(cur_net_num, i))(conv1)
            act1 = Activation(activation, name="net{}_act{}_1".format(cur_net_num, i))(conv1)
            if batch_norm and batch_norm_after_activation:
                act1 = BatchNormalization(name="net{}_bn{}_1".format(cur_net_num, i))(act1)
            # Second Conv2D block with Conv2D and BN only
            conv2 = Conv2D(out_channel, kernel_size=3, padding='same',
                           kernel_initializer=kernel_initializer,
                           name="net{}_conv{}_2".format(cur_net_num, i))(act1)
            if batch_norm and not batch_norm_after_activation:
                conv2 = BatchNormalization(name="net{}_bn{}_2".format(cur_net_num, i))(conv2)
            inner_block_res_added = Add(name="net{}_add{}_1".format(cur_net_num, i))([inner_block_res, conv2])
            act2 = Activation(activation, name="net{}_act{}_2".format(cur_net_num, i))(inner_block_res_added)
            if batch_norm and batch_norm_after_activation:
                act2 = BatchNormalization(name="net{}_bn{}_2".format(cur_net_num, i))(act2)
            if spectral_spatial_fusion:
                act2 = _spectral_spatial_fusion_block(
                    act2,
                    reduction_ratio=spectral_fusion_reduction,
                    name="net{}_enc{}_ssf".format(cur_net_num, i)
                )
            # Max pooling
            if i < depth - 1:
                if remove_first_long_connection:
                    if i > 0:
                        long_connection_store[str(i)] = act2
                else:
                    long_connection_store[str(i)] = act2
                x = MaxPooling2D(padding='same', name="net{}_maxpooling{}_1".format(cur_net_num, i))(act2)
            else:
                x = act2

        for i in range(depth - 2, -1, -1):
            # Upsampling layers
            out_channel = 2 ** i * filter_root

            up1 = UpSampling2D(name="net{}_up_sampling{}_1".format(cur_net_num, i))(x)
            up_conv1 = Conv2D(out_channel, kernel_size=1, activation=activation, padding='same',
                              kernel_initializer=kernel_initializer,
                              name="net{}_up_conv{}_1".format(cur_net_num, i))(up1)

            if remove_first_long_connection and i == 0:
                # Dispose the first long connection if `remove_first_long_connection` is True.
                up_long_connection_concat = up_conv1
            else:
                long_connection = long_connection_store[str(i)]
                if channel_attention and i < depth-3:
                    # Add channel attention on the long connection if `channel_attention` is True.
                    attention_long_connection = _channel_attention_block(long_connection,
                                                                         name="net{}_long_conn{}".format(cur_net_num,
                                                                                                         i))
                    long_connection = Add(name="net{}_long_conn{}_attn_res_add".format(cur_net_num, i))(
                        [long_connection, attention_long_connection])
                #  Concatenate the long connection
                up_long_connection_concat = Concatenate(axis=-1, name="net{}_up_concat{}_1".format(cur_net_num, i))(
                    [up_conv1, long_connection])

            #  Convolutions
            up_conv2 = Conv2D(out_channel, kernel_size=3, padding='same', kernel_initializer=kernel_initializer,
                              name="net{}_up_conv{}_2".format(cur_net_num, i))(up_long_connection_concat)
            if batch_norm and not batch_norm_after_activation:
                up_conv2 = BatchNormalization(name="net{}_up_bn{}_2".format(cur_net_num, i))(up_conv2)

            up_act1 = Activation(activation, name="net{}_up_act{}_2".format(cur_net_num, i))(up_conv2)

            if batch_norm and batch_norm_after_activation:
                up_act1 = BatchNormalization(name="net{}_up_bn{}_2".format(cur_net_num, i))(up_act1)

            up_conv3 = Conv2D(out_channel, kernel_size=3, padding='same', kernel_initializer=kernel_initializer,
                              name="net{}_up_conv{}_3".format(cur_net_num, i))(up_act1)

            if batch_norm and not batch_norm_after_activation:
                up_conv3 = BatchNormalization(name="net{}_up_bn{}_3".format(cur_net_num, i))(up_conv3)

            # Residual/Skip connection
            inner_block_res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False,
                                     kernel_initializer=kernel_initializer,
                                     name="net{}_up_identity{}_1".format(cur_net_num, i))(up_long_connection_concat)

            inner_block_res_added = Add(name="net{}_up_concat{}_res".format(cur_net_num, i))(
                [inner_block_res, up_conv3])

            x = Activation(activation, name="net{}_up_act{}_res".format(cur_net_num, i))(inner_block_res_added)
            if batch_norm and batch_norm_after_activation:
                x = BatchNormalization(name="net{}_up_bn{}_3".format(cur_net_num, i))(x)
            if spectral_spatial_fusion:
                x = _spectral_spatial_fusion_block(
                    x,
                    reduction_ratio=spectral_fusion_reduction,
                    name="net{}_dec{}_ssf".format(cur_net_num, i)
                )

        if net_num > 1 and cur_net_num == 0 and extra_upsampling:
            x = UpSampling2D(name="middle_res_up_sampling", interpolation="bilinear")(x)

    # Final convolution
    output = Conv2D(output_channel, 1, padding='same', activation=final_activation,
                    kernel_initializer=final_kernel_initializer, name='output')(x)
    model = Model(inputs, outputs=output, name='res-block-u-net')
    # Regularization (L2)
    for layer in model.layers:
        if "kernel_regularizer" in layer.__dict__:
            layer.kernel_regularizer = tf.keras.regularizers.l2(l2=0.0001)
    return model
