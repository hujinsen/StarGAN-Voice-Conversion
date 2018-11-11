import tensorflow as tf


def gated_linear_layer(inputs, gates, name=None):

    activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)

    return activation


def instance_norm_layer(inputs, epsilon=1e-05, activation_fn=None, name=None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs=inputs, center=True, scale=True, epsilon=epsilon, activation_fn=activation_fn, scope=name)

    return instance_norm_layer


def conv1d_layer(inputs, filters, kernel_size, strides=1, padding='same', activation=None, kernel_initializer=None, name=None):

    conv_layer = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def conv2d_layer(inputs, filters, kernel_size, strides, padding: list = None, activation=None, kernel_initializer=None, name=None):

    p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    out = tf.pad(inputs, p, name=name + 'conv2d_pad')

    conv_layer = tf.layers.conv2d(
        inputs=out,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='valid',
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def residual1d_block(inputs, filters=1024, kernel_size=3, strides=1, name_prefix='residule_block_'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs=h1_glu, filters=filters // 2, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs=h2, activation_fn=None, name=name_prefix + 'h2_norm')

    h3 = inputs + h2_norm

    return h3


def downsample1d_block(inputs, filters, kernel_size, strides, name_prefix='downsample1d_block_'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def downsample2d_block(inputs, filters, kernel_size, strides, padding: list = None, name_prefix='downsample2d_block_'):

    h1 = conv2d_layer(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample1d_block(inputs, filters, kernel_size, strides, shuffle_size=2, name_prefix='upsample1d_block_'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample2d_block(inputs, filters, kernel_size, strides, name_prefix='upsample2d_block_'):

    # t1=tf.layers.Conv2DTranspose(filters,kernel_size,strides, padding='same',name=name_prefix+'conv1')(inputs)
    # t1 = tf.layers.batch_normalization()

    t1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
    # t2 = tf.keras.layers.BatchNormalization()(t1)
    t2 = tf.contrib.layers.instance_norm(t1, scope=name_prefix + 'instance1')

    x1_gates = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)

    # x1_norm_gates = tf.keras.layers.BatchNormalization()(x1_gates)
    x1_norm_gates = tf.contrib.layers.instance_norm(x1_gates, scope=name_prefix + 'instance2')
    x1_glu = gated_linear_layer(t2, x1_norm_gates)

    return x1_glu


def pixel_shuffler(inputs, shuffle_size=2, name=None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow, oc], name=name)

    return outputs


def generator_gatedcnn(inputs, speaker_id=None, reuse=False, scope_name='generator_gatedcnn'):
    #input shape [batchsize, h, w, c]
    #speaker_id [batchsize, one_hot_vector]
    #one_hot_vector：[0,1,0,0]
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        #downsample
        d1 = downsample2d_block(inputs, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4], name_prefix='down_1')
        print(f'd1: {d1.shape.as_list()}')

        d2 = downsample2d_block(d1, filters=64, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3], name_prefix='down_2')
        print(f'd2: {d2.shape.as_list()}')

        d3 = downsample2d_block(d2, filters=128, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3], name_prefix='down_3')
        print(f'd3: {d3.shape.as_list()}')

        d4 = downsample2d_block(d3, filters=64, kernel_size=[3, 5], strides=[1, 1], padding=[1, 2], name_prefix='down_4')
        print(f'd4: {d4.shape.as_list()}')
        d5 = downsample2d_block(d4, filters=5, kernel_size=[9, 5], strides=[9, 1], padding=[1, 2], name_prefix='down_5')

        #upsample
        speaker_id = tf.convert_to_tensor(speaker_id, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape.dims[-1].value]), tf.float32)
        c = tf.tile(c_cast, [1, d5.shape.dims[1].value, d5.shape.dims[2].value, 1])
        print(c.shape.as_list())
        concated = tf.concat([d5, c], axis=-1)
        # print(concated.shape.as_list())

        u1 = upsample2d_block(concated, 64, kernel_size=[9, 5], strides=[9, 1], name_prefix='gen_up_u1')
        print(f'u1.shape :{u1.shape.as_list()}')

        c1 = tf.tile(c_cast, [1, u1.shape.dims[1].value, u1.shape.dims[2].value, 1])
        print(f'c1 shape: {c1.shape}')
        u1_concat = tf.concat([u1, c1], axis=-1)
        print(f'u1_concat.shape :{u1_concat.shape.as_list()}')

        u2 = upsample2d_block(u1_concat, 128, [3, 5], [1, 1], name_prefix='gen_up_u2')
        print(f'u2.shape :{u2.shape.as_list()}')
        c2 = tf.tile(c_cast, [1, u2.shape[1], u2.shape[2], 1])
        u2_concat = tf.concat([u2, c2], axis=-1)

        u3 = upsample2d_block(u2_concat, 64, [4, 8], [2, 2], name_prefix='gen_up_u3')
        print(f'u3.shape :{u3.shape.as_list()}')
        c3 = tf.tile(c_cast, [1, u3.shape[1], u3.shape[2], 1])
        u3_concat = tf.concat([u3, c3], axis=-1)

        u4 = upsample2d_block(u3_concat, 32, [4, 8], [2, 2], name_prefix='gen_up_u4')
        print(f'u4.shape :{u4.shape.as_list()}')
        c4 = tf.tile(c_cast, [1, u4.shape[1], u4.shape[2], 1])
        u4_concat = tf.concat([u4, c4], axis=-1)
        print(f'u4_concat.shape :{u4_concat.shape.as_list()}')

        u5 = tf.layers.Conv2DTranspose(filters=1, kernel_size=[3, 9], strides=[1, 1], padding='same', name='generator_last_deconv')(u4_concat)
        print(f'u5.shape :{u5.shape.as_list()}')

        return u5


def discriminator(inputs, speaker_id, reuse=False, scope_name='discriminator'):

    # inputs has shape [batch_size, height,width, channels]

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        #convert data type to float32
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, inputs.shape[1], inputs.shape[2], 1])

        concated = tf.concat([inputs, c], axis=-1)

        # Downsample
        d1 = downsample2d_block(
            inputs=concated, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4], name_prefix='downsample2d_dis_block1_')
        c1 = tf.tile(c_cast, [1, d1.shape[1], d1.shape[2], 1])
        d1_concat = tf.concat([d1, c1], axis=-1)

        d2 = downsample2d_block(
            inputs=d1_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3], name_prefix='downsample2d_dis_block2_')
        c2 = tf.tile(c_cast, [1, d2.shape[1], d2.shape[2], 1])
        d2_concat = tf.concat([d2, c2], axis=-1)

        d3 = downsample2d_block(
            inputs=d2_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3], name_prefix='downsample2d_dis_block3_')
        c3 = tf.tile(c_cast, [1, d3.shape[1], d3.shape[2], 1])
        d3_concat = tf.concat([d3, c3], axis=-1)

        d4 = downsample2d_block(
            inputs=d3_concat, filters=32, kernel_size=[3, 6], strides=[1, 2], padding=[1, 2], name_prefix='downsample2d_diss_block4_')
        c4 = tf.tile(c_cast, [1, d4.shape[1], d4.shape[2], 1])
        d4_concat = tf.concat([d4, c4], axis=-1)

        c1 = conv2d_layer(d4_concat, filters=1, kernel_size=[36, 5], strides=[36, 1], padding=[0, 1], name='discriminator-last-conv')

        c1_red = tf.reduce_mean(c1, keepdims=True)

        return c1_red


def domain_classifier(inputs, reuse=False, scope_name='classifier'):

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        #   add slice input shape [batchsize, 8, 512, 1]
        #get one slice
        one_slice = inputs[:, 0:8, :, :]

        d1 = tf.layers.conv2d(one_slice, 8, kernel_size=[4, 4], padding='same', name=scope_name + '_conv2d01')
        d1_p = tf.layers.max_pooling2d(d1, [2, 2], strides=[2, 2], name=scope_name + 'p1')
        print(f'domain_classifier_d1: {d1.shape}')
        print(f'domain_classifier_d1_p: {d1_p.shape}')

        d2 = tf.layers.conv2d(d1_p, 16, [4, 4], padding='same', name=scope_name + '_conv2d02')
        d2_p = tf.layers.max_pooling2d(d2, [2, 2], strides=[2, 2], name=scope_name + 'p2')
        print(f'domain_classifier_d12: {d2.shape}')
        print(f'domain_classifier_d2_p: {d2_p.shape}')

        d3 = tf.layers.conv2d(d2_p, 32, [4, 4], padding='same', name=scope_name + '_conv2d03')
        d3_p = tf.layers.max_pooling2d(d3, [2, 2], strides=[2, 2], name=scope_name + 'p3')
        print(f'domain_classifier_d3: {d3.shape}')
        print(f'domain_classifier_d3_p: {d3_p.shape}')

        d4 = tf.layers.conv2d(d3_p, 16, [3, 4], padding='same', name=scope_name + '_conv2d04')
        d4_p = tf.layers.max_pooling2d(d4, [1, 2], strides=[1, 2], name=scope_name + 'p4')
        print(f'domain_classifier_d4: {d4.shape}')
        print(f'domain_classifier_d4_p: {d4_p.shape}')

        d5 = tf.layers.conv2d(d4_p, 4, [1, 4], padding='same', name=scope_name + '_conv2d05')
        d5_p = tf.layers.max_pooling2d(d5, [1, 2], strides=[1, 2], name=scope_name + 'p5')
        print(f'domain_classifier_d5: {d5.shape}')
        print(f'domain_classifier_d5_p: {d5_p.shape}')

        p = tf.keras.layers.GlobalAveragePooling2D()(d5_p)

        o_r = tf.reshape(p, [-1, 1, 1, p.shape.dims[1].value])
        print(f'classifier_output: {o_r.shape}')

        return o_r