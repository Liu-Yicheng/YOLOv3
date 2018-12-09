class YOLOv3(object):
    """Structure of reseau neural YOLO3"""
    def __init__(self, x, num_classes, trainable=True):
        """
        Create the graph ofthe YOLOv3 model
        :param x: Placeholder for the input tensor: (normalised image (416, 416, 3)/255.)
        :param num_classes: Number of classes in the dataset
               if it isn't in the same folder as this code
        """
        self.X = x
        self.NUM_CLASSES = num_classes
        self.train=trainable
       

    def feature_extractor(self):
        """
        Create the network graph
        :return: feature maps 5+80 in 3 grid (13,13), (26,26), (52, 52)
        """
        with tf.name_scope("Features"):
            conv_1 = self.conv2d(self.X, 1)
            # Downsample#############################################
            conv_2 = self.conv2d(conv_1, 2, stride=2)

            conv_3 = self.conv2d(conv_2, 3)
            conv_4 = self.conv2d(conv_3, 4)
            resn_1 = self.resnet(conv_2, conv_4, 1)
            # Downsample#############################################
            conv_5 = self.conv2d(resn_1, 5, stride=2)

            conv_6 = self.conv2d(conv_5, 6)
            conv_7 = self.conv2d(conv_6, 7)
            resn_2 = self.resnet(conv_5, conv_7, 2)

            conv_8 = self.conv2d(resn_2, 8)
            conv_9 = self.conv2d(conv_8, 9)
            resn_3 = self.resnet(resn_2, conv_9, 3)
            # Downsample#############################################
            conv_10 = self.conv2d(resn_3, 10, stride=2)

            conv_11 = self.conv2d(conv_10, 11)
            conv_12 = self.conv2d(conv_11, 12)
            resn_4 = self.resnet(conv_10, conv_12, 4)

            conv_13 = self.conv2d(resn_4, 13)
            conv_14 = self.conv2d(conv_13, 14)
            resn_5 = self.resnet(resn_4, conv_14, 5)

            conv_15 = self.conv2d(resn_5, 15)
            conv_16 = self.conv2d(conv_15, 16)
            resn_6 = self.resnet(resn_5, conv_16, 6)

            conv_17 = self.conv2d(resn_6, 17)
            conv_18 = self.conv2d(conv_17, 18)
            resn_7 = self.resnet(resn_6, conv_18, 7)

            conv_19 = self.conv2d(resn_7, 19)
            conv_20 = self.conv2d(conv_19, 20)
            resn_8 = self.resnet(resn_7, conv_20, 8)

            conv_21 = self.conv2d(resn_8, 21)
            conv_22 = self.conv2d(conv_21, 22)
            resn_9 = self.resnet(resn_8, conv_22, 9)

            conv_23 = self.conv2d(resn_9, 23)
            conv_24 = self.conv2d(conv_23, 24)
            resn_10 = self.resnet(resn_9, conv_24, 10)

            conv_25 = self.conv2d(resn_10, 25)
            conv_26 = self.conv2d(conv_25, 26)
            resn_11 = self.resnet(resn_10, conv_26, 11)
            # Downsample#############################################
            conv_27 = self.conv2d(resn_11, 27, stride=2)

            conv_28 = self.conv2d(conv_27, 28)
            conv_29 = self.conv2d(conv_28, 29)
            resn_12 = self.resnet(conv_27, conv_29, 12)

            conv_30 = self.conv2d(resn_12, 30)
            conv_31 = self.conv2d(conv_30, 31)
            resn_13 = self.resnet(resn_12, conv_31, 13)

            conv_32 = self.conv2d(resn_13, 32)
            conv_33 = self.conv2d(conv_32, 33)
            resn_14 = self.resnet(resn_13, conv_33, 14)

            conv_34 = self.conv2d(resn_14, 34)
            conv_35 = self.conv2d(conv_34, 35)
            resn_15 = self.resnet(resn_14, conv_35, 15)

            conv_36 = self.conv2d(resn_15, 36)
            conv_37 = self.conv2d(conv_36, 37)
            resn_16 = self.resnet(resn_15, conv_37, 16)

            conv_38 = self.conv2d(resn_16, 38)
            conv_39 = self.conv2d(conv_38, 39)
            resn_17 = self.resnet(resn_16, conv_39, 17)

            conv_40 = self.conv2d(resn_17, 40)
            conv_41 = self.conv2d(conv_40, 41)
            resn_18 = self.resnet(resn_17, conv_41, 18)

            conv_42 = self.conv2d(resn_18, 42)
            conv_43 = self.conv2d(conv_42, 43)
            resn_19 = self.resnet(resn_18, conv_43, 19)
            # Downsample##############################################
            conv_44 = self.conv2d(resn_19, 44, stride=2)

            conv_45 = self.conv2d(conv_44, 45)
            conv_46 = self.conv2d(conv_45, 46)
            resn_20 = self.resnet(conv_44, conv_46, 20)

            conv_47 = self.conv2d(resn_20, 47)
            conv_48 = self.conv2d(conv_47, 48)
            resn_21 = self.resnet(resn_20, conv_48, 21)

            conv_49 = self.conv2d(resn_21, 49)
            conv_50 = self.conv2d(conv_49, 50)
            resn_22 = self.resnet(resn_21, conv_50, 22)

            conv_51 = self.conv2d(resn_22, 51)
            conv_52 = self.conv2d(conv_51, 52)
            resn_23 = self.resnet(resn_22, conv_52, 23)  # [None, 13,13,1024]
            ##########################################################
        with tf.name_scope('SCALE'):
            with tf.name_scope('scale_1'):
                conv_53 = self.conv2d(resn_23, 53)
                conv_54 = self.conv2d(conv_53, 54)
                conv_55 = self.conv2d(conv_54, 55)  # [None,14,14,512]
                conv_56 = self.conv2d(conv_55, 56)
                conv_57 = self.conv2d(conv_56, 57)
                conv_58 = self.conv2d(conv_57, 58)  # [None,13 ,13,1024]
                conv_59 = self.conv2d(conv_58, 59, batch_norm_and_activation=False, trainable=self.train)
                # [yolo layer] 6,7,8 # 82  --->predict    scale:1, stride:32, detecting large objects => mask: 6,7,8
                # 13x13x255, 255=3*(80+1+4)
            with tf.name_scope('scale_2'):
                route_1 = self.route1(conv_57, name="route_1")
                conv_60 = self.conv2d(route_1, 60)
                upsam_1 = self.upsample(conv_60, 2, name="upsample_1")
                route_2 = self.route2(upsam_1, resn_19, name="route_2")
                conv_61 = self.conv2d(route_2, 61)
                conv_62 = self.conv2d(conv_61, 62)
                conv_63 = self.conv2d(conv_62, 63)
                conv_64 = self.conv2d(conv_63, 64)
                conv_65 = self.conv2d(conv_64, 65)
                conv_66 = self.conv2d(conv_65, 66)
                conv_67 = self.conv2d(conv_66, 67, batch_norm_and_activation=False, trainable=self.train)
                # [yolo layer] 3,4,5 # 94  --->predict   scale:2, stride:16, detecting medium objects => mask: 3,4,5
                # 26x26x255, 255=3*(80+1+4)
            with tf.name_scope('scale_3'):
                route_3 = self.route1(conv_65, name="route_3")
                conv_68 = self.conv2d(route_3, 68)
                upsam_2 = self.upsample(conv_68, 2, name="upsample_2")
                route_4 = self.route2(upsam_2, resn_11, name="route_4")
                conv_69 = self.conv2d(route_4, 69)
                conv_70 = self.conv2d(conv_69, 70)
                conv_71 = self.conv2d(conv_70, 71)
                conv_72 = self.conv2d(conv_71, 72)
                conv_73 = self.conv2d(conv_72, 73)
                conv_74 = self.conv2d(conv_73, 74)
                conv_75 = self.conv2d(conv_74, 75, batch_norm_and_activation=False, trainable=self.train)
                # [yolo layer] 0,1,2 # 106 --predict scale:3, stride:8, detecting the smaller objects => mask: 0,1,2
                # 52x52x255, 255=3*(80+1+4)
                # Bounding Box:  YOLOv2: 13x13x5
                #                YOLOv3: 13x13x3x3, 3 for each scale

        return conv_59, conv_67, conv_75

    def conv2d(self, inputs, idx, stride=1, batch_norm_and_activation=True, trainable=False, phase_train=False):
        """
        Convolutional layer
        :param inputs:
        :param idx: conv number
        :param stride:
        :param name:
        :param batch_norm_and_activation:
        :return:
        """
        name_conv = 'conv_' + str(idx)
        name_w = 'weights' + str(idx)
        name_b = 'biases' + str(idx)
        name_mean = 'moving_mean' + str(idx)
        name_vari = 'moving_variance' + str(idx)
        name_beta = 'beta' + str(idx)
        name_gam = 'gamma' + str(idx)
        # tous = True
        tous = False
        # tous = self.ST
        with tf.variable_scope(name_conv):
            if trainable == True:
                # we will initialize weights by a Gaussian distribution with mean 0 and variance 1/sqrt(n)
                if idx == 59:
                    weights = tf.Variable(
                        np.random.normal(size=[1, 1, 1024, 3 * (self.NUM_CLASSES + 1 + 4)], loc=0.0, scale=0.01),
                        trainable=True,
                        dtype=np.float32, name="weights")
                elif idx == 67:
                    weights = tf.Variable(
                        np.random.normal(size=[1, 1, 512, 3 * (self.NUM_CLASSES + 1 + 4)], loc=0.0, scale=0.01),
                        trainable=True,
                        dtype=np.float32, name="weights")
                else:
                    weights = tf.Variable(
                        np.random.normal(size=[1, 1, 256, 3 * (self.NUM_CLASSES + 1 + 4)], loc=0.0, scale=0.01),
                        trainable=True,
                        dtype=np.float32, name="weights")
            else:
                weights = tf.Variable(W(idx), trainable=tous, dtype=tf.float32, name="weights")
            tf.summary.histogram(name_w, weights)  # add summary

            if stride == 2:
                paddings = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
                inputs_pad = tf.pad(inputs, paddings, "CONSTANT")
                conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID', name="nn_conv")
            else:
                conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME', name="conv")

            if batch_norm_and_activation:  # TODO
                # conv_1 ---> conv_75 EXCEPT conv_59, conv_67, conv_75
                with tf.variable_scope('BatchNorm'):
                    variance_epsilon = tf.constant(0.0001, name="epsilon")  # small float number to avoid dividing by 

                    moving_mean, moving_variance, beta, gamma = B(idx)
                    moving_mean = tf.Variable(moving_mean, trainable=tous, dtype=tf.float32, name="moving_mean")
                    tf.summary.histogram(name_mean, moving_mean)  # add summary
                    moving_variance = tf.Variable(moving_variance, trainable=tous, dtype=tf.float32, name="moving_variance")
                    tf.summary.histogram(name_vari, moving_variance)  # add summary
                    beta = tf.Variable(beta, trainable=tous, dtype=tf.float32, name="beta")
                    tf.summary.histogram(name_beta, beta)  # add summary
                    gamma = tf.Variable(gamma, trainable=tous, dtype=tf.float32, name="gamma")
                    tf.summary.histogram(name_gam, gamma)  # add summary
                    conv = tf.nn.batch_normalization(conv, moving_mean, moving_variance, beta, gamma,
                                                     variance_epsilon, name='BatchNorm')
                    # conv = tf.nn.batch_normalization(conv, mean, var, beta, gamma,
                    #                                  variance_epsilon, name='BatchNorm')
                with tf.name_scope('Activation'):
                    alpha = tf.constant(0.1, name="alpha")  # Slope of the activation function at x < 0
                    acti = tf.maximum(alpha * conv, conv)
                return acti

            else:
                # for conv_59, conv67, conv_75
                if trainable == True:
                    # biases may be  init =0
                    biases = tf.Variable(
                        np.random.normal(size=[3 * (self.NUM_CLASSES + 1 + 4), ], loc=0.0, scale=0.01),
                        trainable=True,
                        dtype=np.float32, name="biases")
                else:
                    biases = tf.Variable(B(idx), trainable=False, dtype=tf.float32, name="biases")
                tf.summary.histogram(name_b, biases)  # add summary
                conv = tf.add(conv, biases)
                return conv

    @staticmethod
    def route1(inputs, name):
        """
        :param inputs: [5, 500, 416, 3]
        :param name: name in graph
        :return: output = input [5, 500, 416, 3]
        """
        # [route]-4
        with tf.name_scope(name):
            output = inputs
            return output

    @staticmethod
    def route2(input1, input2, name):
        """
        :param input1: [5, 500, 416, 3]
        :param input2: [5, 500, 416, 32]
        :param name: name in graph
        :return: concatenate{input1, input2} [5, 500, 416, 3+32]
                 (nối lại)
        """
        # [route]-1, 36
        # [route]-1, 61
        with tf.name_scope(name):
            output = tf.concat([input1, input2], -1, name='concatenate')  # input1:-1, input2: 61
            return output

    @staticmethod
    def upsample(inputs, size, name):
        """
        :param inputs: (5, 416, 416, 3) par ex
        :param size: 2 par ex
        :param name: name in graph
        :return: Resize images to size using nearest neighbor interpolation. (5, 832, 832, 3) par ex
        """
        with tf.name_scope(name):
            w = tf.shape(inputs)[1]  # 416
            h = tf.shape(inputs)[2]  # 416
            output = tf.image.resize_nearest_neighbor(inputs, [size * w, size * h])
            return output

    @staticmethod
    def resnet(a, b, idx):
        """
        :param a: [5, 500, 416, 32]
        :param b: [5, 500, 416, 32]
        :param name: name in graph
        :return: a+b [5, 500, 416, 32]
        """
        name_res = 'resn' + str(idx)
        with tf.name_scope(name_res):
            resn = a + b
            return resn


def W(number_conv):
    # Charger weights from the pre-trained in COCO
    import h5py
    with h5py.File(path + '/yolo3/model/yolov3.h5', 'r') as f:
        name = 'conv2d_' + str(number_conv)
        w = f['model_weights'][name][name]['kernel:0']
        weights = tf.cast(w, tf.float32)
    return weights


def B(number_conv):
    # Charger biases, bat_norm from the pre-trained in COCO
    import h5py
    with h5py.File(path + '/yolo3/model/yolov3.h5', 'r') as f:
        if (number_conv == 59) or (number_conv == 67) or (number_conv == 75):
            name = 'conv2d_' + str(number_conv)
            b = f['model_weights'][name][name]['bias:0']
            biases = tf.cast(b, tf.float32)
            return biases
        else:
            if 68 <= number_conv <= 74:
                name = 'batch_normalization_' + str(number_conv-2)
                if number_conv==74:
                    print("Finir de charger les poids!")
            elif 66 >= number_conv >= 60:
                name = 'batch_normalization_' + str(number_conv - 1)
            elif 0 < number_conv <= 58:
                name = 'batch_normalization_' + str(number_conv)
            beta = f['model_weights'][name][name]['beta:0']
            beta = tf.cast(beta, tf.float32)

            gamma = f['model_weights'][name][name]['gamma:0']
            gamma = tf.cast(gamma, tf.float32)

            moving_mean = f['model_weights'][name][name]['moving_mean:0']
            moving_mean = tf.cast(moving_mean, tf.float32)

            moving_variance = f['model_weights'][name][name]['moving_variance:0']
            moving_variance = tf.cast(moving_variance, tf.float32)

            return moving_mean, moving_variance, beta, gamma