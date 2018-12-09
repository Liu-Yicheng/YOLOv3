#yolo_outputs	[(N,13,13,3,85), (N,26,26,3,85), (N,52,52,3,85)]
#y_true			[(N,13,13,3,85), (N,26,26,3,85), (N,52,52,3,85)]
#anchor			[T, 2]
def binary_crossentropy(target, logits):
	return tf.nn.sigmoid_cross_entropy_with_logits(target, logits)
		
		

def box_IoU(b1, b2):
    """
    Calculer IoU between 2 BBs
    # hoi bi nguoc han tinh left bottom, right top TODO
    :param b1: predicted box, shape=[None, 13, 13, 3, 4], 4: xywh
    :param b2: true box, shape=[None, 13, 13, 3, 4], 4: xywh
    :return: iou: intersection of 2 BBs, tensor, shape=[None, 13, 13, 3, 1] ,1: IoU
    b = tf.cast(b, dtype=tf.float32)
    """
    with tf.name_scope('BB1'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        b1 = tf.expand_dims(b1, -2)  # shape= (13, 13, 3, 1, 4)
        b1_xy = b1[..., :2]  # shape= (13, 13, 3, 1, 2)
        b1_wh = b1[..., 2:4]  # shape= (13, 13, 3, 1, 2)
        b1_wh_half = b1_wh / 2. # shape= (13, 13, 3, 1, 2)
        b1_mins = b1_xy - b1_wh_half  # (13, 13, 3, 1, 2)
        b1_maxes = b1_xy + b1_wh_half  # (13, 13, 3, 1, 2)
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # (13, 13, 3, 1)

    with tf.name_scope('BB2'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        # b2 = tf.expand_dims(b2, -2)
        b2 = tf.expand_dims(b2, 0)  # shape= (1, ?, 4)  # TODO 0?
        b2_xy = b2[..., :2]  #    x,y shape= (1, ?, 2)
        b2_wh = b2[..., 2:4]  #   w,h shape= (1, ?, 2)
        b2_wh_half = b2_wh / 2. # w/2, h/2 shape= (1, ?, 2)
        b2_mins = b2_xy - b2_wh_half  # shape= (1, ?, 2)
        b2_maxes = b2_xy + b2_wh_half  # shape= (1, ?, 2)
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # shape= (1, ?, 1)

    with tf.name_scope('Intersection'):
        """Calculate 2 corners: {left bottom, right top} based on BB1, BB2 and area of this box"""
        # intersect_mins = tf.maximum(b1_mins, b2_mins, name='left_bottom')  # (None, 13, 13, 3, 1, 2)
        #bi_mins:(13, 13, 3, 1, 2)  b2_mins(1, ?, 2)
        intersect_mins = tf.maximum(b1_mins, b2_mins) #((13, 13, 3, ?, 2))
        # intersect_maxes = tf.minimum(b1_maxes, b2_maxes, name='right_top')  #
        intersect_maxes = tf.minimum(b1_maxes, b2_maxes) #((13, 13, 3, ?, 2))
        # intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)  # (None, 13, 13, 3, 1, 2), 2: w,h
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.) #((13, 13, 3, ?, 2))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # ((13, 13, 3, ?))

    IoU = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-IoU')  # (13, 13, 3, ?)

    return IoU
	
	
def yolo_head(feature_maps, anchors, num_classes, input_shape, cal_loss=True)
	"""
    Convert final layer features to bounding box parameters.
    (Features learned by the convolutional layers ---> a classifier/regressor which makes the detection prediction)
    :param feature_maps: the feature maps learned by the convolutional layers
                         3 scale [None, 13, 13, 255] from yolov3 structure anchors:[116, 90], [156, 198], [373, 326]
                                 [None, 26, 26, 255]                               [30, 61], [62, 45], [59, 119]
                                 [None, 52, 52, 255]                               [10, 13], [16, 30], [33, 23]
    :param anchors: 3 anchors for each scale shape=(3,2)
    :param num_classes: 80 for COCO
    :param input_shape: 416,416
    :return: box_xy  [None, 13, 13, 3, 2], 2: x,y center point of BB
             box_wh  [None, 13, 13, 3, 2], 2: w,h
             box_conf  [None, 13, 13, 3, 1], 1: conf
             box_class_pred  [None, 13, 13, 3, 80], 80: prob of each class
    """
	num_anchors = len(anchors)
	anchors_tensor = tf.cast(anchors, dtype=feature_maps)
	anchors_tensor = tf.reshape(anchor_tensor, [1, 1, 1, num_anchors, 2])
	with tf.name_scope("create_grid"):
		grid_shape = tf.shape(feature_maps)[1:3]
		grid_y = tf.range(0, grid_shape[0])
		grid_x = tf.range(0, grid_shape[1])
		grid_y = tf.reshape(grid_y, [-1, 1, 1, 1])#[13, 1, 1, 1]
		grid_x = tf.reshape(grid_x, [1, -1, 1, 1])#[1, 13, 1, 1]
		grid_y = tf.tile(grid_y, [1, grid_shape[1], 1, 1])#[13, 13, 1, 1]
		grid_x = tf.tile(grid_x, [grid_shape[0], 1, 1, 1])#[13, 13, 1, 1]
		grid = tf.concat([grid_x, grid_y], axis=-1)#[13, 13, 1, 2]
		grid = tf.cast(grid, dtype=feature_maps.dtype)
		
	feature_maps_reshape = tf.reshape(feature_maps, 
							[-1, grid_shape[0], grid_shape[1], num_anchors, 5+num_classes])
	
	with tf.name_scope("top_feature_maps"):
		box_xy = tf.sigmoid(feature_maps_reshape[...,:2], name='x_y')
		tf.summary.histogram(box_xy.op.name + '/activations', box_xy)
		box_wh = tf.exp(feature_maps_reshape[..., 2:4], name='w_h')
		tf.summary.histogram(box_wh.op.name + '/activations', box_wh)
		box_confidence = tf.sigmoid(feature_maps_reshape[..., 4:5], name='confidence')  
        tf.summary.histogram(box_confidence.op.name + '/activations', box_confidence)
        box_class_probs = tf.sigmoid(feature_maps_reshape[..., 5:], name='class_probs')  
        tf.summary.histogram(box_class_probs.op.name + '/activations', box_class_probs)
		box_xy = (box_xy + grid) / tf.cast(grid[::-1], feature_maps_reshape.dtype)
		box_wh = box_wh * anchor_tensor / tf.cast(input_shape[::-1], feature_maps_reshape.dtype)
	
	if cal_loss == True:
		return grid, feature_maps_reshape, box_xy, box_wh
	
	return box_xy, box_wh, box_confidence, box_class_probs
	
	
def compute_loss(yolo_outputs, y_true, anchor, num_classes):
	anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, y_true[0].dtype)
	grid_shape = [tf.cast(tf.shape(yolo_outputs[l])[1:3], y_true[0].dtype)
				   for l in range(3)]
	#grid_shape:[[13,13],[26,26],[52,52]]
	loss = 0
	m = tf.shape(yolo_outputs[0])[0]
	mf = tf.cast(m, yolo_outputs[0].dtype)
	
	for l in range(3):
		object_mask = y_true[l][..., 4:5]
		true_class_probs = y_true[l][..., 5:]
		grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], anchors[anchor]
												num_classes, input_shape, cal_loss=True)
		pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
		
		raw_true_xy = y_true[l][..., :2] * grid_shape[l][::-1] - grid
		raw_true_wh = tf.log(y_true[l][..., 2:4])/ anchors[anchor_mask[l]] * input_shape[::-1]
		#raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
		box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
		
		ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
		object_mask_bool = tf.cast(object_mask, 'bool')
		
		def loop_body(b, ignore_mask):
			true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
			iou = box_iou(pred_box[b], true_box)
			best_iou = tf.max(iou, axis=-1)
			ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, true_box.dtype)
			return b + 1, ignore_mask
		
		_, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
		ignore_mask = ignore_mask.stack()
		ignore_mask = tf.expand_dims(ignore_mask, -1)
		
		xy_loss = binary_crossentropy(raw_true_xy, raw_pred[..., 0:2]) * object_mask * box_loss_scale
		wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])
		confidence_loss = object_mask * binary_crossentropy(object_mask, raw_pred[..., 4:5])
                       + (1 - object_mask) * binary_crossentropy(object_mask, raw_pred[..., 4:5]) * ignore_mask
		class_loss = object_mask * binary_crossentropy(true_class_probs, raw_pred[..., 5:])
		xy_loss = tf.sum(xy_loss) / mf
        wh_loss = tf.sum(wh_loss) / mf
        confidence_loss = tf.sum(confidence_loss) / mf
        class_loss = tf.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
	return loss
		

		
	
		