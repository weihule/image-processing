import math
import numpy as np


class RetinaAnchors:
    def __init__(self,
                 areas=None,
                 ratios=None,
                 scales=None,
                 strides=None):
        if areas is None:
            self.areas = np.array([[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]], dtype=np.float32)
        else:
            self.areas = np.array(areas, dtype=np.float32)

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2], dtype=np.float32)
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = np.array([2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)], dtype=np.float32)
        else:
            self.scales = scales

        if strides is None:
            self.strides = np.array([8, 16, 32, 64, 128], dtype=np.float32)
        else:
            self.strides = strides

    def __call__(self, fpn_feature_sizes):
        """
        generate one image anchors
        """
        one_image_anchors = []
        for index, area in enumerate(self.areas):
            base_anchors = self.generate_base_anchors(area, self.scales,
                                                      self.ratios)
            feature_anchors = self.generate_anchors_on_feature_map(
                base_anchors, fpn_feature_sizes[index], self.strides[index])
            one_image_anchors.append(feature_anchors)

        # if input size:[640,640]
        # one_image_anchors shape:[[80,80,9,4],[40,40,9,4],[20,20,9,4],[10,10,9,4],[5,5,9,4]]
        # per anchor format:[x_min,y_min,x_max,y_max]
        return one_image_anchors

    def generate_base_anchors(self, area, scales, ratios):
        '''
        generate base anchor
        '''
        # get w,h aspect ratio,shape:[9,2]
        aspects = np.array([[[s * math.sqrt(r), s * math.sqrt(1 / r)]
                             for s in scales] for r in ratios],
                           dtype=np.float32).reshape(-1, 2)
        # base anchor for each position on feature map,shape[9,4]
        base_anchors = np.zeros((len(scales) * len(ratios), 4),
                                dtype=np.float32)

        # compute aspect w\h,shape[9,2]
        base_w_h = area * aspects
        base_anchors[:, 2:] += base_w_h

        # base_anchors format: [x_min,y_min,x_max,y_max],center point:[0,0],shape[9,4]
        base_anchors[:, 0] -= base_anchors[:, 2] / 2
        base_anchors[:, 1] -= base_anchors[:, 3] / 2
        base_anchors[:, 2] /= 2
        base_anchors[:, 3] /= 2

        return base_anchors

    def generate_anchors_on_feature_map(self, base_anchors, feature_map_size,
                                        stride):
        '''
        generate one feature map anchors
        '''
        # shifts_x shape:[w],shifts_y shape:[h]
        shifts_x = (np.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (np.arange(0, feature_map_size[1]) + 0.5) * stride

        # shifts shape:[w,h,2] -> [w,h,4] -> [w,h,1,4]
        shifts = np.array([[[shift_x, shift_y] for shift_y in shifts_y]
                           for shift_x in shifts_x],
                          dtype=np.float32)
        shifts = np.expand_dims(np.tile(shifts, (1, 1, 2)), axis=2)

        # base anchors shape:[9,4] -> [1,1,9,4]
        base_anchors = np.expand_dims(base_anchors, axis=0)
        base_anchors = np.expand_dims(base_anchors, axis=0)

        # generate all featrue map anchors on each feature map points
        # featrue map anchors shape:[w,h,9,4] -> [h,w,9,4]
        feature_map_anchors = np.transpose(base_anchors + shifts,
                                           axes=(1, 0, 2, 3))
        feature_map_anchors = np.ascontiguousarray(feature_map_anchors,
                                                   dtype=np.float32)

        # feature_map_anchors format: [h,w,9,4],4:[x_min,y_min,x_max,y_max]
        return feature_map_anchors