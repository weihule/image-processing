import os
import time
import numpy as np
import torch
import random


def test():
    # shifts_x shape:[w],shifts_x shape:[h]
    shifts_x = (np.arange(0, 3))
    shifts_y = (np.arange(0, 5))

    # shifts shape:[w,h,2] -> [w,h,1,2] -> [w,h,3,2] -> [h,w,3,2]
    mesh_shifts_x, mesh_shifts_y = np.meshgrid(shifts_x, shifts_y)
    print(mesh_shifts_x, mesh_shifts_x.shape)


def test_return():
    obj_reg_cls_heads = []
    obj_reg_cls_heads.append(torch.randn((3, 4)))
    obj_reg_cls_heads.append(torch.randn((3, 4)))
    obj_reg_cls_heads.append(torch.randn((3, 4)))

    return [obj_reg_cls_heads]


if __name__ == "__main__":
    # x = np.arange(0, 5)
    # y = np.arange(0, 4)
    # shift_x, shift_y = np.meshgrid(x, y)
    # print(shift_x, shift_y)

    # test()
    start = time.time()
    shifts_x = np.arange(0, 104)
    shifts_y = np.arange(0, 104)

    # shifts shape:[w,h,2] -> [w,h,1,2] -> [w,h,3,2] -> [h,w,3,2]
    # shifts = np.array([[[shift_x, shift_y] for shift_y in shifts_y]
    #                     for shift_x in shifts_x],
    #                     dtype=np.float32)
    # # print(shifts)
    # # print(shifts.shape)
    # shifts = np.expand_dims(shifts, axis=2)
    # shifts = np.tile(shifts, (1, 1, 3, 1))
    # shifts = np.transpose(shifts, axes=(1, 0, 2, 3))
    # test_time1 = time.time() - start
    # print(test_time1)


    mesh_shifts_x, mesh_shifts_y = np.meshgrid(shifts_x, shifts_y)
    shifts2 = []
    for mesh_shift_x, mesh_shift_y in zip(mesh_shifts_x, mesh_shifts_y):
        mesh_shift_x, mesh_shift_y = mesh_shift_x.reshape(-1, 1), mesh_shift_y.reshape(-1, 1)
        sub_temp = np.expand_dims(np.concatenate((mesh_shift_x, mesh_shift_y), axis=-1), axis=0)
        shifts2.append(sub_temp)
    shifts2 = np.concatenate(shifts2, axis=0)
    shifts2 = np.expand_dims(shifts2, axis=2)
    shifts2 = np.tile(shifts2, (1, 1, 3, 1))
    test_time2 = time.time() - start
    # print(test_time2)
 
    # per_level_anchors = np.array([[10, 13],
    #                               [16, 30],
    #                               [33, 23]])
    # all_anchors_wh = np.expand_dims(np.expand_dims(per_level_anchors, 
    #                                 axis=0), axis=0)
    # all_anchors_wh = np.tile(all_anchors_wh, (13, 13, 1, 1))
    # print(all_anchors_wh)
    # print(all_anchors_wh.shape)

    # 假设有4个gt, 12个anchor
    one_img_gt_cls = np.array([1., 0., 3., 1.])    

    # indices取值范围就是 0, 1, 2, 3, 一共有12个 
    indices = np.array([np.random.randint(0, 4) for _ in range(12)])
    overlap = np.random.uniform(low=0, high=1., size=(12))
    one_image_anchors_gt_cls = one_img_gt_cls[indices[overlap > 0.5]]

    gt_boxes = torch.randint(low=1, high=20, size=(5, 4))
    all_strides = torch.tensor([8, 8, 8, 16, 16, 16, 32, 32, 32])
    new_gt_boxes = ((gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2).unsqueeze(1)
    # print(new_gt_boxes, new_gt_boxes.shape)

    # batch_anchors = [torch.randn(4, 10, 10, 3, 5), torch.randn(4, 8, 6, 3, 5), torch.randn(4, 4, 3, 3, 5)]
    # obj_reg_cls_heads = [torch.randn(4, 10, 10, 3, 85), torch.randn(4, 8, 6, 3, 85), torch.randn(4, 4, 3, 3, 85)]
    # feature_hw = []
    # per_layer_prefix_ids = [0, 0, 0]

    # previous_layer_prefix, cur_layer_prefix = 0, 0
    # for layer_idx, (per_level_heads, per_level_anchors) in enumerate(
    #         zip(obj_reg_cls_heads, batch_anchors)):
    #     B, H, W, _, _ = per_level_anchors.shape

    #     for _ in range(3):
    #         feature_hw.append([H, W])
    #     if layer_idx == 0:
    #         for _ in range(3):
    #             per_layer_prefix_ids.append(H * W * 3)
    #         previous_layer_prefix = H * W * 3
    #     elif layer_idx < len(batch_anchors) - 1:
    #         for _ in range(3):
    #             cur_layer_prefix = H * W * 3
    #             per_layer_prefix_ids.append(previous_layer_prefix +
    #                                         cur_layer_prefix)
    #         previous_layer_prefix = previous_layer_prefix + cur_layer_prefix
    # print(feature_hw)
    # print(per_layer_prefix_ids)

    # res = test_return()
    # print(type(res), len(res))

    gt_9_boxes_grid_xy = torch.tensor([[[65., 48.],
         [65., 48.],
         [65., 48.],
         [32., 24.],
         [32., 24.],
         [32., 24.],
         [16., 12.],
         [16., 12.],
         [16., 12.]],

        [[75., 46.],
         [75., 46.],
         [75., 46.],
         [37., 23.],
         [37., 23.],
         [37., 23.],
         [18., 11.],
         [18., 11.],
         [18., 11.]],

        [[72., 46.],
         [72., 46.],
         [72., 46.],
         [36., 23.],
         [36., 23.],
         [36., 23.],
         [18., 11.],
         [18., 11.],
         [18., 11.]],

        [[79., 46.],
         [79., 46.],
         [79., 46.],
         [39., 23.],
         [39., 23.],
         [39., 23.],
         [19., 11.],
         [19., 11.],
         [19., 11.]],

        [[ 9., 52.],
         [ 9., 52.],
         [ 9., 52.],
         [ 4., 26.],
         [ 4., 26.],
         [ 4., 26.],
         [ 2., 13.],
         [ 2., 13.],
         [ 2., 13.]],

        [[23., 47.],
         [23., 47.],
         [23., 47.],
         [11., 23.],
         [11., 23.],
         [11., 23.],
         [ 5., 11.],
         [ 5., 11.],
         [ 5., 11.]],

        [[37., 40.],
         [37., 40.],
         [37., 40.],
         [18., 20.],
         [18., 20.],
         [18., 20.],
         [ 9., 10.],
         [ 9., 10.],
         [ 9., 10.]],

        [[62., 78.],
         [62., 78.],
         [62., 78.],
         [31., 39.],
         [31., 39.],
         [31., 39.],
         [15., 19.],
         [15., 19.],
         [15., 19.]],

        [[34., 71.],
         [34., 71.],
         [34., 71.],
         [17., 35.],
         [17., 35.],
         [17., 35.],
         [ 8., 17.],
         [ 8., 17.],
         [ 8., 17.]]])

    feature_hw = torch.tensor([[80, 80], [80, 80], [80, 80],
                               [40, 40], [40, 40], [40, 40],
                               [20, 20], [20, 20], [20, 20]])

    per_level_num_anchors = 3

    grid_inside_ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    per_layer_prefix_ids = torch.tensor([0, 0, 0, 19200, 19200, 19200, 4800, 4800, 4800])

    print(grid_inside_ids.unsqueeze(0).shape)


    global_ids = ((gt_9_boxes_grid_xy[:, :, 1] * feature_hw[:, 1].unsqueeze(0) +
                gt_9_boxes_grid_xy[:, :, 0]) * per_level_num_anchors +
                grid_inside_ids.unsqueeze(0) + per_layer_prefix_ids.unsqueeze(0)).long()

    print(global_ids)
    
