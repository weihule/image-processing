import os
import json
import torch
from pycocotools.cocoeval import COCOeval


def validate(val_dataset, model, decoder):
    # model = model.module
    model.eval()
    with torch.no_grad():
        all_eval_result = evaluate_coco(val_dataset, model, decoder)


def evaluate_coco(val_dataset, model, decoder):
    results, image_ids = list(), list()
    for index in range(len(val_dataset)):
        datas = val_dataset[index]
        scale = datas['scale']
        val_inputs = torch.from_numpy(datas['img']).cuda().permute(2, 0, 1).unsqueeze(dim=0)
        cls_heads, reg_heads, batch_anchors = model(val_inputs)

        # for pred_cls_head, pred_reg_head, pred_batch_anchor in zip(
        #         cls_heads, reg_heads, batch_anchors
        # ):
        #     print(pred_cls_head.shape, pred_reg_head.shape, pred_batch_anchor.shape)
        scores, classes, boxes = decoder(cls_heads, reg_heads, batch_anchors)
        scores, classes, boxes = scores.cpu(), classes.cpu(), boxes.cpu()
        boxes /= scale

        # make sure decode batch_size is 1
        # scores shape: [1, max_detection_num]
        # classes shape: [1, max_detection_num]
        # boxes shape: [1, max_detection_num, 4]
        if scores.shape[0] != 1:
            raise ValueError('batch_size num expected 1, but got {}'.format(scores.shape[0]))
        scores = scores.squeeze(0)
        classes = classes.squeeze(0)
        boxes = boxes.squeeze(0)

        # for coco_eval,we need [x_min,y_min,w,h] format pred boxes
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        for object_score, object_class, object_box in zip(scores, classes, boxes):
            object_score = float(object_score)
            object_class = float(object_class)
            object_box = object_box.tolist()

            if object_class == -1:
                continue

            image_result = {
                'image_id': val_dataset.image_ids[index],
                'category_id': val_dataset.find_category_id_from_coco_label(object_class),
                'score': object_score,
                'bbox': object_box
            }
            results.append(image_result)
        # 每张图片记录一次
        image_ids.append(val_dataset.image_ids[index])

        print('{}/{}'.format(index, len(val_dataset)))

    if len(results) == 0:
        print('No target detected in test set images')
        return

    json.dump(results, open('{}_box_results.json'.format(val_dataset.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco = val_dataset.coco
    coco_pred = coco.loadRes('{}_box_results.json'.format(val_dataset.set_name))
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    all_eval_result = coco_eval.stats

    return all_eval_result