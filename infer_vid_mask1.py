# export_onnx.py
import argparse
import torch
import onnx
import util.misc as utils
from models import build_model
from util.misc import NestedTensor
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import time
import torch.nn.functional as F
from torchvision.ops import nms

#python3 infer_vid_mask1.py --resume=output/weights_trunk_mask_c/checkpoint.pth --masks



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='output/weights_trunk_box_a/checkpoint0299.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    
    
    return parser


CLASS_NAMES = ['NA','trunk','NA']


def box_cxcywh_to_xyxy(box):
    x_c, y_c, w, h = box.unbind(-1)
    return torch.stack([x_c - 0.5 * w,
                        y_c - 0.5 * h,
                        x_c + 0.5 * w,
                        y_c + 0.5 * h], dim=-1)

# ========= Preprocess image ==========
def preprocess(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).cuda()


    
# ========= Inference ==========
   
def detect(model, image_bgr, threshold=0.7, iou_threshold=0.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(image_rgb)

    with torch.no_grad():
        outputs = model(img_tensor)

    logits = outputs['pred_logits'][0]
    boxes = outputs['pred_boxes'][0]
    masks = outputs['pred_masks'][0]

    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)

    keep = (scores > threshold)
    final_boxes = box_cxcywh_to_xyxy(boxes[keep])
    final_boxes = final_boxes * torch.tensor([
        image_bgr.shape[1], image_bgr.shape[0],
        image_bgr.shape[1], image_bgr.shape[0]
    ], dtype=torch.float32).cuda()

    final_scores = scores[keep]
    final_labels = labels[keep]

   
    masks = F.interpolate(
        masks.unsqueeze(1),
        size=image_bgr.shape[:2],
        mode="bilinear",
        align_corners=False
    ).squeeze(1)

    final_masks = masks[keep] > 0.5

    #nms 
    keep_indices = nms(final_boxes, final_scores, iou_threshold)

    final_boxes = final_boxes[keep_indices].int().cpu()
    final_labels = final_labels[keep_indices].cpu()
    final_scores = final_scores[keep_indices].cpu()
    final_masks = final_masks[keep_indices].cpu()

    return final_boxes, final_labels, final_scores, final_masks
    

# ========= Draw ==========
def draw_detections(image, boxes, labels, scores, masks):
    overlay = image.copy()
    for box, label, score, mask in zip(boxes, labels, scores, masks):
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)

        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask.numpy()] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        # bounding box
        if CLASS_NAMES[label]=="trunk":
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color.tolist(), 2)
        #text = f'{CLASS_NAMES[label]}: {score:.2f}'
        #cv2.putText(overlay, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)
    return overlay



def load_model(args):
    device = torch.device(args.device)


    model, _, _ = build_model(args)
    
    checkpoint = torch.load(args.resume, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval().cuda()
        
    return model


def main(args):
    model = load_model(args)
    cap = cv2.VideoCapture("/home/rnil/Documents/model/yolact-all/test_images/test_video1.mp4")
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()

        boxes, labels, scores, masks = detect(model, frame, 0.7)

        frame = draw_detections(frame, boxes, labels, scores, masks)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("DETR Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR ONNX export script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)




