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

NUM_CLASSES = 2

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
def detect(model, image_bgr, threshold=0.7):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(image_rgb)
    #print(img_tensor)
    with torch.no_grad():
        outputs = model(img_tensor)    
    
    logits = outputs['pred_logits'][0]  
    boxes = outputs['pred_boxes'][0]  
    
    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)

    keep = (scores > threshold)

    final_boxes = box_cxcywh_to_xyxy(boxes[keep])
    final_boxes = final_boxes * torch.tensor([image_bgr.shape[1], image_bgr.shape[0],
                                              image_bgr.shape[1], image_bgr.shape[0]], dtype=torch.float32).cuda()

    return final_boxes.int().cpu(), labels[keep].cpu(), scores[keep].cpu()

# ========= Draw boxes ==========
def draw_boxes(image, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v.item()) for v in box]  
        
        
        name = CLASS_NAMES[label]
        if name=="trunk":
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #text = f'{name}: {score:.2f}'
            #cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
    return image


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
    image_name="image_210.png"
    image="/home/rnil/Documents/model/yolact-all/test_images/Sim-data/"+image_name
    
    
    image_bgr = cv2.imread(image)
    if image_bgr is None:
        print(f"Failed to load image: {args.image}")
        return

    boxes, labels, scores = detect(model, image_bgr, 0.4)
    image_bgr = draw_boxes(image_bgr, boxes, labels, scores)

    #cv2.imshow("Detections", image_bgr)
    
    cv2.imwrite(image_name, image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR ONNX export script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    

