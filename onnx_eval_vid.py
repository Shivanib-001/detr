# infer_detr_onnx_video.py
import onnxruntime as ort
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.ops import nms

# --- preprocessing same as DETR ---
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

COCO_CLASSES = ['NA', 'Trunk', 'NA']


def rescale_bboxes(out_bbox, size):
    """Rescale DETR output boxes to image size"""
    img_w, img_h = size
    b = out_bbox
    b = np.array([
        (b[0] - b[2]/2) * img_w,
        (b[1] - b[3]/2) * img_h,
        (b[0] + b[2]/2) * img_w,
        (b[1] + b[3]/2) * img_h,
    ])
    return b.astype(int)


def run_video(video_path, onnx_model="mask_trunk1.onnx", score_thresh=0.8, nms_thresh=0.45, use_webcam=False):
    
    sess = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        orig_img = frame.copy()
        h, w = frame.shape[:2]

        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = transform(img_rgb).unsqueeze(0).numpy()

        # Inference
        outputs = sess.run(None, {"images": img_t})
        pred_logits, pred_boxes = outputs[:2]
        pred_masks = outputs[-1] if len(outputs) > 2 else None

        
        probs = torch.softmax(torch.from_numpy(pred_logits[0]), -1)
        scores, labels = probs.max(-1)
        keep = scores > score_thresh

        boxes, scores_keep, labels_keep = [], [], []
        for box, label, score in zip(pred_boxes[0][keep], labels[keep], scores[keep]):
            bbox = rescale_bboxes(box, (w, h))
            boxes.append(bbox)
            scores_keep.append(score.item())
            labels_keep.append(label.item())

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores_keep = torch.tensor(scores_keep)
            labels_keep = torch.tensor(labels_keep)

            # NMS
            keep_idx = nms(boxes, scores_keep, nms_thresh)

            # Draw detections
            for i in keep_idx:
                bbox = boxes[i].int().numpy()
                cls_id = labels_keep[i].item()
                score = scores_keep[i].item()
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)

                if cls_name == "Trunk":
                    cv2.rectangle(orig_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(orig_img, f"{cls_name}:{score:.2f}",
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

            # Overlay masks
            '''if pred_masks is not None:
                masks = torch.from_numpy(pred_masks[0])
                masks = F.interpolate(
                    masks.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(1)

                masks = masks[keep][keep_idx]
                for m in masks:
                    prob_mask = m.sigmoid().cpu().numpy()
                    prob_mask = (prob_mask * 255).astype(np.uint8)
                    colored_mask = cv2.applyColorMap(prob_mask, cv2.COLORMAP_JET)
                    orig_img = cv2.addWeighted(orig_img, 1, colored_mask, 0.4, 0)'''
            
            '''if pred_masks is not None:
                
                masks = pred_masks[0][keep.cpu().numpy()]  
                masks = masks[keep_idx.cpu().numpy()]     

                for m in masks:
                    
                    prob_mask = 1 / (1 + np.exp(-m))

                   
                    prob_mask = cv2.resize(prob_mask, (w, h), interpolation=cv2.INTER_LINEAR)

                    prob_mask = (prob_mask * 255).astype(np.uint8)
                    #alpha_mask = (prob_mask * 0.5).astype(np.float32)[:, :, None] 
                    #orig_img = (orig_img * (1 - alpha_mask) + (alpha_mask * 255)).astype(np.uint8)
                    colored_mask = cv2.applyColorMap(prob_mask, cv2.COLORMAP_JET)
                    orig_img = cv2.addWeighted(orig_img, 1, colored_mask, 0.4, 0)'''


        # Show video
        cv2.imshow("DETR Video Inference", orig_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    run_video("/home/rnil/Documents/model/yolact-all/test_images/test_video1.mp4", "mask_trunk1.onnx")

