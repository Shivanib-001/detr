# python3 demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml   --input bear.jpg   --opts MODEL.WEIGHTS model_0214999.pth


import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from fvcore.common.checkpoint import Checkpointer
from collections import UserDict
import types

from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.utils.comm import is_main_process
from detectron2.utils.visualizer import Visualizer, ColorMode

# Constants
WINDOW_NAME = "COCO Detections"

# ------------------------------------------------------------
# Metadata Handling
# ------------------------------------------------------------

class Metadata(types.SimpleNamespace):
    name: str = "N/A"
    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }

    def __getattr__(self, key):
        if key in self._RENAMED:
            return getattr(self, self._RENAMED[key])
        if len(self.__dict__) > 1:
            raise AttributeError(
                f"Attribute '{key}' not found in metadata '{self.name}'. Available: {list(self.__dict__.keys())}"
            )
        else:
            raise AttributeError(
                f"Attribute '{key}' not found in metadata '{self.name}': metadata is empty."
            )

    def get(self, key, default=None):
        return getattr(self, key, default)

class _MetadataCatalog(UserDict):
    def get(self, name):
        assert name
        if name not in self:
            self[name] = Metadata(name=name)
        return self[name]

MetadataCatalog = _MetadataCatalog()

# ------------------------------------------------------------
# Model Building and Checkpointer
# ------------------------------------------------------------

def build_model(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

class DetectionCheckpointer(Checkpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process() if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_model(self, checkpoint):
        incompatible = super()._load_model(checkpoint)
        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            if k in model_buffers and k in incompatible.missing_keys:
                incompatible.missing_keys.remove(k)
        return incompatible

# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------

class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()

        if cfg.DATASETS.TEST:
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        else:
            self.metadata = Metadata()

        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], f"Invalid format: {self.input_format}"

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

# ------------------------------------------------------------
# Visualization Demo
# ------------------------------------------------------------

class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        predictions = self.predictor(image)
        image = image[:, :, ::-1]  # Convert BGR to RGB for visualizer
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

# ------------------------------------------------------------
# Config and CLI
# ------------------------------------------------------------

def setup_cfg(args):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Inference Demo")
    parser.add_argument("--config-file", default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml", metavar="FILE", help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum score to show predictions")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Additional config options (KEY VALUE pairs)")
    return parser

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------

def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    image = np.asarray(Image.open(args.input).convert("RGB"))
    predictions, vis_output = demo.run_on_image(image)

    cv2.imshow(WINDOW_NAME, vis_output.get_image()[:, :, ::-1]) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
