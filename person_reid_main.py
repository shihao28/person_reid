import os
import torch
import numpy as np
import logging
import cv2
import sys
from PIL import Image
import argparse
import time
sys.path.append('yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox
from config import cfg as ctl_cfg
from train_ctl_model import CTLModel
from inference.inference_utils import _inference
from datasets.transforms import ReidTransforms
from reid_tracker.reid_tracker import ReidTracker


class PeopleDetect:
    def __init__(
        self, source, detect_every_n_frame=1, device='cuda',
        display=True, save=False, yolov5_weights='weights/yolov5/yolov5s.pt',
        fp16=False, imgsz=640, conf_thres=0.5, iou_thres=0.5,
        ctl_model_config='configs/256_resnet50.yml',
        ctl_weights='weights/ctl/market1501_resnet50_256_128_epoch_120.ckpt',
        instance_count_for_matching=10, max_cosine_dist=0.5,
        maxDisappeared=10, momentum=0.9
        ):

        logging.info("Initializing...")

        # Common
        self.input_type = "video" if source.lower().endswith("mp4") else "image"
        self.source = source
        self.detect_every_n_frame = detect_every_n_frame
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_cuda = True if 'cuda' in device.type else False
        self.display = display
        self.save = save

        # yolov5
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # ctl-based tracker
        self.instance_count_for_matching = instance_count_for_matching

        # Initialization starts here
        # Initialize video reader
        self.cap = cv2.VideoCapture(self.source)
        if self.save:
            # Initialize video writer
            dir_name = os.path.dirname(self.source).replace('input', 'output')
            frame_name = os.path.basename(self.source)
            video_filename = os.path.join(dir_name, frame_name)
            os.makedirs(dir_name, exist_ok=True)
            video_out_fps = self.cap.get(cv2.CAP_PROP_FPS) // self.detect_every_n_frame
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_out_H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_out_W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_out = cv2.VideoWriter(
                video_filename, fourcc, video_out_fps,
                (video_out_W, video_out_H))

        # yolov5
        # Initialize model
        self.model = DetectMultiBackend(yolov5_weights, device=device, dnn=False, data=None, fp16=fp16).eval()
        self.stride, names, pt = self.model.stride, self.model.names, self.model.pt
        # Determine inference image size
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size, s is downsize ratio
        if isinstance(self.imgsz, int):
            self.imgsz = (self.imgsz, self.imgsz)
        # warmup
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        # ctl
        # Initialize
        self.reid_model = CTLModel.load_from_checkpoint(ctl_weights).eval()
        self.reid_model.to(device)
        # image transform
        ctl_cfg.merge_from_file(ctl_model_config)
        transforms_base = ReidTransforms(ctl_cfg)
        self.ctl_val_transforms = transforms_base.build_transforms(is_train=False)

        # Tracker
        self.tracker = ReidTracker(
            maxDisappeared=maxDisappeared,
            n_init=3, max_cosine_dist=max_cosine_dist,
            use_bisoftmax=False, momentum=momentum)

    def _yolo_img_preparation(self, ori_frame):
        frame, _, _ = letterbox(ori_frame, self.imgsz, stride=self.stride, auto=True)
        frame = frame[np.newaxis]
        frame = frame[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        frame = np.ascontiguousarray(frame)  # contiguous
        frame = torch.from_numpy(frame).to(self.model.device)
        frame = frame.half() if self.model.fp16 else frame.float()  # uint8 to fp16/32
        frame /= 255  # 0 - 255 to 0.0 - 1.0

        return frame

    def _ctl_img_preparation(self, ori_frame, pred):
        pred = pred.astype(int)
        obj_instances = []
        for pred_tmp in pred:
            x1, y1, x2, y2 = pred_tmp[:4]

            # Convert tensor to PIL Image
            frame_tmp = Image.fromarray(ori_frame[y1:y2, x1:x2])

            # Val transform
            obj_instance = self.ctl_val_transforms(frame_tmp)

            obj_instances.append(obj_instance)

        # Convert to pytorch tensor
        obj_instances = torch.tensor(np.stack(obj_instances, 0))
        obj_instances = obj_instances.to(self.reid_model.device)

        # Insert empty str so that can use existing inference utility
        obj_instances = (obj_instances, '', '')

        return obj_instances

    def _draw_bbox(self, ori_frame, objects):
        # Draw bbox
        for object_id, bbox in objects.items():
            x1, y1, x2, y2 = bbox.astype(np.int16)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(
                ori_frame, (x1, y1), (x2, y2),
                (0, 0, 255), 2)
            cv2.putText(
                ori_frame, f'ID: {object_id:02d}',
                (int(x1 + 0.02*w), int(y1 + 0.15*h)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return ori_frame

    @torch.no_grad()
    def run(self):
        logging.info("Detecting...")

        start_time = time.time()
        frame_count = 0
        while True:
            # Fetch frame
            _, ori_frame = self.cap.read()
            if ori_frame is None:
                break
            elif frame_count % self.detect_every_n_frame != 0:
                frame_count += 1
                continue

            # yolo inference
            # Image preparation for yolo, Return Tensor
            frame = self._yolo_img_preparation(ori_frame)
            pred = self.model(frame, augment=False, visualize=False)
            # NMS, Return List
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=[0])[0]
            pred[:, :4] = scale_coords(frame.shape[2:], pred[:, :4], ori_frame.shape)
            pred = pred.cpu().numpy()

            # Reid inference
            # Image preparation for ctl
            obj_instances = self._ctl_img_preparation(ori_frame, pred)
            # Get embeddings for current detection
            embedding, _ = _inference(self.reid_model, obj_instances, self.use_cuda)
            embedding = torch.nn.functional.normalize(
                embedding, dim=1, p=2
                )

            # Track
            objects = self.tracker.update(pred, embedding.cpu())

            # Draw bbox
            ori_frame = self._draw_bbox(ori_frame, objects)

            # display
            if self.display:
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', ori_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.save:
                self.video_out.write(ori_frame)

            frame_count += 1

        # Close windows, video reader and writer
        cv2.destroyAllWindows()
        self.cap.release()
        if self.save:
            self.video_out.release()

        # Calculate fps
        stop_time = time.time()
        fps = frame_count / self.detect_every_n_frame / (stop_time - start_time)

        logging.info("Detection ran successfully")
        logging.info(f"FPS: {fps:.2f}")

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for person reidentification")
    parser.add_argument("--source", default="videos/inputs/video.mp4", help="path to input video", type=str)
    parser.add_argument("--detect-every-n-frame", default=8, help="detect every n-th frame", type=int)
    parser.add_argument("--device", default="cuda", help="device for inferencing", type=str)
    parser.add_argument("--display", action="store_false", help="flag this to display output frame while inferencing")
    parser.add_argument("--save", action="store_true", help="flag this to save output video. Video is saved to videos/outputs")
    parser.add_argument("--yolov5-weights", default="weights/yolov5/yolov5s.pt", help="path to yolov5 weight", type=str)
    parser.add_argument("--fp16", action="store_true", help="flag this to run in half-precision mode")
    parser.add_argument("--imgsz", default=640, help="path to config file", type=int)
    parser.add_argument("--conf-thres", default=0.5, help="confidence threshold for detection", type=float)
    parser.add_argument("--iou-thres", default=0.5, help="iou threshold for nms", type=float)
    parser.add_argument("--ctl-model-config", default="configs/256_resnet50.yml", help="path to CTL model config", type=str)
    parser.add_argument("--ctl_weights", default="weights/ctl/market1501_resnet50_256_128_epoch_120.ckpt", help="path to config file", type=str)
    parser.add_argument("--instance-count-for-matching", default=10, help="number of most recent instances used for reid", type=int)
    parser.add_argument("--max-cosine-dist", default=0.5, help="maximum distance allowed for tagging a detection to existing object", type=float)
    parser.add_argument("--maxDisappeared", default=8, help="maximum number of frame allowed for continuous tracking of an id", type=float)
    parser.add_argument("--momentum", default=0.9, help="momentum for embedding aggregation. Set this to -1 will perform simple average over --instance-count-for-matching number of most recent embeddings", type=float)
    args = parser.parse_args()
    PeopleDetect(
        source=args.source,
        detect_every_n_frame=args.detect_every_n_frame,
        device=args.device, display=args.display, save=args.save,
        yolov5_weights=args.yolov5_weights, fp16=args.fp16, imgsz=args.imgsz,
        conf_thres=args.conf_thres, iou_thres=args.iou_thres,
        ctl_model_config=args.ctl_model_config,
        ctl_weights=args.ctl_weights,
        instance_count_for_matching=args.instance_count_for_matching,
        max_cosine_dist=args.max_cosine_dist, maxDisappeared=args.maxDisappeared,
        momentum=args.momentum
    ).run()
