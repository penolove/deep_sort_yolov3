import argparse
import random
import warnings
import time

import numpy as np
from eyewitness.mot.tracker import ObjectTracker
from eyewitness.mot.video import Mp4AsVideoData
from eyewitness.mot.evaluation import VideoTrackedObjects
from eyewitness.mot.visualize_mot import draw_tracking_result
from eyewitness.object_detector import ObjectDetector
from eyewitness.detection_utils import DetectionResult
from eyewitness.config import BoundedBoxObject
from eyewitness.image_id import ImageId
from eyewitness.image_utils import Image

from tools import generate_detections as gdet
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from yolo import YOLO

parser = argparse.ArgumentParser(prog="naive_tracker.py")
parser.add_argument("--input-video", type=str, help="path to the input video")
parser.add_argument("--output-video", type=str, default="output.mp4")

warnings.filterwarnings("ignore")


class YoloV3Detector(ObjectDetector):
    def __init__(self):
        self.core_model = None

    def build(self):
        self.core_model = YOLO()

    def detect(self, image_obj) -> DetectionResult:
        if self.core_model is None:
            self.build()
        boxes = self.core_model.detect_image(image_obj.pil_image_obj)
        detected_objects = []
        for bbox in boxes:
            x1, y1, w, h = bbox
            x2, y2 = (x1 + w, y1 + h)

            detected_objects.append(BoundedBoxObject(x1, y1, x2, y2, "", "", ""))

        image_dict = {
            "image_id": image_obj.image_id,
            "detected_objects": detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result


def bbox2xywh(bbox):
    x1 = bbox.x1
    y1 = bbox.y1
    w = bbox.x2 - bbox.x1
    h = bbox.y2 - bbox.y1
    return [x1, y1, w, h]


class DeepSortTracker(ObjectTracker):
    def __init__(self, detector, model_filename):
        super().__init__()
        self.detector = detector
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        # self.width, self.height = opt.img_size

    def track(self, video_data):
        """
        Parameters
        ----------
        video_data: VideoData
            the video data to be tracked
        Returns
        -------
        video_tracked_result: VideoTrackedObjects
            the tracked video result
        """
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget
        )
        tracker = Tracker(metric)
        tracked_objects = VideoTrackedObjects()

        fps = 0.0
        for idx in range(1, int(video_data.n_frames) + 1):
            t1 = time.time()
            pil_image_obj = video_data[idx]

            image_id = ImageId(channel="demo", timestamp=idx, file_format="jpg")
            image_obj = Image(image_id, pil_image_obj=pil_image_obj)
            detection_result = self.detector.detect(image_obj)

            # xywh
            bboxes = [bbox2xywh(i) for i in detection_result.detected_objects]
            frame = np.array(pil_image_obj)[..., ::-1]
            features = self.encoder(frame, bboxes)

            # score to 1.0 here).
            detections = [
                Detection(bbox, 1.0, feature) for bbox, feature in zip(bboxes, features)
            ]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, self.nms_max_overlap, scores
            )
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x1, y1, x2, y2 = track.to_tlbr()
                tracked_objects[idx].append(
                    BoundedBoxObject(x1, y1, x2, y2, int(track.track_id), float(0), "")
                )
            fps = (fps + (1.0 / (time.time() - t1))) / 2
            print("fps= %f" % (fps))
        return tracked_objects


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


if __name__ == "__main__":
    opt = parser.parse_args()

    detector = YoloV3Detector()
    model_filename = "model_data/mars-small128.pb"
    tracker = DeepSortTracker(detector, model_filename)
    mp4_as_video = Mp4AsVideoData(opt.input_video)

    color_list = get_spaced_colors(100)
    random.shuffle(color_list)
    result = tracker.track(mp4_as_video)

    draw_tracking_result(
        result, color_list, mp4_as_video, output_video_path=opt.output_video,
    )
