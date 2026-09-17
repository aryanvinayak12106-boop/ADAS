"""Detect vehicles in a road image using OpenCV DNN with SSD MobileNet."""

import cv2
import matplotlib.pyplot as plt
import numpy as np


CONFIG = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS = "frozen_inference_graph.pb"
IMAGE_PATH = "vehicle.jpg"

CLASS_NAMES = [
	"background", "person", "bicycle", "car", "motorcycle",
	"airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter",
	"bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase",
	"frisbee", "skis", "snowboard", "sports ball", "kite",
	"baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork",
	"knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush",
]
VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck"}


def main() -> None:
	net = cv2.dnn_DetectionModel(WEIGHTS, CONFIG)
	net.setInputSize(320, 320)
	net.setInputScale(1.0 / 127.5)
	net.setInputMean((127.5, 127.5, 127.5))
	net.setInputSwapRB(True)

	image = cv2.imread(IMAGE_PATH)
	if image is None:
		raise FileNotFoundError(f"{IMAGE_PATH} not found")

	output = image.copy()
	class_ids, scores, boxes = net.detect(
		image, confThreshold=0.45, nmsThreshold=0.40
	)
	class_ids = np.array(class_ids).flatten()
	scores = np.array(scores).flatten()

	for class_id, score, box in zip(class_ids, scores, boxes):
		class_id = int(class_id)
		if class_id >= len(CLASS_NAMES):
			continue
		name = CLASS_NAMES[class_id]
		if name not in VEHICLE_NAMES:
			continue

		x, y, width, height = box
		cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)
		cv2.putText(
			output, f"{name} {score:.2f}", (x, max(y - 10, 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
		)

	output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
	plt.figure(figsize=(14, 8))
	plt.imshow(output_rgb)
	plt.title("Vehicle Detection using SSD MobileNet")
	plt.axis("off")
	plt.show()


if __name__ == "__main__":
	main()