"""Detect possible forward collisions in a dash-camera video.

This is a video-analysis aid, not a certified safety system. It uses a
pretrained YOLO model to find vehicles and flags a possible impact when a
vehicle in the driving corridor rapidly grows in the image or disappears
after becoming very large.

Example:
    python 3.py test.mp4 --output collision_analysis.mp4
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck in COCO
MODEL_NAME = "yolov8n.pt"
MIN_CONFIDENCE = 0.35
LARGE_VEHICLE_AREA = 0.12
AREA_GROWTH_RATIO = 1.8
FRAME_HISTORY = 10
EVENT_COOLDOWN_FRAMES = 30


class CollisionDetector:
	"""Track the most threatening vehicle and detect rapid approach patterns."""

	def __init__(self) -> None:
		self.vehicle_history: deque[tuple[int, float]] = deque(maxlen=FRAME_HISTORY)
		self.last_vehicle_seen = -1
		self.last_event_frame = -EVENT_COOLDOWN_FRAMES

	def update(
		self, frame_number: int, vehicle_area: float | None
	) -> tuple[bool, str]:
		if vehicle_area is None:
			if (
				self.vehicle_history
				and frame_number - self.last_vehicle_seen <= 3
				and self.vehicle_history[-1][1] >= LARGE_VEHICLE_AREA
			):
				return self._event(frame_number, "vehicle disappeared after rapid approach")
			return False, "No vehicle detected in driving corridor"

		self.last_vehicle_seen = frame_number
		self.vehicle_history.append((frame_number, vehicle_area))
		if len(self.vehicle_history) < 4:
			return False, "Monitoring"

		oldest_area = min(area for _, area in self.vehicle_history)
		growth_ratio = vehicle_area / max(oldest_area, 0.001)
		if (
			vehicle_area >= LARGE_VEHICLE_AREA
			and growth_ratio >= AREA_GROWTH_RATIO
		):
			return self._event(frame_number, "vehicle rapidly expanded in image")
		return False, "Monitoring"

	def _event(self, frame_number: int, reason: str) -> tuple[bool, str]:
		if frame_number - self.last_event_frame < EVENT_COOLDOWN_FRAMES:
			return False, "Monitoring after previous event"
		self.last_event_frame = frame_number
		return True, reason


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("video", help="Path to the input MP4 or other video file")
	parser.add_argument(
		"--output",
		default="collision_analysis.mp4",
		help="Path for the annotated output video",
	)
	parser.add_argument(
		"--no-display",
		action="store_true",
		help="Do not open a preview window while processing",
	)
	return parser.parse_args()


def largest_forward_vehicle(result, frame_width: int, frame_height: int):
	"""Return the largest confident vehicle in the central forward corridor."""
	if result.boxes is None:
		return None

	best_area = 0.0
	for box in result.boxes:
		confidence = float(box.conf[0])
		class_id = int(box.cls[0])
		if confidence < MIN_CONFIDENCE or class_id not in VEHICLE_CLASSES:
			continue

		x1, y1, x2, y2 = map(int, box.xyxy[0])
		center_x = (x1 + x2) / 2 / frame_width
		bottom_y = y2 / frame_height
		area = max(0, x2 - x1) * max(0, y2 - y1) / (frame_width * frame_height)
		if 0.15 <= center_x <= 0.85 and bottom_y >= 0.45 and area > best_area:
			best_area = area

	return best_area or None


def analyze_video(video_path: Path, output_path: Path, show_preview: bool) -> bool:
	if not video_path.exists():
		raise FileNotFoundError(f"Input video not found: {video_path}")

	video = cv2.VideoCapture(str(video_path))
	if not video.isOpened():
		raise RuntimeError(f"OpenCV could not open video: {video_path}")

	width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = video.get(cv2.CAP_PROP_FPS) or 30.0
	writer = cv2.VideoWriter(
		str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
	)
	model = YOLO(MODEL_NAME)
	detector = CollisionDetector()
	collision_detected = False
	frame_number = 0

	try:
		for result in model.track(
			source=str(video_path),
			stream=True,
			persist=True,
			conf=MIN_CONFIDENCE,
			classes=list(VEHICLE_CLASSES),
			verbose=False,
		):
			frame = result.orig_img
			area = largest_forward_vehicle(result, width, height)
			collision, reason = detector.update(frame_number, area)
			collision_detected = collision_detected or collision

			annotated = result.plot()
			color = (0, 0, 255) if collision else (0, 180, 0)
			label = "POSSIBLE COLLISION" if collision_detected else "NO COLLISION DETECTED"
			cv2.putText(annotated, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			cv2.putText(annotated, reason, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
			writer.write(annotated)

			if show_preview:
				cv2.imshow("Forward Collision Warning", annotated)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
	finally:
		video.release()
		writer.release()
		cv2.destroyAllWindows()

	return collision_detected


def main() -> None:
	args = parse_arguments()
	collision_detected = analyze_video(Path(args.video), Path(args.output), not args.no_display)
	print(f"Annotated video saved to: {args.output}")
	print("Result: POSSIBLE COLLISION DETECTED" if collision_detected else "Result: NO COLLISION DETECTED")


if __name__ == "__main__":
	main()
