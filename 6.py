"""Detect traffic signs using their colors in an input image."""

import cv2
import numpy as np


IMAGE_PATH = "traffic_signs.jpeg"


def main() -> None:
	image = cv2.imread(IMAGE_PATH)
	if image is None:
		raise FileNotFoundError(f"{IMAGE_PATH} not found")

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
	red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
	red = red1 | red2
	blue = cv2.inRange(hsv, (90, 80, 70), (130, 255, 255))
	yellow = cv2.inRange(hsv, (18, 80, 80), (40, 255, 255))
	combined = red | blue | yellow

	kernel = np.ones((5, 5), np.uint8)
	combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
	combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
	contours, _ = cv2.findContours(
		combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
	)

	for contour in contours:
		area = cv2.contourArea(contour)
		if area < 150:
			continue

		perimeter = cv2.arcLength(contour, True)
		if perimeter == 0:
			continue

		circularity = 4 * np.pi * area / (perimeter * perimeter)
		if circularity > 0.35:
			x, y, width, height = cv2.boundingRect(contour)
			cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
			cv2.putText(
				image, "Traffic Sign", (x, max(y - 8, 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
			)

	cv2.imshow("Color Based Traffic Sign Detection", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()