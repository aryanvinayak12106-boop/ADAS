import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\Aryan salunkhe\Downloads\road2.jpg"
image = cv2.imread(image_path)
if image is None:
	raise FileNotFoundError(f"{image_path} not found")
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
h, w = edges.shape
mask = np.zeros_like(edges)

polygon = np.array([[
	(0, h),
	(w, h),
	(int(0.58 * w), int(0.58 * h)),
	(int(0.42 * w), int(0.58 * h))
]], dtype=np.int32)

cv2.fillPoly(mask, polygon, 255)
roi_edges = cv2.bitwise_and(edges, mask)

# HOUGH TRANSFORM

lines = cv2.HoughLinesP(
	roi_edges,
	rho=1,
	theta=np.pi / 180,
	threshold=30,
	minLineLength=30,
	maxLineGap=100
)

if lines is not None:
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("Lane Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
