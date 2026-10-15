from pathlib import Path

import cv2

image_path = Path(__file__).with_name("896b9fb6-56e9-469c-a5f2-a115c78f4bc7.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
rects, weights=hog.detectMultiScale(
    image,
    winStride=(4,4),
    padding=(8,8),
    scale=1.03
)
for (x,y,w,h), weight in zip(rects, weights):
    cv2.rectangle(
        image,
        (x,y),
        (x+w,y+h),
        (0,255,0),
        2
    )
    cv2.putText(
        image,
        f"Person:{weight:.2f}",
        (x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2
    )
print("Number of pedestrians detected:",len(rects))
cv2.imshow("Pedestrian Detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()