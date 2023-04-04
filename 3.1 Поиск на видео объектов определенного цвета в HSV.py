import cv2
import numpy as np
from matplotlib import pyplot as pl


cap = cv2.VideoCapture('images2/video_climb.MOV')
# Создаем цикл, перебираем картинки
cf = 0

while True:
    success, img = cap.read()
    # В success передаем T или F в зависимости, удалось ли прочитать текущее изображение в видео

    # Уменьшаем шум на фото, но сохраняем контуры
    img = cv2.bilateralFilter(img, 30, 35, 45)

    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    low_hsv = (130, 20, 20)
    up_hsv = (174, 255, 255)

    hsv_mask = cv2.inRange(img_HSV, low_hsv, up_hsv)

    cv2.imshow('Mask', hsv_mask)
    cv2.imshow('Image', img)

    # Позволяет проигрывать кадры с нужной скоростью и при необходимости
    # выйти из видео с помощью клавиши q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


"""
# Copy the thresholded image.
im_floodfill = hsv_mask.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = hsv_mask.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = hsv_mask | im_floodfill_inv

# Display images.
cv2.imshow("Thresholded Image", hsv_mask)
cv2.imshow("Floodfilled Image", im_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Финальный без дырок", im_out)
cv2.waitKey(0)
"""

# cv2.imshow('Window', violet_hsv_mask)
# cv2.waitKey()
