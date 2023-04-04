import cv2
import numpy as np
from matplotlib import pyplot as pl

# Импортируем изображение
directory = 'images2/cont_climb4.jpg'
img = cv2.imread(directory)
orig_image = img.copy()

# Морфологическая операция Закрытия, заливаем дыры
kernel = np.ones((21, 21), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pl.show()