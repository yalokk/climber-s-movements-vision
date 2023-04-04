import cv2 as cv
import numpy as np
import pylab as pl
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


k = 6  # Количество кластеризованных категорий
path = "images2/climb7.jpg"
iteration = 4  # Текущий номер 4
iterations = 200  # Максимальное количество циклов кластеризации

# Загружаем и показываем изображение
img = cv.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Уменьшаем медианный шум - удаляем точки
img = cv2.medianBlur(img, 7)
pl.imshow(img)
pl.show()

# Уменьшаем шум на фото, но сохраняем контуры
img = cv2.bilateralFilter(img, 30, 35, 45)
pl.imshow(img)
pl.show()

# Изменяем форму изображения, чтобы оно представляло собой список пикселей
image = img.reshape((img.shape[0] * img.shape[1], 3))

# Вызываем реализацию алгоритма кластеризации
# clt = KMeans(n_clusters = k, n_jobs = iteration, max_iter = iterations)
clt = KMeans(n_clusters=k)
clt.fit(image)

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)

# Отображаем гистограмму
fig = plt.figure()
ax = fig.add_subplot(211)
ax.imshow(img)
ax = fig.add_subplot(212)
ax.imshow(bar)
plt.show()

cv.waitKey(0)
