import rasterio
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Polygon
import geopandas as gpd
matplotlib.use('Agg')

# -------------------------------
# Загрузка и чтение снимков
# -------------------------------

# радарный снимок
with rasterio.open('Sentinel-1.tiff') as s1:
    vv = s1.read(1)
    vh = s1.read(2)
    data_mask_s1 = s1.read(6)   #область без данных

# оптический снимок
with rasterio.open('Sentinel-2L2A.tiff') as s2:
    red = s2.read(4)   #B04
    green = s2.read(3) #B03
    nir = s2.read(8)   #B08
    scl = s2.read(16)  #маска облаков, теней
    data_mask_s2 = s2.read(17)    
    scl = s2.read(16)

# ---------------------------------------------
# функии обработки шум со снимков (сглаживание)
# ---------------------------------------------

#    для s1 фильтр Ли
def filter_lee (imgane, size_window = 3):
    imgane_float = imgane.astype(np.float32)

    mean = ndimage.uniform_filter(imgane_float, size_window)
    mean_sq = ndimage.uniform_filter(imgane_float**2, size_window)

    variance = mean_sq - mean**2

    noise_var = (mean * 0.2)**2

    k = variance/(variance + noise_var + 1e-10)

    result = mean + k * (imgane_float - mean)

    return result

#     для s2 гауссово размытие
def gaussian_blur (imagane, sigma = 1.0):
    return ndimage.gaussian_filter(imagane.astype(np.float32), sigma = sigma)

#Очистка шума
vv_clean = filter_lee(vv, size_window=3)
vh_clean = filter_lee(vh, size_window=3)

red_clean = gaussian_blur(red, sigma = 1.0)
green_clean = gaussian_blur(green, sigma = 1.0)
nir_clean = gaussian_blur(nir, sigma = 1.0)

print("Очистка снимков от шума завершена")

# ---------------------------------------------
# оставляем только реальные данные
# ---------------------------------------------

#оставляем scl = 4(растительность) / 5(почва) / 6(вода) / 11(снег)
valid_scl = np.isin(scl, [4, 5, 6, 11])
valid_data = (data_mask_s1 > 0) & (data_mask_s2 > 0)
final_mask = valid_scl & valid_data

print('Оставлены только валидные объекты')

# ---------------------------------------------
# расчёт признаков
# ---------------------------------------------

# нахожу границы между растительностями
ndvi = (nir_clean - red_clean)/(nir_clean + red_clean + 1e-10)
ndvi = np.where(final_mask, ndvi, np.nan)

print('NDVI найдены')

#ndwi влажность
ndwi = (green_clean - nir_clean)/(green_clean + nir_clean + 1e-10)
ndwi = np.where(final_mask, ndwi, np.nan)

print('NDWI найден')

# VV/VH ratio (шерховатость \ гладкость поверхности)
vv_vh_ratio = vv_clean / (vh_clean + 1e-10)
vv_vh_ratio = np.where(final_mask, vv_vh_ratio, np.nan)

print('VV/VH ratio найден')

# SAVI (почвенный индекс растительности)
L = 0.5   # коэффицент корректировки почвы
savi = ((nir_clean - red_clean) * (1 + L)) / (nir_clean + red_clean + L + 1e-10)
savi = np.where(final_mask, savi, np.nan)

print('SAVI найден')

# ---------------------------------------------
# нормализация признаков
# ---------------------------------------------
ndvi_n = cv2.normalize(np.nan_to_num(ndvi), None, 0, 1, cv2.NORM_MINMAX)
ndwi_n = cv2.normalize(np.nan_to_num(ndwi), None, 0, 1, cv2.NORM_MINMAX)
savi_n = cv2.normalize(np.nan_to_num(savi), None, 0, 1, cv2.NORM_MINMAX)
vvvh_n = cv2.normalize(np.nan_to_num(vv_vh_ratio), None, 0, 1, cv2.NORM_MINMAX)

ndvi_filtered = np.where(final_mask, ndvi, np.nan)
ndvi_for_vis = np.nan_to_num(ndvi_filtered, nan=0.0)
ndvi_norm = cv2.normalize(ndvi_for_vis,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

# -------------------------------
# многопризнаковая сегментация
# -------------------------------

stacked_features = np.dstack([
    ndvi_n,
    ndwi_n,
    savi_n,
    vvvh_n
])

valid_pixels = stacked_features[final_mask]
pixels = np.float32(valid_pixels)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

K = 10

_, labels, centers = cv2.kmeans(
    pixels,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

labels_full = np.full(ndvi.shape, -1, dtype=np.int32)
labels_full[final_mask] = labels.flatten()
labels_2d = labels_full

# Вычисляем средний сырой NDVI для каждого кластера
cluster_raw_ndvi = []
for k in range(K):
    mask_k = (labels_full == k)
    if np.sum(mask_k) > 0:
        cluster_raw_ndvi.append(np.nanmean(ndvi[mask_k]))
    else:
        cluster_raw_ndvi.append(-999)  # пустой кластер

# выбираем кластеры растительности
field_mask = np.zeros_like(labels_2d, dtype=np.uint8)

for i in range(K):
    raw_ndvi = cluster_raw_ndvi[i]
    print(f"Кластер {i}: сырой NDVI = {raw_ndvi:.3f}")
    
    if 0.15 < raw_ndvi <= 0.6:  # Реальные значения NDVI для полей
        field_mask[labels_2d == i] = 255

field_mask = np.where(final_mask, field_mask, 0).astype(np.uint8)

print('KMeans сегментация выполнена')

# -------------------------------
# морфология
# -------------------------------

kernel = np.ones((3, 3), np.uint8)

field_mask = cv2.morphologyEx(
    field_mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations = 1
)

field_mask = cv2.morphologyEx(
    field_mask,
    cv2.MORPH_CLOSE, 
    kernel, 
    iterations = 1
)

print('Морфология выполнена')

# сохраняем маску
plt.figure(figsize=(10,10))
plt.imshow(field_mask, cmap='gray')
plt.title('Morphology')
plt.axis('off')

plt.savefig('morphology.png', bbox_inches='tight')
plt.close()

#ndvi_grad = cv2.Sobel(ndvi_n, cv2.CV_64F, 1, 0, ksize=3)
#ndvi_grad = cv2.convertScaleAbs(ndvi_grad)
#vv_grad = cv2.Sobel(vvvh_n, cv2.CV_64F, 1, 0, ksize=3)
#vv_grad = cv2.convertScaleAbs(vv_grad)

#combined_grad = cv2.max(ndvi_grad, vv_grad)

#_, edges = cv2.threshold(combined_grad, 60, 255, cv2.THRESH_BINARY)

# Где есть резкий перепад поле должно разделиться
#field_mask = cv2.subtract(field_mask, edges)
#field_mask = np.clip(field_mask, 0, 255).astype(np.uint8)

# очистка после вычитания
#field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# -------------------------------
# векторизация
# -------------------------------

# ищем контуры белых областей
contours, _ = cv2.findContours(
    field_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print('Контуров найдено:', len(contours))

filtered_contours = []
for cnt in contours:

    area = cv2.contourArea(cnt)

    if 1800 < area:
        filtered_contours.append(cnt)

print (f"После фильтрации по площади осталось : { len(filtered_contours)} контуров")

with rasterio.open('Sentinel-2L2A.tiff') as src:
    profile = src.profile

profile.update(dtype=rasterio.uint8,count=1)

with rasterio.open('ndvi_result.tif','w',**profile) as dst:
    dst.write(ndvi_norm, 1)

with rasterio.open('field_mask.tif','w',**profile) as dst:
    dst.write(field_mask, 1)

print('GeoTIFF сохранены')

polygons = []
transform = profile['transform']

for cnt in filtered_contours:
    if len(cnt) >= 3:
        try:
            epsilon = 0.003 * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, epsilon, True)

            coords = []

            for point in cnt[:, 0, :]:
                x_pix, y_pix = point

                x_geo, y_geo = rasterio.transform.xy(
                    transform,
                    int(y_pix),
                    int(x_pix)
                )

                coords.append((x_geo, y_geo))

            # polygon only if enough points
            if len(coords) >= 4:

                # замыкаем контур
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

                poly = Polygon(coords)

                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)
        except:
            pass

gdf = gpd.GeoDataFrame(geometry=polygons)
gdf.set_crs("EPSG:4326", inplace=True)

# сохраняем результат в двух форматах
gdf.to_file(
    'contours.geojson',
    driver='GeoJSON'
)

gdf.to_file(
    'contours.gpkg',
      driver='GPKG'
)

print('GeoJSON сохранен')

contour_image = np.zeros_like(field_mask)

cv2.drawContours(
    contour_image,
    filtered_contours,
    -1,
    255,
    1
)

plt.figure(figsize=(10,10))
plt.imshow(contour_image, cmap='gray')
plt.title('Contours')
plt.axis('off')

plt.savefig('contours.png', bbox_inches='tight')
plt.close()

print('Contours сохранены')

#наглядный результат
plt.figure(figsize=(10,10))
plt.imshow(ndvi_norm, cmap='RdYlGn')
plt.title('NDVI')
plt.axis('off')

plt.savefig('ndvi.png', bbox_inches='tight')
plt.close()

print('Изображения сохранены')


