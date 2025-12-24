import cv2
import numpy as np

# 1. 이미지 로드
img = cv2.imread("/Users/gimjieun/Desktop/vnd_1000.jpg")

# 2. 강한 블러 (지폐 디테일 제거)
smooth = cv2.bilateralFilter(img, d=15, sigmaColor=150, sigmaSpace=150)

# 3. 색상 극단적 축소 (아이콘화 핵심)
def reduce_colors(img, k=3):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    _, labels, centers = cv2.kmeans(
        Z, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img.shape)

flat_color = reduce_colors(smooth, k=3)

# 4. 윤곽선 추출
gray = cv2.cvtColor(flat_color, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# 5. 굵은 외곽선
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges_inv = cv2.bitwise_not(edges)
edges_col = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

# 6. 색상 + 윤곽 결합
icon = cv2.bitwise_and(flat_color, edges_col)

# 7. 중앙 크롭 → 정사각형 (UI 친화)
# h, w, _ = icon.shape
# size = min(h, w)
# start_x = (w - size) // 2
# start_y = (h - size) // 2
# icon_square = icon[start_y:start_y+size, start_x:start_x+size]

# # 8. 아이콘 크기 통일
# icon_final = cv2.resize(icon_square, (256, 256), interpolation=cv2.INTER_AREA)

# 9. 저장
cv2.imwrite("/Users/gimjieun/Desktop/vnd_1000_icon_ui.png", icon)
