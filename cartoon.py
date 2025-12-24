import cv2
import numpy as np

# 1. 이미지 로드
img = cv2.imread("/Users/gimjieun/Desktop/krw_10000.jpg")

# 2. 컬러 단순화 (강한 노이즈 제거)
color = cv2.bilateralFilter(img, d=11, sigmaColor=100, sigmaSpace=100)

# 3. 색상 수 줄이기 (Posterization)
def posterize(img, levels=6):
    img = np.float32(img) / 255.0
    img = np.floor(img * levels) / levels
    return np.uint8(img * 255)

color_simple = posterize(color, levels=6)

# 4. 그레이 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5. 블러 (디테일 제거)
gray_blur = cv2.medianBlur(gray, 7)

# 6. 에지 검출 (카툰 스타일용)
edges = cv2.Canny(gray_blur, 80, 160)

# 7. 외곽선 굵게 (일러스트 느낌)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

# 8. 흰 배경 + 검은 선
edges_inv = cv2.bitwise_not(edges)
edges_col = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

# 9. 색상 + 외곽선 합성
cartoon = cv2.bitwise_and(color_simple, edges_col)

# 10. 최종 저장
cv2.imwrite("/Users/gimjieun/Desktop/krw_10000_cartoon_safe.png", cartoon)
