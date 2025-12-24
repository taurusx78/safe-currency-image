import cv2
import numpy as np

currency_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]

for value in currency_list:
    input_path = f"/Users/gimjieun/Desktop/currency/origin/vnd/vnd_{value}.jpg"
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Skipping: {input_path} (File not found)")
        continue
        
    print(f"Processing: vnd_{value}.jpg")

    # 1. 컬러 유지 + 노이즈 제거
    color_smooth = cv2.bilateralFilter(
        img, d=9, sigmaColor=75, sigmaSpace=75
    )

    # 2. 그레이
    gray = cv2.cvtColor(color_smooth, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. 에지 검출 (임계값 ↑ → 윤곽선 감소)
    edges = cv2.Canny(gray_blur, 80, 160)

    # 4. 선 굵기 최소화 (dilate 제거)
    # kernel = np.ones((2, 2), np.uint8)
    # edges = cv2.dilate(edges, kernel, iterations=1)

    # 5. 윤곽선 마스크 생성
    edges_inv = cv2.bitwise_not(edges)
    edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # 6. 윤곽선 강도 줄여서 합성
    stylized = cv2.addWeighted(
        color_smooth, 0.9,   # 원본 컬러 비중 ↑
        edges_colored, 0.1,  # 윤곽선 비중 ↓
        0
    )

    # 7. 색상 수 줄이기
    def reduce_colors(img, k=8):
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        _, label, center = cv2.kmeans(
            Z, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        return center[label.flatten()].reshape(img.shape)

    final_img = reduce_colors(stylized, k=8)

    # 8. 높이 100px 리사이즈 (비율 유지)
    target_height = 100
    h, w = final_img.shape[:2]
    scale = target_height / h
    target_width = int(w * scale)

    final_img_resized = cv2.resize(
        final_img,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    output_path = f"/Users/gimjieun/Desktop/currency/safe/vnd/vnd_{value}.png"
    cv2.imwrite(output_path, final_img_resized)
    print(f"Saved: {output_path}")
