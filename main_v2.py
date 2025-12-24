import cv2
import numpy as np

# 색상 수 줄이기 (Posterization → 복제 방지)
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

currency_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]

for value in currency_list:
    input_path = f"/Users/gimjieun/Desktop/currency/origin/vnd/vnd_{value}.jpg"
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Skipping: {input_path} (File not found)")
        continue
        
    print(f"Processing: vnd_{value}.jpg")

    # 2. 컬러 유지 + 노이즈 제거 (지폐 디테일 제거)
    color_smooth = cv2.bilateralFilter(
        img,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    # 3. 그레이 변환
    gray = cv2.cvtColor(color_smooth, cv2.COLOR_BGR2GRAY)

    # 4. 추가 노이즈 제거
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 5. 에지 검출 (윤곽만)
    edges = cv2.Canny(gray_blur, 50, 120)

    # 6. 선 굵기 조절 (지폐 재현 방지 핵심)
    kernel = np.ones((2, 2), np.uint8)
    edges_thick = cv2.dilate(edges, kernel, iterations=1)

    # 7. 흰 배경 + 검은 선
    edges_inv = cv2.bitwise_not(edges_thick)
    edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # 8. 컬러 + 윤곽 합성
    stylized = cv2.bitwise_and(color_smooth, edges_colored)

    final_img = reduce_colors(stylized, k=8)

    # 10. 높이 200px로 리사이즈 (비율 유지)
    target_height = 200
    h, w = final_img.shape[:2]
    scale = target_height / h
    target_width = int(w * scale)

    final_img_resized = cv2.resize(
        final_img,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    # 11. 저장
    output_path = f"/Users/gimjieun/Desktop/currency/safe/v2/vnd/vnd_{value}.png"
    cv2.imwrite(output_path, final_img_resized)
    print(f"Saved: {output_path}")
