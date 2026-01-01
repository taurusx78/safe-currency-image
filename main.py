import cv2
import numpy as np
import os

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

countries_data = {
    'krw': [1000, 5000, 10000, 50000],
    'vnd': [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
    'jpy': [1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
    'idr': [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
    'thb': ['25s', '50s', 1, 2, 5, 10, 20, 50, 100, 500, 1000],  # 25, 50 사땅(satang)은 s를 붙여 구분
    'php': [1, 5, 10, 20, 50, 100, 500, 1000]
}

for country, currency_list in countries_data.items():
    print(f"--- Processing {country.upper()} ---")
    
    # 출력 경로 생성
    output_dir = f"/currency/safe/v3/{country}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for value in currency_list:
        input_path = f"/currency/origin/{country}/{country}_{value}.jpg"
        img = cv2.imread(input_path)
        
        if img is None:
            print(f"Skipping: {input_path} (File not found)")
            continue
            
        print(f"Processing: {country}_{value}.jpg")

        # 2. 컬러 유지 + 노이즈 제거 (지폐 디테일 제거 및 고유번호 뭉개기)
        color_smooth = cv2.bilateralFilter(
            img,
            d=15,  # 필터 크기 확대
            sigmaColor=80,
            sigmaSpace=80
        )

        # 3. 그레이 변환
        gray = cv2.cvtColor(color_smooth, cv2.COLOR_BGR2GRAY)

        # 4. 추가 노이즈 제거 (텍스트 및 미세 디테일 제거 핵심)
        gray_blur = cv2.medianBlur(gray, 7)  # Median Blur 추가: 고유번호 제거 효과
        gray_blur = cv2.GaussianBlur(gray_blur, (9, 9), 0)

        # 5. 에지 검출 (윤곽만)
        edges = cv2.Canny(gray_blur, 40, 100)

        # 6. 선 굵기 조절 (지폐 재현 방지 핵심)
        kernel = np.ones((1, 1), np.uint8)
        edges_thick = cv2.dilate(edges, kernel, iterations=1)

        # 7. 흰 배경 + 검은 선
        edges_inv = cv2.bitwise_not(edges_thick)
        edges_colored = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

        # 8. 컬러 + 윤곽 합성
        stylized = cv2.bitwise_and(color_smooth, edges_colored)

        final_img = reduce_colors(stylized, k=6)

        # 10. 높이 100px로 리사이즈 (비율 유지)
        target_height = 100
        h, w = final_img.shape[:2]
        scale = target_height / h
        target_width = int(w * scale)

        final_img_resized = cv2.resize(
            final_img,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )

        # 11. 저장
        output_path = f"{output_dir}/{country}_{value}.png"
        cv2.imwrite(output_path, final_img_resized)
        print(f"Saved: {output_path}")
