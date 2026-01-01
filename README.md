# safe-currency-image
지폐(통화) 원본 이미지를 **디테일을 단순화한 이미지(PNG)** 로 변환하는 Python 스크립트 모음입니다.  
노이즈 제거/윤곽선 추출/색상 수 축소 등을 통해 고유번호·미세 문양 같은 재현 가능한 디테일을 약화시킨 이미지를 생성합니다.

## 실행 환경
- Python 3.x
- `opencv-python`
- `numpy`

설치 예시
```bash
pip install opencv-python numpy
```

## 파일 설명
- `main.py`: 여러 국가/권종 이미지를 읽어 safe 스타일로 변환 후 저장(높이 100px로 리사이즈)
- `cartoon.py`: 단일 이미지에 카툰 스타일 변환 실험용 스크립트
- `game.py`: 단일 이미지에 아이콘화(색상 축소 + 윤곽선) 실험용 스크립트

## 사용 방법
1) `main.py` 안의 경로를 내 환경에 맞게 수정합니다.
- 입력: `currency/origin/{country}/{country}_{value}.jpg`
- 출력: `currency/safe/.../{country}/{country}_{value}.png`

2) 실행
```bash
python main.py
```