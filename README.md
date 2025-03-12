# 얼굴 이미지 보정 감지 프로젝트

## 개요
이 프로젝트는 얼굴 이미지의 보정 정도를 0~5단계로 분석하는 시스템입니다. 다음과 같은 주요 지표를 기반으로 보정 강도를 평가합니다:

- **왜곡 감지**: 얼굴 구조가 비정상적으로 변형되었는지 확인
- **채도 분석**: 색상이 과도하게 강조되었는지 측정
- **블러 감지**: 픽셀의 부드러움 정도를 평가하여 인위적 피부 보정 여부 판별
- **AI 보정 감지**: 대비와 색조 균일성을 분석하여 AI 기반 보정 여부 판단

이미지에서 얼굴이 감지되지 않을 경우, 경고 메시지를 출력하고 보정 단계를 표시하지 않습니다.

---

## 주요 기능
✅ OpenCV DNN 모델을 활용한 얼굴 감지
✅ 0~5단계의 보정 강도 평가
✅ 각 지표(왜곡, 채도, 블러, AI 보정) 점수 제공
✅ 인위적인 피부 보정 및 색 보정 감지
✅ 얼굴이 감지되지 않을 경우 경고 메시지 출력
✅ ImageMagick을 활용하여 PNG 파일의 잘못된 sRGB 프로필 자동 수정
✅ ImageMagick이 없을 경우 OpenCV를 사용하여 자동 수정

---

## 설치 방법

Python과 필요한 라이브러리를 설치하세요:

```bash
pip install opencv-python numpy
```

### ImageMagick 설치 (선택 사항)
PNG 파일의 sRGB 문제를 자동으로 수정하려면 ImageMagick을 설치해야 합니다.

- **MacOS (Homebrew 사용)**
  ```bash
  brew install imagemagick
  ```
- **Ubuntu/Linux**
  ```bash
  sudo apt install imagemagick
  ```
- **Windows**
  [ImageMagick 공식 사이트](https://imagemagick.org/script/download.php)에서 다운로드 후 설치

---

## 사용법

다음 명령어를 실행하여 이미지의 보정 강도를 분석할 수 있습니다:

```bash
python pholice.py <이미지 파일 경로>
```

### 예시 출력:

```
===== 분석 결과 =====
📏 왜곡 지표: 2.50/5
🎨 채도 지표: 4.00/5
🖌 블러(뭉개짐) 지표: 3.50/5
🤖 AI 보정 지표: 4.20/5
🔎 최종 보정 강도: 4/5

📋 분석 근거:
✅ 색상이 과도하게 강조됨 (채도 증가)
✅ 피부 질감이 부드러워짐 (AI 보정으로 인한 블러 가능성)
✅ 대비 감소 및 색조 균일화 감지 (AI 보정 흔적)
```

얼굴이 감지되지 않을 경우:

```
인물이 감지되지 않았습니다!
```

---

## 작동 방식

1. **얼굴 감지**
   - OpenCV DNN(Caffe 기반 SSD) 모델을 사용하여 얼굴 감지
   - 얼굴이 감지되지 않으면 경고 메시지를 출력

2. **왜곡 분석**
   - 엣지 검출을 사용하여 얼굴 윤곽선 변형 여부 판단

3. **채도 분석**
   - 이미지를 HSV 색 공간으로 변환 후 평균 채도 계산

4. **블러 감지**
   - 라플라시안 변환을 활용하여 피부 질감 손실 여부 측정

5. **AI 보정 감지**
   - CLAHE 대비 분석 및 색조 분산 분석을 통해 AI 보정 여부 평가

6. **PNG의 sRGB 프로필 자동 수정**
   - ImageMagick이 설치된 경우, `magick -strip` 명령어를 사용하여 sRGB 프로필을 제거
   - ImageMagick이 없을 경우, OpenCV를 이용해 이미지를 다시 저장하여 문제 해결

---

## 향후 개선 사항

- 얼굴 왜곡 분석을 위한 딥러닝 기반 랜드마크 탐지 추가
- 특정 편집 소프트웨어(예: 포토샵, FaceTune) 감지 기능 추가
- 다중 이미지 일괄 처리 지원

---

## 라이선스
이 프로젝트는 MIT 라이선스로 배포됩니다.

---

## 개발자
Jaemin 개발
