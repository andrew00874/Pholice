import cv2
import numpy as np
import os
import sys
import subprocess

def remove_srgb_profile(image_path):
    """ImageMagick을 사용하여 PNG의 sRGB 프로필 제거"""
    fixed_image_path = "fixed_image.png"

    try:
        # ImageMagick 버전 확인
        version_output = subprocess.run(["magick", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "ImageMagick" in version_output.stdout:
            convert_cmd = ["magick"]  # IMv7 버전이면 "magick" 사용
        else:
            convert_cmd = ["convert"]  # 구버전에서는 "convert" 그대로 사용

        # ImageMagick을 사용하여 sRGB 프로필 제거
        subprocess.run(convert_cmd + [image_path, "-strip", fixed_image_path], check=True)
        return fixed_image_path

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ ImageMagick이 설치되지 않았거나 사용할 수 없습니다. OpenCV로 해결합니다.")

        # OpenCV로 이미지 다시 저장 (대체 방법)
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imwrite(fixed_image_path, image)
            return fixed_image_path
        else:
            print("❌ 이미지를 불러올 수 없습니다.")
            sys.exit(1)

def detect_face_and_enhancement_level(image_path):
    # 현재 스크립트의 폴더 경로
    base_path = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(base_path, "models/deploy.prototxt")
    model_path = os.path.join(base_path, "models/res10_300x300_ssd_iter_140000.caffemodel")

    # DNN 모델 파일 존재 여부 확인
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print("❌ DNN 모델 파일을 찾을 수 없습니다! 'models' 폴더에 올바른 파일이 있는지 확인하세요.")
        sys.exit(1)

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # 입력 인자가 올바른지 확인
    if len(sys.argv) < 2:
        print("❌ 사용법: python pholice.py <이미지 파일 경로>")
        sys.exit(1)

    image_path = sys.argv[1]

    # 이미지 파일 존재 여부 확인
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        sys.exit(1)

    # PNG의 sRGB 문제 해결을 위해 ImageMagick 또는 OpenCV를 사용하여 이미지 수정
    image_path = remove_srgb_profile(image_path)

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 이미지를 불러올 수 없습니다! 지원되는 형식인지 확인하세요.")
        sys.exit(1)

    (h, w) = image.shape[:2]

    # 얼굴 감지 수행
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # 얼굴이 감지되지 않은 경우
    if detections.shape[2] == 0:
        print("⚠️ 인물이 감지되지 않았습니다!")
        return

    # 얼굴 감지 시 분석 수행
    distortion_score = detect_distortion(image)
    saturation_score = detect_saturation(image)
    blur_score = detect_blur(image)
    artificial_look_score = detect_ai_correction(image)

    # 보정 정도 판단 (0~5단계)
    enhancement_level = classify_enhancement(distortion_score, saturation_score, blur_score, artificial_look_score)

    # 결과 출력
    print("\n===== 📊 분석 결과 =====")
    print(f"📏 왜곡 지표: {distortion_score:.2f}/5")
    print(f"🎨 채도 지표: {saturation_score:.2f}/5")
    print(f"🖌 블러(뭉개짐) 지표: {blur_score:.2f}/5")
    print(f"🤖 AI 보정 지표: {artificial_look_score:.2f}/5")
    print(f"🔎 최종 보정 강도: {enhancement_level}/5")

    # 분석 근거 출력
    print("\n📋 분석 근거:")
    reasons = []
    if distortion_score > 3:
        reasons.append("얼굴 또는 윤곽선이 비정상적으로 변형됨 (왜곡 가능성)")
    if saturation_score > 3:
        reasons.append("색이 과도하게 강조됨 (채도 증가)")
    if blur_score > 3:
        reasons.append("피부 질감이 부드러워짐 (AI 보정으로 인한 블러 가능성)")
    if artificial_look_score > 3:
        reasons.append("대비 감소 및 색조 균일화가 감지됨 (AI 보정 흔적)")

    if reasons:
        for r in reasons:
            print(f"✅ {r}")
    else:
        print("✅ 자연스러운 사진으로 판단됨!")

def detect_distortion(image):
    """ 왜곡 정도를 평가하는 함수 """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])
    return min(edge_density * 10, 5)  # 0~5 점수 스케일링

def detect_saturation(image):
    """ 채도를 평가하는 함수 """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    avg_saturation = np.mean(saturation)
    return min(avg_saturation / 50, 5)  # 0~5 점수 스케일링

def detect_blur(image):
    """ 블러(픽셀 뭉개짐) 정도를 평가하는 함수 """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var < 50:
        return 5
    elif laplacian_var < 100:
        return 4
    elif laplacian_var < 150:
        return 3
    elif laplacian_var < 200:
        return 2
    else:
        return 1

def detect_ai_correction(image):
    """ AI 보정의 전형적인 특징 감지 """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    contrast = np.var(enhanced_l)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_std = np.std(hsv[:, :, 0])

    return 5 if contrast < 1000 and hue_std < 10 else min((contrast / 1000), 5)

def classify_enhancement(distortion, saturation, blur, ai_correction):
    """ 보정 강도를 종합하여 평가 """
    score = (distortion + saturation + blur + ai_correction) / 4
    return round(score)

if __name__ == "__main__":
    detect_face_and_enhancement_level(sys.argv[1])
