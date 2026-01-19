# 1. cai thu vien
import cv2
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient

def tomato_classification(color_fruit, shape_fruit):
    if color_fruit == "Ripe" and shape_fruit == "Beautiful":
        return "Loai A"
    if color_fruit == "Ripe" and shape_fruit == "Distorted":
        return "Loai B"
    if color_fruit == "Unripe" and shape_fruit == "Beautiful":
        return "Loai C"
    if color_fruit == "Unripe" and shape_fruit == "Distorted":
        return "Loai D"
    return "Không xác định"
# duong dan hinh anh co trong source code
img_path = "tomato_6.jpg"

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="9Jn7ADj8ghfbbPwGxKS2"
)

result1 = CLIENT.infer(img_path, model_id="tomato-uvzpm/1")
predictions = result1['predictions']
img_main = cv2.imread(img_path)

if len(predictions) > 0:
    for pred in predictions:
        # Lấy tọa độ (Roboflow trả về tâm x, y và kích thước w, h)
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        label = pred['class']
        confidence = pred['confidence']

        # Chuyển từ tọa độ tâm sang tọa độ góc để vẽ hình chữ nhật
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        # Vẽ khung (màu xanh lá: 0, 255, 0)
        cv2.rectangle(img_main, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ghi tên và độ tin tưởng
        text = f"{label} {confidence:.2f}"
        cv2.putText(img_main, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ca chua da cat
        tomato_roi = img_main[y1:y2, x1:x2]
        if tomato_roi.size == 0:
            continue

        # mask màu đỏ
        hsv = cv2.cvtColor(tomato_roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red_mask = ((h < 10) | (h > 160)) & (s > 80)
        red_ratio = red_mask.sum() / red_mask.size

        if red_ratio > 0.35:
            color_result = "Ripe"
        else:
            color_result = "Unripe"

        ratio = width / height
        if 0.8 <= ratio <= 1.4:
            shape_result = "Beautiful"
        else:
            shape_result = "Distorted"

        tomatoClass = tomato_classification(color_result,shape_result)
        cv2.putText(
            img_main,
            f"{tomatoClass} | {color_result} | {shape_result}",
            (x1, y2 + 50),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 0, 150),
            2,
            cv2.LINE_AA
        )
        img_main = cv2.resize(img_main, (800, int(height * (800 / width))))
else:
    print("khong tim thay qua trai")

# ket noi toi workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="9Jn7ADj8ghfbbPwGxKS2"
)

# Goi API xu li hinh anh
result = client.run_workflow(
    workspace_name="hi-vdtti",
    workflow_id="detect-count-and-visualize",
    images={
        "image": img_path
    },
    use_cache=True
)

count_objects = result[0]['count_objects']
if count_objects > 0:
    # xử lí dữ liệu trả về
    output_image = result[0]['output_image']
    image_data = result[0]['predictions']['image']
    w,h = image_data['width'], image_data['height']

    # cấu hình dữ liệu hình ảnh
    img_bytes = base64.b64decode(output_image)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (800, int(h * (800 / w))))

    # Hiển thị ảnh đã thu nhỏ
    cv2.imshow("Ket qua nhan dien", img)
    cv2.waitKey(0)  # Nhấn phím bất kỳ để đóng cửa sổ
    cv2.destroyAllWindows()
else:
    print('Khong phat hien bi sau')

cv2.imshow("Ket qua nhan dien", img_main)
cv2.waitKey(0)  # Nhấn phím bất kỳ để đóng cửa sổ
cv2.destroyAllWindows()