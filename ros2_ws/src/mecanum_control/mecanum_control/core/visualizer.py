import cv2

# ===== Overlay helpers =====
def draw_label_top_right(img, text, margin=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x2 = img.shape[1] - margin
    y1 = margin
    x1 = x2 - tw - 16
    y2 = y1 + th + 16
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x1 + 8, y2 - 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_labeled_box(img, box, color=(0, 0, 255), label="TARGET"):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx1, ty1 = x1, max(0, y1 - th - 10)
    tx2, ty2 = x1 + tw + 12, ty1 + th + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.putText(img, label, (tx1 + 6, ty2 - 6), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
