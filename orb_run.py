import cv2
import sys

def draw_keypoints_radius1(image, keypoints, color=(0, 255, 0), radius=1, thickness=1):
    img = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img, (x, y), radius, color, thickness)
    return img

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 orb_run_radius1.py <image>")
        sys.exit(1)

    img_path = sys.argv[1]
    frame = cv2.imread(img_path)

    if frame is None:
        print("Error: Image not found.")
        sys.exit(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    frame_kp = draw_keypoints_radius1(
        frame, keypoints, color=(0, 255, 0), radius=1, thickness=1
    )

    cv2.imshow("ORB Keypoints (radius=1)", frame_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

