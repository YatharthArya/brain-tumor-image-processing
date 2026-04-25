import cv2
import numpy as np
import os

# Choose folder to test
input_folder = 'sample_images'  # or 'dataset/no'

# Get all image files
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

for i, filename in enumerate(image_files):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))

    # === STEP 1: Show Original Image ===
    cv2.imshow("Step 1: Original MRI", img)
    print(f"\n[{i+1}] {filename} - Step 1: Original MRI")
    cv2.waitKey(0)  # Wait until key press

    # === STEP 2: Grayscale + Blur ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Step 2: Blurred Image", blur)
    print(f"[{i+1}] Step 2: Blurred Image")
    cv2.waitKey(0)

    # === STEP 3: Threshold + Contour Detection ===
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

    contour_img = img.copy()
    if valid_contours:
        cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 2)
        tumor_status = "Yes"
    else:
        tumor_status = "No"

    cv2.imshow("Step 3: Tumor Detection", contour_img)
    print(f"[{i+1}] Step 3: Tumor Detected: {tumor_status}")
    cv2.waitKey(0)

    # === Close all windows before next image ===
    cv2.destroyAllWindows()

    # Optional: Ask user to continue or exit
    cont = input("Press Enter for next image or type 'q' to quit: ")
    if cont.lower() == 'q':
        break

