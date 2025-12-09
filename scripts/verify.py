#!/usr/bin/env python3
"""Visualize collected dataset with bounding boxes"""
import cv2
import os
import sys
import random

def draw_yolo_box(img, class_id, x_center, y_center, width, height):
    """Draw YOLO format bbox on image"""
    h, w = img.shape[:2]
    
    # Convert normalized to pixel coords
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    
    # Class colors
    colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}  # Red, Yellow, Green
    names = {0: 'red', 1: 'yellow', 2: 'green'}
    
    color = colors.get(class_id, (255, 255, 255))
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, names[class_id], (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img

def main():
    if len(sys.argv) < 2:
        print("Usage: verify_dataset.py <dataset_dir>")
        print("Example: ./verify_dataset.py ~/Desktop/TrafficSenseAI/yolo_dataset")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    img_dir = os.path.join(dataset_dir, 'images', 'train')
    lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
    
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} not found")
        sys.exit(1)
    
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    if not images:
        print("No images found!")
        sys.exit(1)
    
    print(f"Found {len(images)} images. Press 'q' to quit, any other key for next image.")
    
    # Show random sample
    random.shuffle(images)
    
    for img_file in images[:30]:  # Show up to 30 samples
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.replace('.jpg', '.txt'))
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Draw annotations
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:])
                        img = draw_yolo_box(img, class_id, x_c, y_c, w, h)
        
        cv2.imshow('Dataset Verification', img)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("Verification complete!")

if __name__ == '__main__':
    main()