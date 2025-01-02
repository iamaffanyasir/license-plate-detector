import cv2
import numpy as np
import pytesseract
import time
import arabic_reshaper
from bidi.algorithm import get_display

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def enhance_plate_image(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None
    
    height, width = plate_img.shape[:2]
    if width < 150:
        scale = 150 / width
        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    return morph

def get_detection_region(frame):
    height, width = frame.shape[:2]
    zone_width = width // 3  
    x1 = (width - zone_width) // 2  
    x2 = x1 + zone_width
    y1 = (height * 3) // 4  
    y2 = height - 20
    return (x1, y1, x2, y2)

def detect_license_plates(frame, region_coords):
    x1, y1, x2, y2 = region_coords
    plates = []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = w * h
            
            if 2.0 <= aspect_ratio <= 5.5 and area > 1000:
                plate_image = frame[y:y+h, x:x+w]
                if plate_image is not None and plate_image.size > 0:
                    enhanced_plate = enhance_plate_image(plate_image)
                    if enhanced_plate is not None:
                        plates.append({
                            'bbox': (x, y, x+w, y+h),
                            'image': enhanced_plate,
                            'original': plate_image
                        })
    
    return plates

def read_license_plate(plate_image):
    if plate_image is None:
        return None
    
    try:
        configs = [
            '--oem 3 --psm 7 -l eng+ara -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 8 -l eng+ara -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 6 -l eng+ara -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]
        
        best_text = None
        max_confidence = 0
        
        for config in configs:
            data = pytesseract.image_to_data(plate_image, config=config, output_type=pytesseract.Output.DICT)
            
            for i, text in enumerate(data['text']):
                conf = int(data['conf'][i])
                text = text.strip()
                
                if text and conf > max_confidence:
                    filtered_text = ''.join(c for c in text if c.isalnum())
                    
                    if filtered_text and 4 <= len(filtered_text) <= 10:
                        has_number = any(c.isdigit() for c in filtered_text)
                        if has_number and conf > 30:
                            best_text = filtered_text
                            max_confidence = conf
        
        if best_text:
            try:
                reshaped_text = arabic_reshaper.reshape(best_text)
                bidi_text = get_display(reshaped_text)
                return bidi_text
            except:
                return best_text
                
        return None
    except:
        return None

def main():
    print("Starting camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Camera initialized successfully!")
    print("Press 'q' to quit")
    
    
    last_detection_time = 0
    detection_cooldown = 60.0  
    last_plate = None
    plate_present = False
    current_plate = None
    start_time = None
    exit_time = None
    exit_display_start_time = None
    elapsed_time = None
    in_cooldown = False
    cooldown_end_time = None
    valid_plate_detected = False
    last_detection_time = time.time()  
    EXIT_GRACE_PERIOD = 2.0  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        display_frame = frame.copy()
        region_coords = get_detection_region(frame)
        x1, y1, x2, y2 = region_coords
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        current_time = time.time()
        
        
        if cooldown_end_time and current_time < cooldown_end_time:
            remaining_cooldown = int(cooldown_end_time - current_time)
            cv2.putText(display_frame, f"Cooldown: {remaining_cooldown}s", 
                       (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
            cv2.imshow('License Plate Detection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        
        if not cooldown_end_time or current_time >= cooldown_end_time:
            detection_region = frame[y1:y2, x1:x2].copy()
            plates = detect_license_plates(detection_region, (0, 0, x2-x1, y2-y1))
            
            if plates:
                last_detection_time = current_time  
                
                if not plate_present:
                    plate_present = True
                    current_plate = None

                for plate in plates:
                    plate_text = read_license_plate(plate['image'])
                    if plate_text:  
                        if not valid_plate_detected:  
                            start_time = current_time
                            valid_plate_detected = True
                            
                        if plate_text != current_plate:
                            current_plate = plate_text
                            last_plate = plate_text
                            print(f"LICENSE PLATE DETECTED: {plate_text}")
                        
                        
                        px1, py1, px2, py2 = plate['bbox']
                        cv2.rectangle(display_frame, 
                                    (x1 + px1, y1 + py1), 
                                    (x1 + px2, y1 + py2), 
                                    (255, 0, 0), 2)
                        cv2.putText(display_frame, plate_text, 
                                  (x1 + px1, y1 + py2 + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        break

                
                if valid_plate_detected and start_time:
                    elapsed_time = int(current_time - start_time)
                    minutes = elapsed_time // 60
                    seconds = elapsed_time % 60
                    timer_text = f"Time: {minutes:02d}:{seconds:02d}"
                    cv2.putText(display_frame, timer_text, 
                              (frame.shape[1] - 200, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            
            if not plates and plate_present and valid_plate_detected and last_plate:
                time_since_last_detection = current_time - last_detection_time
                
                
                if time_since_last_detection >= EXIT_GRACE_PERIOD:
                    plate_present = False
                    valid_plate_detected = False  
                    elapsed_time = int(current_time - start_time)
                    exit_time = time.strftime('%H:%M:%S', time.localtime())
                    print(f"Plate {last_plate} left detection zone. Final time: {elapsed_time} seconds. Exit time: {exit_time}")
                    exit_display_start_time = current_time
                    cooldown_end_time = current_time + detection_cooldown
            
            
            if exit_display_start_time and current_time - exit_display_start_time <= 60 and last_plate:
                final_time_text = f"Plate {last_plate}: {elapsed_time}s in zone"
                exit_time_text = f"Exit Time: {exit_time}"
                cv2.putText(display_frame, final_time_text, 
                          (frame.shape[1] - 350, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, exit_time_text, 
                          (frame.shape[1] - 350, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('License Plate Detection', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()