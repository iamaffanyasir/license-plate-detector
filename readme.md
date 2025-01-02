License Plate Detection System
This project leverages OpenCV, Tesseract OCR, and Python to detect and read license plates from live camera footage. It highlights license plates, processes them for optimal readability, and reads both English and Arabic characters. It includes cooldown and detection grace periods, with a display of the detection time and exit time when the plate leaves the detection zone.

Features
Live Camera Feed: Detects license plates in real-time using your webcam.
Plate Detection: Identifies rectangular plates in the designated detection region.
OCR for Plate Recognition: Reads alphanumeric characters on the plates using Tesseract OCR.
Supports Multiple Languages: Detects plates with English and Arabic characters.
Cooldown Mechanism: Implements a cooldown period to avoid redundant detections.
Timer: Displays a timer showing how long the license plate remains within the detection zone.
Grace Period: Provides a grace period before marking the plate as 'left the zone'.
Prerequisites
Python 3.x
OpenCV
Tesseract OCR
arabic_reshaper
bidi.algorithm
To install the required Python libraries, run:

bash
Copy code
pip install opencv-python pytesseract arabic-reshaper python-bidi numpy
Additionally, ensure that Tesseract OCR is installed. You can download it from here and set the path in the code:

python
Copy code
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
How It Works
Camera Setup: The program uses OpenCV to open the webcam, capture frames, and display the live feed.
License Plate Detection: It scans the frames for rectangular contours that likely represent license plates.
Image Enhancement: Plates are processed with CLAHE and Gaussian blur to enhance readability.
OCR: Tesseract OCR is used to extract text from the detected plates.
Text Post-Processing: The detected text is filtered, reshaped (for Arabic), and displayed.
Cooldown and Grace Period: After detecting a plate, the program waits for a set cooldown period before allowing another detection.
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/license-plate-detection.git
Navigate to the project directory:

bash
Copy code
cd license-plate-detection
Run the script:

bash
Copy code
python detect_license_plate.py
The camera will start and detect license plates within the frame. Press q to quit the program.

Output
The detected license plate text will be displayed on the screen in real-time.
The time spent by the plate in the detection zone will be shown along with its exit time when it leaves the zone.
When no plate is detected, a cooldown period will initiate before allowing the next detection.
License
This project is licensed under the MIT License - see the LICENSE file for details.