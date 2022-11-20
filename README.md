# Credit and Acknowledgement
This person reidentification repository is built upon
1. yolov5
https://github.com/ultralytics/yolov5
2. On the Unreasonable Effectiveness of Centroids in Image Retrieval
https://arxiv.org/pdf/2104.13643v1.pdf

# How to setup:
1. git clone https://github.com/shihao28/person_reid.git
2. cd person_reid
3. python3 -m venv venv
4. pip install -r requirements.txt

# Download pretrained weights
1. Download pretrained weights of resnet50 from
https://github.com/mikwieczorek/centroids-reid (Under Section Getting Started)
2. Download pretrained weights of yolov5 official repository from
https://github.com/ultralytics/yolov5 and place it within weights/yolov5
3. Download pretrained weights of CTL from
https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK
and place it within weights/ctl

# Download sample videos
1. Download sample videos from
https://www.pexels.com/video/people-in-a-park-3105196/
and place it within videos/inputs

# Inference:
1. source venv/bin/activate
2. python3 person_reid_main.py --source <path/to/your/video.mp4>
