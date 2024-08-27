## Info
'activeren van python envirement'
conda activate museum_env


pip install opencv-python numpy matplotlib pillow dotenv requests openai ultralytics torch
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

python main.py


deactivate

## Projectstructuur

main.py: Het hoofdscript van het project dat de workflow van het programma aanstuurt.

vision_agent.py: Bevat functies voor het laden van het YOLO-model en het verwerken van webcambeelden.

helpers.py: Bevat hulpfuncties voor het tekenen van bounding boxes en het genereren van kleuren.

output_images/: Map waar gegenereerde kunstwerken worden opgeslagen.

.env: Bestand voor het opslaan van API-sleutels en andere omgevingsvariabelen.

## Training

Install the needed library in conda environment

yolo detect train data=data.yaml model=yolov5su.pt epochs=10 imgsz=640 batch=8 project="C:\YOLO_Project\runs

## Author

- Jasmine Van Ryckeghem