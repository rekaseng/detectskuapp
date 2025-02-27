
# YOLO Detection App - Inference Guide



## Run Inference Locally

1. Clone the repository

```bash
  git clone https://github.com/rekaseng/detectskuapp.git
  cd detectskuapp
```
2. Update the model file path
Open main.py. Modify the model path to point to the correct YOLO model file used for inferencing.

3. Run the app

```bash
  python main.py
```

## Export Annotations to CVAT

1. Create a task in CVAT
Upload a video file for inferencing to CVAT.

2. Clone the repository

```bash
  git clone https://github.com/rekaseng/detectskuapp.git
  cd detectskuapp
```

3. Update file paths in export_cvat.py
Open export_cvat.py.
Set the correct video file path and model path for inferencing.

4. Run the export script

```bash
  python export_cvat.py
```
5. Upload the generated annotations
Locate the output.xml file generated.
Upload it to the task dataset already created in CVAT.





    