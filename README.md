# Animal Detection

This project runs locally by default with a bundled detection pipeline. You can:

- detect animals from images
- upload a custom YOLO dataset ZIP
- start training from the app UI

Cloud providers are optional. Normal use does not require Google, AWS, or Azure.

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm

## First-Time Setup

Backend:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\backend"
pip install -r requirements.txt
```

Frontend:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\frontend"
npm install
```

## Normal Run

Open 2 terminals.

Terminal 1, backend:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\backend"
python app.py
```

Terminal 2, frontend:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\frontend"
npm run dev
```

Then open:

```text
http://localhost:5173
```

Important:

- keep both terminals running
- local mode does not use online APIs
- if dependencies are already installed, later you only need `python app.py` and `npm run dev`

## Image Detection Flow

1. Open `http://localhost:5173`
2. In the `Upload Image` section, choose an image
3. Wait for detection to finish
4. Check the result image, detected class, and confidence

If detection does not work:

- make sure backend is running on `http://localhost:5000`
- refresh the browser
- check the backend terminal for the real error

## Dataset Upload Flow

The app supports uploading a YOLO dataset ZIP directly from the UI.

Go to the `Dataset` section on the main page and click `Upload ZIP`.

The ZIP should contain:

```text
images/train/
images/val/
labels/train/
labels/val/
```

Optional:

```text
images/test/
labels/test/
```

Rules:

- every image must have a matching `.txt` label file
- image name and label name must match
- labels must be in YOLO format:

```text
class_id x_center y_center width height
```

Example:

```text
images/train/dog1.jpg
labels/train/dog1.txt
images/val/dog2.jpg
labels/val/dog2.txt
```

## Upload Progress and Status

After upload:

- the UI shows upload percent
- a success message shows how many files were imported
- `train` and `val` image/label counts should become non-zero

If counts stay zero, the upload did not complete correctly.

## Automatic Normalize

If the ZIP structure is slightly wrong, the backend tries to normalize it automatically before importing.

If the ZIP is still too messy, you can manually normalize it:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection"
python scripts\normalize_dataset_zip.py "C:\path\to\your.zip" "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\normalized.zip"
```

Then upload `normalized.zip` from the UI.

## Training Flow

Training starts from the same `Dataset` card.

Requirements before training:

- `train` split must have images and labels
- `val` split must have images and labels
- backend must still be running

Steps:

1. Upload dataset ZIP
2. Wait until counts appear in the UI
3. Click `Start Training`
4. Watch the `Training Log` box
5. Keep backend terminal open until training finishes

Notes:

- CPU training can take a long time
- large datasets can take many hours
- do not close the backend while training is active

## Where Data Goes

Uploaded dataset files are merged into:

```text
datasets/animals10/
```

Training logs are written under:

```text
runs/logs/
```

## Common Problems

### `YOLO dataset layout not found`

Meaning:

- your ZIP does not contain the required `images/...` and `labels/...` structure

Fix:

- rebuild the ZIP with the correct folders
- or run `scripts/normalize_dataset_zip.py`

### `Failed to load dataset status`

Meaning:

- frontend could not read `/api/datasets/status`

Fix:

- make sure backend is still running
- refresh the page
- check backend terminal for error output

### `Request failed with status code 500`

Meaning:

- backend crashed or training hit an internal error

Fix:

- look at the backend terminal
- copy the traceback
- fix the dataset structure or training config, then retry

### Vite proxy `ECONNRESET`

Meaning:

- frontend could not reach the backend

Fix:

- restart backend with `python app.py`
- refresh `http://localhost:5173`

## Dataset ZIP Limit

Default dataset ZIP upload limit:

```text
12 GB
```

You can change it in `backend/.env.example` using:

```text
MAX_DATASET_UPLOAD_BYTES
```

## Optional Cloud Providers

Default mode:

- `local`: bundled YOLO + CLIP, no cloud setup required

Optional:

- `google-cloud-vision`
- `aws-rekognition`
- `azure-custom-vision`
- `seek-inaturalist`

Cloud settings are documented in [backend/.env.example](<C:/Users/HOTSPOT/OneDrive/Desktop/Animal detection/backend/.env.example>).

Important:

- local mode does not send images to online APIs
- `Seek / iNaturalist` is kept as an external/manual option, not a direct hosted prediction API

## Quick Summary

Normal daily flow:

1. Run backend
2. Run frontend
3. Open `http://localhost:5173`
4. Detect image or upload dataset ZIP
5. Start training from the UI if dataset is ready
