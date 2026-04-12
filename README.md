# Animal Detection

This app now opens in simple local mode by default. Cloud providers remain optional backend integrations.

## Run

Backend:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\backend"
pip install -r requirements.txt
python app.py
```

Frontend:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\frontend"
npm install
npm run dev
```

Open `http://localhost:5173`.

You can also upload a YOLO dataset ZIP from the app UI. The backend now normalizes and restructures malformed layouts automatically before importing (it still expects matching `images/train|val` + `labels/train|val`). Valid image/label pairs are merged into `datasets/animals10`, and the UI now includes a `Start Training` button for launching background training jobs. The default dataset ZIP limit is 12 GB.

## Default Mode

- `local`: bundled YOLO + CLIP pipeline, no cloud setup required

## Optional Cloud Providers

- `google-cloud-vision`: requires `google-cloud-vision` credentials
- `aws-rekognition`: uses standard AWS credentials, optionally with Custom Labels
- `azure-custom-vision`: requires a published Azure Custom Vision detector
- `seek-inaturalist`: external/manual option only

## Cloud Config

See `backend/.env.example` for the environment variables used by the backend.

Important note: the official iNaturalist API docs expose observations APIs, but not a public hosted image-prediction endpoint for direct backend integration. In this app, `Seek / iNaturalist` is added as an external/manual option rather than an automatic API detector.
