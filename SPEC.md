# Animal Detection Web Application - Specification

## 1. Project Overview

**Project Name:** Animal Detection App  
**Type:** Full-stack Web Application  
**Core Functionality:** Upload images to detect animals using YOLOv8 model, display bounding boxes with confidence scores  
**Target Users:** Wildlife researchers, pet owners, nature enthusiasts

---

## 2. Technology Stack

### Frontend
- **Framework:** React 18 with Vite
- **Styling:** Tailwind CSS
- **HTTP Client:** Axios
- **Icons:** Lucide React

### Backend
- **Framework:** Flask (Python)
- **ML Model:** YOLOv8 (Ultralytics)
- **Image Processing:** Pillow, OpenCV
- **CORS:** Flask-CORS

---

## 3. Folder Structure

```
animal-detection/
├── backend/
│   ├── app.py                 # Flask application
│   ├── requirements.txt      # Python dependencies
│   ├── models/               # YOLOv8 model files
│   └── uploads/              # Temporary image storage
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── DetectionResults.jsx
│   │   │   └── Header.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── SPEC.md
└── README.md
```

---

## 4. API Design

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Upload image and get detections |
| GET | `/api/health` | Health check endpoint |

### Request/Response Format

**POST /api/detect**
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "success": true,
  "image_url": "/uploads/filename.jpg",
  "detections": [
    {
      "class": "dog",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## 5. UI/UX Specification

### Color Palette
- **Background:** #0f172a (slate-900)
- **Card Background:** #1e293b (slate-800)
- **Primary Accent:** #22c55e (green-500)
- **Secondary:** #3b82f6 (blue-500)
- **Text Primary:** #f8fafc (slate-50)
- **Text Secondary:** #94a3b8 (slate-400)
- **Border:** #334155 (slate-700)

### Typography
- **Font Family:** Inter (Google Fonts)
- **Headings:** 700 weight
- **Body:** 400 weight

### Layout
- Max container width: 1200px
- Responsive grid: 1 column mobile, 2 columns desktop
- Card border-radius: 12px
- Spacing unit: 4px base

### Components

1. **Header**
   - App title with animal icon
   - Subtle gradient text effect

2. **Image Upload Zone**
   - Drag-and-drop area with dashed border
   - File input fallback
   - Preview after selection
   - Loading spinner during upload

3. **Results Display**
   - Image with bounding boxes overlay
   - Detection cards showing:
     - Animal class name
     - Confidence percentage (progress bar)
     - Bounding box coordinates

---

## 6. Acceptance Criteria

- [ ] Frontend loads without errors
- [ ] Image upload works via drag-drop and file picker
- [ ] Backend processes image with YOLOv8
- [ ] Bounding boxes displayed on image
- [ ] Confidence scores shown for each detection
- [ ] Responsive design works on mobile/desktop
- [ ] Error handling for invalid images
- [ ] Health check endpoint responds

---

## 7. Deployment Steps

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Production Build
```bash
# Frontend
npm run build

# Backend (Gunicorn)
pip install gunicorn
gunicorn -w 4 app:app
```