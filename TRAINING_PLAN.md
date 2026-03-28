# Training Plan

Target animal classes for the next custom model:

1. dog
2. cat
3. cow
4. horse
5. deer
6. elephant
7. zebra
8. giraffe
9. tiger
10. lion

Notes:

- The current pretrained YOLOv8m model can only detect a subset of these classes.
- `deer`, `tiger`, and `lion` require a custom-trained model because they are not available as direct classes in the current COCO-based detector.
- The backend health endpoint exposes both the current supported classes and the long-term target classes.

Training-ready files:

- `datasets/animals10/data.yaml`
- `backend/train_animals10.py`

Run training with:

```powershell
cd "C:\Users\HOTSPOT\OneDrive\Desktop\Animal detection\backend"
python train_animals10.py --epochs 100 --imgsz 960 --batch 8 --device cpu
```
