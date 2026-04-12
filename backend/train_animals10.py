from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT_DIR / "datasets" / "animals10" / "data.yaml"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "yolov8m.pt"
DEFAULT_PROJECT_DIR = ROOT_DIR / "runs"
TARGET_ANIMAL_CLASSES = [
    "dog",
    "cat",
    "cow",
    "horse",
    "deer",
    "elephant",
    "zebra",
    "giraffe",
    "tiger",
    "lion",
]


def ensure_dataset_config():
    dataset_root = (ROOT_DIR / "datasets" / "animals10").resolve().as_posix()
    names_block = "\n".join(
        f"  {index}: {name}"
        for index, name in enumerate(TARGET_ANIMAL_CLASSES)
    )
    DATASET_YAML.write_text(
        (
            f"path: {dataset_root}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
            f"{names_block}\n"
        ),
        encoding="utf-8",
    )


def validate_dataset_layout():
    ensure_dataset_config()
    required_paths = [
        DATASET_YAML,
        ROOT_DIR / "datasets" / "animals10" / "images" / "train",
        ROOT_DIR / "datasets" / "animals10" / "images" / "val",
        ROOT_DIR / "datasets" / "animals10" / "labels" / "train",
        ROOT_DIR / "datasets" / "animals10" / "labels" / "val",
    ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Training dataset layout is incomplete. Missing:\n- "
            + "\n- ".join(missing_paths)
        )


def build_parser():
    parser = ArgumentParser(
        description="Train a custom YOLO model for the 10 selected animal classes."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--name", default="animals10-yolov8m")
    return parser


def main():
    validate_dataset_layout()
    args = build_parser().parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
