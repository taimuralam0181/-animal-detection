import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


REQUIRED_SPLITS = ('train', 'val')
ALL_SPLITS = ('train', 'val', 'test')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def find_dataset_root(extracted_dir: Path) -> Path | None:
    for candidate in [extracted_dir] + list(extracted_dir.rglob('*')):
        if not candidate.is_dir():
            continue

        if all((candidate / 'images' / split).exists() and (candidate / 'labels' / split).exists()
               for split in REQUIRED_SPLITS):
            return candidate
    return None


def collect_pairs(root: Path, split: str):
    image_dir = root / 'images' / split
    label_dir = root / 'labels' / split
    for image_path in sorted(image_dir.glob('*')):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f'{image_path.stem}.txt'
        if not label_path.exists():
            continue
        yield split, image_path, label_path


def rebuild_dataset(source_root: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'images').mkdir(exist_ok=True)
    (destination / 'labels').mkdir(exist_ok=True)

    for split in ALL_SPLITS:
        (destination / 'images' / split).mkdir(exist_ok=True, parents=True)
        (destination / 'labels' / split).mkdir(exist_ok=True, parents=True)

    total_pairs = 0
    missing_labels = 0

    for split in REQUIRED_SPLITS:
        split_pairs = 0
        for _, image_path, label_path in collect_pairs(source_root, split):
            dest_image = destination / 'images' / split / image_path.name
            dest_label = destination / 'labels' / split / label_path.name
            shutil.copy2(image_path, dest_image)
            shutil.copy2(label_path, dest_label)
            total_pairs += 1
            split_pairs += 1

        expected_images = len(list((source_root / 'images' / split).glob('*')))
        missing_labels += max(0, expected_images - split_pairs)

    return total_pairs, missing_labels


def write_zip(source_dir: Path, output_path: Path):
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source_dir.rglob('*')):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(source_dir))


def main():
    parser = argparse.ArgumentParser(description='Normalize a YOLO dataset ZIP for upload.')
    parser.add_argument('input', type=Path, help='Path to the original ZIP file.')
    parser.add_argument('output', type=Path, nargs='?', default=Path('normalized_dataset.zip'),
                        help='Destination ZIP path (default: normalized_dataset.zip).')
    args = parser.parse_args()

    if not args.input.exists():
        print('Input file does not exist.', file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(args.input, 'r') as archive:
            archive.extractall(tmp_path)

        dataset_root = find_dataset_root(tmp_path)
        if dataset_root is None:
            print('Could not locate the dataset root containing images/labels splits.', file=sys.stderr)
            sys.exit(1)

        normalized_dir = tmp_path / 'normalized'
        total_pairs, missing = rebuild_dataset(dataset_root, normalized_dir)

        if total_pairs == 0:
            print('No image/label pairs found. Ensure each image has a matching .txt label.', file=sys.stderr)
            sys.exit(1)

        write_zip(normalized_dir, args.output)

        print(f'Normalized ZIP written to {args.output}')
        print(f'Collected {total_pairs} image/label pairs.')
        if missing:
            print(f'{missing} images were skipped because their labels were missing.')


if __name__ == '__main__':
    main()
