import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits a dataset into training, validation, and test sets, while removing images that have missing or empty labels.

    :param images_dir: Directory containing images.
    :param labels_dir: Directory containing YOLO format label .txt files.
    :param output_dir: Root directory where 'train', 'val', and 'test' directories will be created.
    :param train_ratio: Ratio of data to be used for training.
    :param val_ratio: Ratio of data to be used for validation.
    :param test_ratio: Ratio of data to be used for testing.
    """

    # Create output directories
    output_dir = Path(output_dir)
    train_img_dir = output_dir / 'images' / 'train'
    val_img_dir = output_dir / 'images' / 'val'
    test_img_dir = output_dir / 'images' / 'test'
    train_lbl_dir = output_dir / 'labels' / 'train'
    val_lbl_dir = output_dir / 'labels' / 'val'
    test_lbl_dir = output_dir / 'labels' / 'test'

    for dir_path in [train_img_dir, val_img_dir, test_img_dir, train_lbl_dir, val_lbl_dir, test_lbl_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # List of image files
    images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Get corresponding labels by replacing image extensions with .txt
    valid_images = []
    valid_labels = []

    for image_file in images:
        label_file = Path(labels_dir) / Path(image_file).with_suffix('.txt')

        # Check if the label file exists and is not empty
        if label_file.exists() and label_file.stat().st_size > 0:
            valid_images.append(image_file)
            valid_labels.append(label_file.name)
        else:
            # Delete the image if the label file is missing or empty
            image_path = Path(images_dir) / image_file
            print(f"Deleting image and label: {image_file}, {label_file}")
            image_path.unlink(missing_ok=True)
            label_file.unlink(missing_ok=True)

    # Split data into train, val, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(valid_images, valid_labels, test_size=test_ratio, random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # Helper function to copy files
    def copy_files(file_list, source_dir, dest_dir):
        for file_name in file_list:
            src_file = Path(source_dir) / file_name
            dst_file = Path(dest_dir) / file_name
            shutil.copy(src_file, dst_file)

    # Copy training files
    copy_files(train_images, images_dir, train_img_dir)
    copy_files(train_labels, labels_dir, train_lbl_dir)

    # Copy validation files
    copy_files(val_images, images_dir, val_img_dir)
    copy_files(val_labels, labels_dir, val_lbl_dir)

    # Copy test files
    copy_files(test_images, images_dir, test_img_dir)
    copy_files(test_labels, labels_dir, test_lbl_dir)

    print(f"Dataset split complete. Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

if __name__ == "__main__":
    images_dir = "/teamspace/studios/this_studio/cropped_images"  # Directory containing images
    labels_dir = "/teamspace/studios/this_studio/cropped_labels"  # Directory containing YOLO .txt files
    output_dir = "/teamspace/studios/this_studio/crop_dataset"  # Output directory where 'train', 'val', and 'test' directories will be created

    split_dataset(images_dir, labels_dir, output_dir)
