import os
from pathlib import Path
from PIL import Image

def load_annotations(label_path):
    """
    Loads annotations from a YOLOv8 format label file.
    
    :param label_path: Path to the YOLOv8 format label file.
    :return: A list of tuples representing class_id and bounding boxes (x_center, y_center, width, height).
    """
    annotations = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id = int(data[0])
            bbox = list(map(float, data[1:]))
            annotations.append((class_id, bbox))
    return annotations

def normalize_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Normalize bounding box coordinates to YOLO format where values are between 0 and 1.

    :param x_center: Center x-coordinate of the bounding box.
    :param y_center: Center y-coordinate of the bounding box.
    :param width: Width of the bounding box.
    :param height: Height of the bounding box.
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :return: Normalized bounding box (x_center_norm, y_center_norm, width_norm, height_norm).
    """
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return x_center_norm, y_center_norm, width_norm, height_norm

def crop_and_save_person_images(image_path, label_path, output_images_dir, output_labels_dir):
    """
    Crops and saves single person images with updated and normalized PPE annotations.
    
    :param image_path: Path to the original image.
    :param label_path: Path to the YOLOv8 format label file.
    :param output_images_dir: Directory where cropped images will be saved.
    :param output_labels_dir: Directory where the corresponding updated labels will be saved.
    """
    # Load the original image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Load the annotations
    annotations = load_annotations(label_path)

    # Separate person bounding boxes (class_id = 0) and PPE bounding boxes (other class_ids)
    person_bboxes = [bbox for class_id, bbox in annotations if class_id == 0]
    ppe_annotations = [(class_id, bbox) for class_id, bbox in annotations if class_id != 0]

    # Process each person bounding box
    for i, person_bbox in enumerate(person_bboxes):
        # Convert YOLO format to pixel format for cropping
        person_x_center, person_y_center, person_width, person_height = person_bbox
        person_x_min = int((person_x_center - person_width / 2) * img_width)
        person_x_max = int((person_x_center + person_width / 2) * img_width)
        person_y_min = int((person_y_center - person_height / 2) * img_height)
        person_y_max = int((person_y_center + person_height / 2) * img_height)

        # Crop the image to the person bounding box
        cropped_img = img.crop((person_x_min, person_y_min, person_x_max, person_y_max))
        cropped_img_width, cropped_img_height = cropped_img.size

        # Adjust and normalize PPE annotations for the cropped image
        cropped_ppe_annotations = []
        for class_id, ppe_bbox in ppe_annotations:
            ppe_x_center, ppe_y_center, ppe_width, ppe_height = ppe_bbox
            
            # Convert YOLO format to pixel format
            ppe_x_center_px = ppe_x_center * img_width
            ppe_y_center_px = ppe_y_center * img_height
            ppe_width_px = ppe_width * img_width
            ppe_height_px = ppe_height * img_height
            
            # Check if the PPE bounding box is within the person's bounding box
            if person_x_min <= ppe_x_center_px <= person_x_max and person_y_min <= ppe_y_center_px <= person_y_max:
                
                # Adjust the PPE bounding box relative to the cropped image
                new_x_center_px = ppe_x_center_px - person_x_min
                new_y_center_px = ppe_y_center_px - person_y_min

                # Normalize the bounding box to the cropped image dimensions
                new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm = normalize_bbox(
                    new_x_center_px, new_y_center_px, ppe_width_px, ppe_height_px, cropped_img_width, cropped_img_height
                )

                cropped_ppe_annotations.append((class_id, [new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm]))

        # Save the cropped image
        cropped_img_name = f"{image_path.stem}_person_{i}.jpg"
        cropped_img_path = output_images_dir / cropped_img_name
        cropped_img.save(cropped_img_path)

        # Save the updated annotations for the cropped image
        cropped_label_path = output_labels_dir / f"{cropped_img_name[:-4]}.txt"
        with open(cropped_label_path, 'w') as f:
            for class_id, bbox in cropped_ppe_annotations:
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

def process_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir):
    """
    Processes the entire dataset by converting full images with multiple persons into cropped single person images.
    
    :param images_dir: Directory containing the original images.
    :param labels_dir: Directory containing the YOLOv8 format labels.
    :param output_images_dir: Directory to save cropped person images.
    :param output_labels_dir: Directory to save updated and normalized annotations for the cropped person images.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images_dir.glob("*.jpg"):  # Assuming images are in .jpg format
        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            crop_and_save_person_images(image_path, label_path, output_images_dir, output_labels_dir)

if __name__ == "__main__":
    images_dir = "/teamspace/studios/this_studio/datasets/images"
    labels_dir = "/teamspace/studios/this_studio/mlabels"
    output_images_dir = "/teamspace/studios/this_studio/cropped_images"
    output_labels_dir = "/teamspace/studios/this_studio/cropped_labels"

    process_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir)
