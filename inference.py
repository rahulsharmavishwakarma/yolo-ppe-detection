import os
import cv2
from pathlib import Path
import argparse
from ultralytics import YOLO


def draw_bounding_boxes(image, bboxes, class_names, confidences):
    """
    Draw bounding boxes and class names on an image using OpenCV.

    :param image: Image on which to draw bounding boxes.
    :param bboxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
    :param class_names: List of class names corresponding to the bounding boxes.
    :param confidences: List of confidence scores corresponding to the bounding boxes.
    :return: Image with bounding boxes drawn.
    """
    for bbox, class_name, conf in zip(bboxes, class_names, confidences):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text = f'{class_name} {conf:.2f}'
        cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def perform_inference(image_dir, person_model_weights, ppe_model_weights, output_dir):
    """
    Perform inference on images, first detecting persons and then detecting PPE on cropped persons.

    :param image_dir: Directory containing input images.
    :param person_model_weights: Weights for YOLOv8 person detection model.
    :param ppe_model_weights: Weights for YOLOv8 PPE detection model.
    :param output_dir: Directory to save output images with drawn bounding boxes.
    """
    # Load YOLOv8 models
    person_model = YOLO(person_model_weights)
    ppe_model = YOLO(ppe_model_weights)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the input directory
    for image_path in Path(image_dir).glob("*.jpg"):  # Assuming images are .jpg
        # Load image
        image = cv2.imread(str(image_path))

        # Perform person detection
        results_person = person_model.predict(source=image)

        # Extract person detection results
        for result in results_person:
            bboxes_person = result.boxes.xyxy.cpu().numpy()
            confidences_person = result.boxes.conf.cpu().numpy()
            class_ids_person = result.boxes.cls.cpu().numpy()

            # Get class names using class IDs
            class_names_person = [result.names[int(class_id)] for class_id in class_ids_person]

            # For each detected person, crop and perform PPE detection
            for i, bbox_person in enumerate(bboxes_person):
                # Get bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, bbox_person[:4])
                
                # Crop the image around the detected person
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Perform PPE detection on the cropped image
                results_ppe = ppe_model.predict(source=cropped_image)

                # Extract PPE detection results
                for ppe_result in results_ppe:
                    bboxes_ppe = ppe_result.boxes.xyxy.cpu().numpy()
                    confidences_ppe = ppe_result.boxes.conf.cpu().numpy()
                    class_ids_ppe = ppe_result.boxes.cls.cpu().numpy()

                    # Get class names using class IDs
                    class_names_ppe = [ppe_result.names[int(class_id)] for class_id in class_ids_ppe]

                    # Convert PPE bounding boxes back to full image coordinates
                    for bbox_ppe, class_name_ppe, confidence_ppe in zip(bboxes_ppe, class_names_ppe, confidences_ppe):
                        ppe_x_min = int(bbox_ppe[0]) + x_min
                        ppe_y_min = int(bbox_ppe[1]) + y_min
                        ppe_x_max = int(bbox_ppe[2]) + x_min
                        ppe_y_max = int(bbox_ppe[3]) + y_min

                        # Draw the PPE bounding boxes on the full image
                        cv2.rectangle(image, (ppe_x_min, ppe_y_min), (ppe_x_max, ppe_y_max), (0, 255, 255), 2)
                        cv2.putText(image, f'{class_name_ppe} {confidence_ppe:.2f}', (ppe_x_min, ppe_y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Save the image with bounding boxes to the output directory
        output_image_path = output_dir / image_path.name
        cv2.imwrite(str(output_image_path), image)

        print(f"Saved inference result to {output_image_path}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Perform inference on images using YOLOv8 models for person and PPE detection.")
    
    parser.add_argument("input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save output images with bounding boxes.")
    parser.add_argument("person_det_model", type=str, help="Path to YOLOv8 weights for person detection model.")
    parser.add_argument("ppe_detection_model", type=str, help="Path to YOLOv8 weights for PPE detection model.")
    
    args = parser.parse_args()

    # Perform inference
    perform_inference(args.input_dir, args.person_det_model, args.ppe_detection_model, args.output_dir)

# python inference.py /teamspace/studios/this_studio/ppe_dataset/images/test /teamspace/studios/this_studio/results /teamspace/studios/this_studio/runs/detect/train/weights/last.pt /teamspace/studios/this_studio/runs/detect/train6/weights/last.pt


