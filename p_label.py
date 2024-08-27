import os

def filter_class_labels(input_dir, output_dir, target_class_id=0):
    """
    Filters out all class labels except for the specified target_class_id (default is 0) in YOLO format files.

    :param input_dir: Directory containing the original label files.
    :param output_dir: Directory to save the filtered label files.
    :param target_class_id: The class ID to keep (default is 0).
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all label files in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read the original label file
        with open(input_path, 'r') as infile:
            lines = infile.readlines()

        # Filter the lines for the target class id
        filtered_lines = [line for line in lines if line.startswith(f"{target_class_id} ")]

        # If there are any remaining lines, write them to the output file
        if filtered_lines:
            with open(output_path, 'w') as outfile:
                outfile.writelines(filtered_lines)
        else:
            # If no class 0 is present, delete the label file
            if os.path.exists(output_path):
                os.remove(output_path)

    print(f"Filtering complete. Labels saved to {output_dir}.")


# Example usage
input_labels_dir = "/teamspace/studios/this_studio/mlabels"  # Input labels folder path
output_labels_dir = "/teamspace/studios/this_studio/plabels"  # Output folder path for filtered labels
filter_class_labels(input_labels_dir, output_labels_dir, target_class_id=0)
