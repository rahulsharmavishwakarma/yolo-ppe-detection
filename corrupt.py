import os

def fix_labels_and_remove_empty(label_dir, image_dir):
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, label_file.replace(".txt", ".jpg"))
        
        # Check if the label file is empty
        if os.path.getsize(label_path) == 0:
            print(f"Empty label file found: {label_path}, deleting...")
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)  # Delete corresponding image
            continue
        
        with open(label_path, 'r') as file:
            lines = file.readlines()

        fixed_lines = []
        for line in lines:
            values = line.strip().split()
            if len(values) != 5:  # Each label line should have 5 values (class_id, bbox)
                continue  # Skip corrupt lines
            
            class_id = values[0]
            bbox = list(map(float, values[1:5]))

            # Check if bbox values are normalized
            if all(0 <= v <= 1 for v in bbox):
                fixed_lines.append(line)
            else:
                print(f"Found non-normalized or out-of-bounds bbox in {label_path}")

        # If no valid lines are left or file is empty, delete the file and corresponding image
        if not fixed_lines:
            print(f"Deleting corrupt or empty label file and its image: {label_path}")
            os.remove(label_path)
            if os.path.exists(image_path):
                os.remove(image_path)  # Delete corresponding image
        else:
            # Overwrite the label file with fixed lines
            with open(label_path, 'w') as file:
                file.writelines(fixed_lines)

# Define the directories for validation images and labels
label_dir = "/teamspace/studios/this_studio/dataset/labels/val"
image_dir = "/teamspace/studios/this_studio/dataset/images/val"
fix_labels_and_remove_empty(label_dir, image_dir)
