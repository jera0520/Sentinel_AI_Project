
import json
import os

# FIXME: Please update this dictionary with the correct class mappings.
# The key should be the string from the JSON 'class' field,
# and the value should be the integer class ID for your YOLO model.
CLASS_MAPPING = {
    "60": 0,  # "person"
    # Add other class mappings here if they exist. For example:
    # "helmet": 1,
    # "no_helmet": 2,
    # "fallen": 3,
}

def convert_annotation(json_path, output_dir):
    """
    Converts a single JSON annotation file to the YOLO .txt format.

    Args:
        json_path (str): The absolute path to the input JSON file.
        output_dir (str): The absolute path to the directory where the .txt file will be saved.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing JSON file: {json_path}")
        print(e)
        return

    image_width = data.get("image", {}).get("resolution", [None, None])[0]
    image_height = data.get("image", {}).get("resolution", [None, None])[1]

    if not image_width or not image_height:
        print(f"Warning: Image dimensions not found in {json_path}. Skipping.")
        return

    base_filename = os.path.splitext(os.path.basename(json_path))[0]
    txt_filename = f"{base_filename}.txt"
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        annotations = data.get("annotations", [])
        for ann in annotations:
            class_id_str = ann.get("class")
            if class_id_str not in CLASS_MAPPING:
                print(f"Warning: Unknown class '{class_id_str}' in {json_path}. Skipping.")
                continue

            class_id = CLASS_MAPPING[class_id_str]

            points = ann.get("point", [])
            visible_points = [p for p in points if len(p) == 3 and p[2] != 0]

            if not visible_points:
                continue

            x_coords = [p[0] for p in visible_points]
            y_coords = [p[1] for p in visible_points]

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min

            # Normalize
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            width_norm = width / image_width
            height_norm = height / image_height

            txt_file.write(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n")

def process_directory(input_dir, output_dir):
    """
    Processes all JSON files in a directory.

    Args:
        input_dir (str): Absolute path to the directory with JSON files.
        output_dir (str): Absolute path to the directory to save .txt files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            convert_annotation(json_path, output_dir)

if __name__ == '__main__':
    # Define base directories
    train_base_dir = "/home/jera/Sentinel_AI_Project/Dataset/Data/Keypoint/1.Tranining"
    valid_base_dir = "/home/jera/Sentinel_AI_Project/Dataset/Data/Keypoint/2.Vaildation"

    # Define input and output directories
    train_json_dir = os.path.join(train_base_dir, "라벨링데이터(zip)_230328_add")
    train_label_dir = os.path.join(train_base_dir, "labels")
    valid_json_dir = os.path.join(valid_base_dir, "라벨링데이터(zip)_230328_add")
    valid_label_dir = os.path.join(valid_base_dir, "labels")

    print("Starting conversion for training data...")
    process_directory(train_json_dir, train_label_dir)
    print("Finished converting training data.")

    print("Starting conversion for validation data...")
    process_directory(valid_json_dir, valid_label_dir)
    print("Finished converting validation data.")
