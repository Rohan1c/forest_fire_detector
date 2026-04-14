import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_voc_to_yolo(xml_file, output_dir, class_mapping={'fire': 0}):
    """Convert Pascal VOC XML to YOLO format"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    filename = root.find('filename').text
    base_name = os.path.splitext(filename)[0]

    txt_file = os.path.join(output_dir, f"{base_name}.txt")

    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                continue

            class_id = class_mapping[class_name]

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # Convert to YOLO format 
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def convert_dataset(input_dir, output_dir):
    """Convert entire dataset"""
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in Path(input_dir).glob('*.xml'):
        convert_voc_to_yolo(str(xml_file), output_dir)

if __name__ == "__main__":
    convert_dataset(
        'fire-dataset/train/annotations',
        'fire-dataset/train/labels'
    )
    convert_dataset(
        'fire-dataset/validation/annotations',
        'fire-dataset/validation/labels'
    )

    print("Conversion completed!")
