import os
import albumentations as A
import torch
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_valid_transform():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2(),
    ])

class LungsAnnotationDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        super().__init__()
        self.image_dir = image_dir
        self.image_ids = os.listdir(image_dir)  # List all image files in the directory
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir, image_id)

        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((512, 512))
        image = np.array(image).astype(np.float32) / 255.0

        h, w, c = image.shape
        image = np.transpose(image, (2, 0, 1))  
        image = torch.tensor(image)

        return image, image_id  # Return image and image_id

    def __len__(self):
        return len(self.image_ids)

def get_predictions_for_image(model, image_path, device, class_brands):
    model.to(device)  # Ensure the model is on the same device as the inputs
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((512, 512))
    image_tensor = torch.tensor(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]  # Get output for a single image

    pred_boxes = output['boxes']
    labels_pred = output['labels']
    scores = output['scores'].data.cpu().numpy()

    # Filter predicted boxes based on confidence score
    valid_indices = scores >= 0.6
    boxes_pred = pred_boxes[valid_indices]
    labels_pred = labels_pred[valid_indices]

    # Convert results to a list of dictionaries
    pred_results = []
    for box, label in zip(boxes_pred, labels_pred):
        box_np = box.detach().cpu().numpy().astype(int).tolist()  # Convert tensor to list
        class_name = class_brands.get(label.item(), 'Unknown')
        pred_results.append({
            'box': box_np,
            'class_label': class_name
        })

    return image, pred_results

def draw_and_save_image_with_boxes(image, predictions, output_dir='output_folder'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    draw = ImageDraw.Draw(image)

    for prediction in predictions:
        box = prediction['box']
        class_label = prediction['class_label']
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1] - 10), class_label, fill="red")

    # Save the image with bounding boxes
    output_path = os.path.join(output_dir, 'predicted_image.jpg')
    image.save(output_path)

def main(image_path, model_path):
    num_classes = 15
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    class_brands = {
        0: 'Aortic enlargement',
        1: 'Atelectasis',
        2: 'Calcification',
        3: 'Cardiomegaly',
        4: 'Consolidation',
        5: 'ILD',
        6: 'Infiltration',
        7: 'Lung Opacity',
        8: 'Nodule/Mass',
        9: 'Other lesion',
        10: 'Pleural effusion',
        11: 'Pleural thickening',
        12: 'Pneumothorax',
        13: 'Pulmonary fibrosis'
    }

    
    image, predictions = get_predictions_for_image(model, image_path, device, class_brands)
    draw_and_save_image_with_boxes(image, predictions)
    return predictions
    # print(predictions)  # Print or use the predictions as needed

if __name__ == "__main__":
    image_path = '../../test_resized/0af8628f5cbe0786db483a10934d1be5.png'  # Replace with the path of your specific image
    fastercnn_model_path='../../x_ray_models/model_fasterRCNN_finetuned.pth'
    main(image_path, fastercnn_model_path)
