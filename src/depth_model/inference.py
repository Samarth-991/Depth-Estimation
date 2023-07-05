import torch
from PIL import Image
import os
import cv2 
import numpy as np
try:
    import yolov7
except ImportError as err:
    os.system("pip install yolov7detect")
    import yolov7

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from utils.data_transforms import pre_process
from utils.load_tof_images import create_from_zip_absolute  as load_assignment_data
from data_loader.data_loader_assignment import CreateAssignemntDataset
from depth_model.model import PTModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL = yolov7.load('kadirnar/yolov7-tiny-v0.1', hf_model=True)

IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained("vinvino02/glpn-nyu")
DEPTH_MODEL = AutoModelForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

def detect_child(image):
    # load pretrained or custom model
    
    meta_info = None
    # set model parameters
    YOLO_MODEL.conf = 0.25  # NMS confidence threshold
    YOLO_MODEL.iou = 0.45  # NMS IoU threshold
    YOLO_MODEL.classes = None  # (optional list) filter by class

    results = YOLO_MODEL([image])
    predictions = results.pred[0]
    category = predictions[:, 5].numpy()
    if category[0] == 0:
        boxes = predictions[:, :4].numpy() # x1, y1, x2, y2
        # scores = predictions[:, 4].numpy()
        # categories = predictions[:, 5].numpy()
        meta_info=  boxes.astype(np.int32)[0] 
    return meta_info 
        

def predict_with_opensource_model(image,model,processor,image_size=(240,320)):
    pixel_values = processor(image, return_tensors="pt").pixel_values   
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_size,
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.numpy()
    return output


def tensor_to_numpy_image(image,rgb=True):
    image = image.permute(1,0,2,3).squeeze(axis=0).cpu().detach().numpy()
    if rgb:
        image = image.squeeze(axis=1)
        return np.array(image.transpose(1, 2, 0)*255, dtype=np.uint8) 
    else:
        return image.squeeze(axis=0)


def predict_with_trained_model(test_image,model):
    model.eval()
    test_image = cv2.resize(test_image ,(480,640),interpolation=cv2.INTER_LINEAR)
    test_image = torch.from_numpy(np.transpose(test_image, (2, 0, 1)))
    test_image = test_image.float().to(DEVICE)
    test_image = torch.unsqueeze(test_image,dim=0) # torch.Size([1, 3, 480, 640])
    
    y_pred = model(test_image)
    y_pred = tensor_to_numpy_image(y_pred,rgb=False)
    return y_pred


def inference_rgbimage(rgb_image,depth_image_size=(240,320),depth_scale=0.01,checkpoint="vinvino02/glpn-nyu"):
    result = np.zeros_like(rgb_image)
    if "vinvino02/glpn-nyu" in checkpoint:
        result  = predict_with_opensource_model(rgb_image,
                                                model=DEPTH_MODEL,
                                                processor=IMAGE_PROCESSOR,
                                                image_size=depth_image_size)

    else:
        model = PTModel().float().to(DEVICE)
        print("loading checkpoints from {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint))
        result = predict_with_trained_model(rgb_image,model)

    formatted = (result*255).astype(np.float32)
    return formatted*depth_scale

