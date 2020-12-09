from pathlib import Path

import click
import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
import torchvision
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def plot_image_new(frame, img_tensor, annotation, xStart=None, yStart=None, block=True):
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow( np.array( img_tensor.permute(1, 2, 0) ) )
    frameFlag = False
    boundingBoxes = []

    for box, label in zip( annotation["boxes"], annotation["labels"] ):
        print("label",label)
        print("box",box)
        xmin, ymin, xmax, ymax = box
        # Create a Rectangle patch
        if label==2 or label == 3:            
            boundingBoxes.append((xmin, ymin, xmax, ymax))        
    return boundingBoxes

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    return model


# @click.command(help="""
#                     modelPath: path to model.ckpt\n
#                     videoPath: path to video file to annotate
#                     """)
# @click.argument('modelpath')
# @click.argument('videopath')
# @click.option('--output', 'outputPath', type=Path,
#               help='specify output path to save video with annotations')
@torch.no_grad()
def tagVideo(modelpath, videopath, outputPath=None):
    """ detect if persons in video are wearing masks or not
    """   
    model = get_model_instance_segmentation(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(modelpath, map_location=device), strict=False)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model = model.to(device)
    model.eval()

    
    data_transform = transforms.Compose([
        ToPILImage(),
        transforms.ToTensor(), 
    ])


    if outputPath:
        writer = FFmpegWriter(str(outputPath))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    labels = ['No mask', 'Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]
    img_count = 0
    outputDir = os.path.dirname(os.path.realpath(outputPath))
    frame_count = 0
    boundingBoxes = []
    for frame in vreader(str(videopath)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print('Frame:', frame_count)

        if frame_count%30==0:
            frameTensor = data_transform(frame)
            frameTensor = torch.unsqueeze(frameTensor, 0).to(device)
            output = model(frameTensor)
            boundingBoxes = plot_image_new(frame, frameTensor[0], output[0])            
            
        if len(boundingBoxes)>0:
            for bb in boundingBoxes:
                cv2.rectangle(frame,
                            (bb[0], bb[1]),
                            (bb[2], bb[3]),
                            (54, 66, 227),
                            thickness=2)

        cv2.imshow('main', frame)
        if outputPath:
            writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1
    if outputPath:
        writer.close()
    cv2.destroyAllWindows()

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    modelpath = 'saves/final_run/model_2_epochs.pt'
    videopath = 'sample_videos/sample_facemask_video_11.mov'
    outputPath = 'sample_videos_output/faster_rcnn_output.mov'
    
    tagVideo(modelpath, videopath, outputPath)