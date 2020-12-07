""" Detect people wearing masks in videos
"""
from pathlib import Path

import click
import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
import torchvision
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

# from common.facedetector import FaceDetector
# from train import MaskDetector
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def plot_image_new(frame, img_tensor, annotation, xStart=None, yStart=None, block=True):
    # img_tensor = torch.from_numpy(img_tensor)
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow( np.array( img_tensor.permute(1, 2, 0) ) )
    # print(annotation)
    # print(annotation["labels"])
    frameFlag = False
    boundingBoxes = []

    for box, label in zip( annotation["boxes"], annotation["labels"] ):
        print("label",label)
        print("box",box)
        xmin, ymin, xmax, ymax = box
        # Create a Rectangle patch
        if label==2 or label == 3:
            # rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none')
            # cv2.rectangle(frame,
            #               (xmin, ymin),
            #               (xmax, ymax),
            #               (126, 65, 64),
            #               thickness=2)
            boundingBoxes.append((xmin, ymin, xmax, ymax))
        # else:
        #     rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
    #     ax.add_patch(rect)
    #     ax.axis("off")
    # plt.show(block=block)
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
    # model = MaskDetector()
    
    model = get_model_instance_segmentation(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.load_state_dict(torch.load(modelpath, map_location=device), strict=False)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model = model.to(device)
    model.eval()
    
    # faceDetector = FaceDetector(
    #     prototype='covid-mask-detector/face_detection_model/deploy.prototxt.txt',
    #     model='covid-mask-detector/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel',
    # )
    
    # transformations = Compose([
    #     ToPILImage(),
    #     Resize((100, 100)),
    #     ToTensor(),
    # ])
    
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
            # faces = faceDetector.detect(frame)
            # for face in faces:
            #     xStart, yStart, width, height = face
            #     xStart -= 100
            #     yStart -= 100
            #     width += 2*100
            #     height += 2*100
                
            #     # clamp coordinates that are outside of the image
            #     xStart, yStart = max(xStart, 0), max(yStart, 0)
                
            #     # predict mask label on extracted face
            #     faceImg = frame[yStart:yStart+height, xStart:xStart+width]

                
            #     cv2.imwrite(os.path.join(outputDir, 'images', str(img_count)+'.png'), faceImg) 
            #     img_count += 1

            #     # faceImgTranspose = np.transpose(faceImg, (2,0,1))

                
            #     faceImgTensor = data_transform(faceImg)
            #     # faceImgTensor = torch.from_numpy(faceImg)
            #     # faceImgTensor = faceImgTensor.permute(2,0,1)
            #     faceImgTensor = torch.unsqueeze(faceImgTensor, 0).to(device)

            #     output = model(faceImgTensor)
            #     boundingBoxes = plot_image_new(frame, faceImgTensor[0], output[0], xStart=xStart, yStart=yStart)

            #     # _, predicted = torch.max(output.data, 1)
                
            #     # draw face frame
            #     # cv2.rectangle(frame,
            #     #               (xStart, yStart),
            #     #               (xStart + width, yStart + height),
            #     #               (126, 65, 64),
            #     #               thickness=2)
                
            #     # # center text according to the face frame
            #     # textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            #     # textX = xStart + width // 2 - textSize[0] // 2
                
            #     # # draw prediction label
            #     # cv2.putText(frame,
            #     #             labels[predicted],
            #     #             (textX, yStart-20),
            #     #             font, 1, labelColor[predicted], 2)
            
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