from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import json
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torchvision
import numpy as np
import torch.nn.functional as F
import cv2

# Create your views here.
with open('./model/imagenet_classes.json','r') as f:
    labelInfo=f.read()


labelInfo=json.loads(labelInfo)
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
PATH = "model/covid.pth"
model.load_state_dict(torch.load(PATH))
model.eval()
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    #image = Image.open(image).convert('0')
    return transform(image).unsqueeze(0)




def index(request):
    context={'a':1}
    return render(request,'index.html')
def predictImage(request):
    if request.method == 'POST':
        file=request.FILES['filePath']
        img_bytes = file.read()
        fs=FileSystemStorage()
        filePathName=fs.save(file.name,file)
        filePathName=fs.url(filePathName)
        testimage='.'+filePathName


        #img_bytes = cv2.imread('img_bytes',1)
        tensor = transform_image(img_bytes)
        prediction = get_prediction(tensor)
        print(prediction)
        #data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
        imagenet_class_mapping = json.load(open('model/imagenet_classes.json'))
        pred=str(prediction.item())
        preds=imagenet_class_mapping[pred]
        context={'a':preds,'filePathName':filePathName}
    return render(request,'index.html',context)

    #images = testimage.reshape(-1, 224*224)
#    outputs = model(image)
        # max returns (value ,index)
#    _, predicted = torch.max(outputs.data, 1)


def get_prediction(image_tensor):
    images = image_tensor#.reshape(64,3,7,7)
    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs,1)
    return predicted
    #import numpy as np
    #predictedLabel=labelInfo[str(np.argmax(predicted))]
    #context={'a':1}

    #context={'filePathName':filePathName,'predictedLabel':predictedLabel[1]}
    #return render(request,'index.html',context)


def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context)
