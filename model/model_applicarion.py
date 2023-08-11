import os
from model import VGG_model
import numpy as np
import torch
from PIL import Image 


def model_application(img, model_name = "vgg_grayscale"):

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]

    # Number of classes in the dataset
    num_classes = 2

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    model_ft, input_size = VGG_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)


    # Load the best checkpoint
    checkpoint_path = './saved_models/Alex_epoch10.pt'  # Replace X with the best epoch number
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    #checkpoint = torch.load(checkpoint_path)
    model_ft.load_state_dict(checkpoint['model_state_dict'])


    #load images
    # image_dir = "C:/Users/eenysh/OneDrive - University of Leeds/IAA application/Data_collection(20221011)/Fabric 14"

    img_size = (512,128)
    #img = img[:,600:3500]
    im = Image.fromarray(np.uint8(img))


    img = im.resize(img_size, Image.Resampling.LANCZOS)  # Resize the image
    
    img_array = np.array(img) / 255.0
    
    #convert img_array to tensor
    img_tensor = torch.from_numpy(img_array).float()

    #apply the model on the image
    model_ft.eval()
    with torch.no_grad():
        output = model_ft(img_tensor.unsqueeze(0).unsqueeze(0))  # add extra dimension for batch size and num_channels  
        result = torch.argmax(output, dim=1)
    
    return result


