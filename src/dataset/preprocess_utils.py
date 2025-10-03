from PIL import Image
import torchvision.transforms.functional as TF

def preprocess_image(image_path,model_input_size=(256,256)):
    image=Image.open(image_path).convert('L')
    original_size=image.size

    #resize
    image_resized=image.resize(model_input_size,Image.BILINEAR)

    #to tensor
    tensor=TF.to_tensor(image_resized).float().unsqueeze(0)

    return tensor,image,original_size