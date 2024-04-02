import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform

image_path = 'recognize-anything/images/demo/demo1.jpg'
pth_path = 'xx'
image_size = 384

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = get_transform(image_size=image_size)
model = ram_plus(pretrained=pth_path,
                 image_size=image_size,
                 vit='swin_l')
model.eval()
model = model.to(device)
image = transform(Image.open(image_path)).unsqueeze(0).to(device)

res = inference(image, model)
print("Image Tags: ", res[0])

