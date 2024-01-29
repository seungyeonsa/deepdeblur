import matplotlib.pyplot as plt
from datasetloader import GOPRODataset
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
from model import MainModel
import numpy as np

def pil_to_np(image):
    return np.array(image)


def visualize_results(model, dataset, num_pairs_to_display=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fig, axes = plt.subplots(num_pairs_to_display, 3, figsize=(12, 8))

    for i in range(num_pairs_to_display):
        blur_image, sharp_image = dataset[i]

        blur_np = pil_to_np(blur_image)
        sharp_np = pil_to_np(sharp_image)

        axes[i, 0].imshow(blur_np)
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Blurry Image')

        axes[i, 1].imshow(sharp_np)
        axes[i, 1].axis('off')
        axes[i, 1].set_title('Sharp Image')
        
        blur_image_tensor1 = TF.to_tensor(blur_image).unsqueeze(0).to(device)
        blur_image_tensor2 = F.interpolate(blur_image_tensor1, scale_factor=0.5, mode="bicubic").to(device)
        blur_image_tensor3 = F.interpolate(blur_image_tensor2, scale_factor=0.5, mode="bicubic").to(device)

        with torch.no_grad():
            _, _, model_out = model(blur_image_tensor3, blur_image_tensor2, blur_image_tensor1)
        import pdb; pdb.set_trace()
        predicted_sharp_np = pil_to_np(TF.to_pil_image(model_out.squeeze(0).cpu()))

        axes[i, 2].imshow(predicted_sharp_np)
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Predicted Sharp Image')

    plt.show()

if __name__ == '__main__':
    train_set = GOPRODataset(
        train_path='./data/GOPRO_Large/train',
        train_ext='png'
    )
    model = MainModel(3, 64, 128)
    model_path = './models_3'
    load_path = '{}/trained_model_epoch{}.pth'.format(model_path, 1)
    state_dict = torch.load(load_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)
    visualize_results(model, train_set)
