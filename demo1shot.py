import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models.fewshot import FewShotSeg
from dataloaders.common import BaseDataset
from dataloaders.transforms import ToTensorNormalize, Resize
# from dataloaders.customized import getMask #l'ho rifatta per convenienza
from util.utils import get_bbox
from util.visual_utils import apply_mask_overlay
from pycocotools.coco import COCO
from config import ex
import numpy as np


def getMask(label, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        class_id:
            semantic class of interest

        class_ids:
            all class id in this episode

    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,}



class QueryDataset(Dataset):
    def __init__(self, image_paths, input_size):
        self.image_paths = image_paths
        self.input_size = input_size
        self.transform = Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


class SupportDataset(BaseDataset):
    def __init__(self, base_dir, annotation_file, input_size, to_tensor=None):
        super().__init__(base_dir)
        self.coco = COCO(annotation_file)
        self.ids = self.coco.getImgIds()
        t = [Resize(size=input_size)]
        self.transforms = Compose(t)
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_meta = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self._base_dir, img_meta['file_name'])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        masks = {}
        for ann in anns:
            catId = ann['category_id']
            mask = self.coco.annToMask(ann)
            if catId in masks:
                masks[catId][mask == 1] = catId
            else:
                semantic_mask = torch.zeros((img_meta['height'], img_meta['width']), dtype=torch.uint8)
                semantic_mask[mask == 1] = catId
                masks[catId] = semantic_mask

        instance_mask = torch.zeros_like(semantic_mask)
        scribble_mask = torch.zeros_like(semantic_mask)

        sample = {'image': image, 'label': masks, 'inst': instance_mask, 'scribble': scribble_mask}

        #debugging
        print("Type of self.transforms:", type(self.transforms))
        print("Type of Resize:", type(Resize))
        # print("Applying Resize to sample:", sample)
        ###
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = img_id
        return sample


def denormalize_tensor(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizza un'immagine (C, H, W) applicando l'operazione inversa della normalizzazione.
    Se necessario, l'immagine viene clipppata per garantire che i valori siano compresi tra [0, 1].
    """
    batch = False
    if len(img.shape) == 4:
        batch = True
        img = img.squeeze(0)
    mean = torch.tensor(mean).view(3, 1, 1).to(img.device)
    std = torch.tensor(std).view(3, 1, 1).to(img.device)
    img = img * std + mean 
    img = torch.clamp(img, 0, 1)  
    if batch:
        img = img.unsqueeze(0)
    return img


def plot_image(query_images: torch.Tensor, desc = " "):
    """
    Plots input images.

    :param query_images: Input images tensor of shape (B, 3, H, W) with values in [0, 1].
    """
    # Converti il tensore in un array NumPy
    np_images = query_images.detach().cpu().numpy()
    # Se il batch contiene più di una immagine, prendi la prima (oppure potresti iterare su tutte)
    img = np_images[0]  # Forma: (3, H, W)
    # Trasponi in modo da avere (H, W, 3) per plt.imshow()
    img = np.transpose(img, (1, 2, 0))
    # Crea la figura e l'asse
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.set_title(desc)
    ax.axis("off")

    plt.show()



@ex.automain
def main(_config):
    torch.cuda.set_device(_config['gpu_id'])
    torch.set_num_threads(1)

    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = torch.nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id']])
    model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()

    # input_folder = './demo/queries'
    # annotation_path = './demo/support/support_annotation.json'
    # support_base_dir = './demo/support/'


    files = './ripetuto' #spal #demo #ripetuto
    input_folder = os.path.join(files, 'queries')
    support_base_dir = os.path.join(files, 'support')
    annotation_path = os.path.join(support_base_dir, 'support_annotation.json')

    n_shot = _config['n_shots']
    n_ways = _config['n_ways']


    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]

    query_dataset = QueryDataset(image_paths, _config['input_size'])
    query_loader = DataLoader(query_dataset, batch_size=_config['task']['n_queries'], shuffle=False)

    support_dataset = SupportDataset(support_base_dir, annotation_path, _config['input_size'], to_tensor=ToTensorNormalize())
    support_loader = DataLoader(support_dataset, batch_size=_config['task']['n_shots'], shuffle=False)

    with torch.no_grad():
        support_samples = next(iter(support_loader))

        # Organizza le immagini di supporto nel formato
        # support images way x shot x [B x 3 x H x W], list of lists of tensors
        support_images = [
            [support_samples['image'][shot_idx].cuda().unsqueeze(0) for shot_idx in range(len(support_samples['image']))]
        ]

        # Converte le maschere in una lista ordinata e le organizza per shot (deve essere lista di liste di tensori)
        support_masks = [
            [list(support_samples['label'].values())[shot_idx].cuda() for shot_idx in range(n_shot)]
        ]

        class_ids = list(support_samples.get('label').keys())
        if len(class_ids) != n_ways:
            raise ValueError(f"Expected {n_ways} classes, but got {len(class_ids)}")

        mask_classes = [] # lista di dict contenente le foreground masks and background masks
        for i, class_id in enumerate(class_ids):
            for j, shot in enumerate(support_masks[i]):
                shot = getMask(shot, class_id, class_ids)
                mask_classes.append(shot) # list of dicts, each dict contains fg_mask and bg_mask for a class

        #foreground and background masks for support images
        support_fg_mask = [[shot.float().cuda() for shot in way['fg_mask']]
                           for way in mask_classes] #  way x shot x [B x H x W], list of lists of tensors
        support_bg_mask = [[shot.float().cuda() for shot in way['bg_mask']] # way x shot x [B x H x W], list of lists of tensors
                           for way in mask_classes]

        # print('mask:', support_fg_mask[0][0].shape)
        # print('mask:', support_bg_mask[0][0].shape)

        for i, query_images in enumerate(query_loader):
            query_images = query_images.cuda() #N x [B x 3 x H x W], tensors ( N is # of queries x batch)

            # Passa i supporti nel formato corretto al modello
            query_preds, _ = model(support_images, support_fg_mask, support_bg_mask, [query_images])
            query_preds = torch.where(query_preds > 0.75, torch.ones_like(query_preds), torch.zeros_like(query_preds))
            if type(query_images) == list:
                query_images = query_images[0]

            query_images = denormalize_tensor(query_images)
            query_preds = apply_mask_overlay(query_images, query_preds.argmax(dim=1))
            plot_image(query_preds, desc=f'Query Image {i+1}')

        for i, (images, masks) in enumerate(zip(support_images, support_masks)):
            for img, mask in zip(images, masks):
                img = denormalize_tensor(img)
                img = apply_mask_overlay(img, mask.squeeze(0))
                plot_image(img, desc=f"Support Image {i+1}")

    print("Inference completed!")
