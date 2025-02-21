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

    def getitem(self):
        pass

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


# def plot_results(preds, support_images):
#     """
#     Plots support images and query predictions in two separate figures.
#
#     - Support images are provided as a list of lists; each inner list contains
#       |shot| tensori per una determinata classe. The grid is arranged with:
#          * number of rows = number of classes
#          * number of columns = number of shots per class
#     - Query predictions (preds) are provided as a list of tensori (RGB images)
#       and are displayed in a separate figure.
#
#     Args:
#       preds: list of query tensors (each of shape (1, 3, H, W))
#       support_images: list of lists; each inner list contains support image tensors
#                       (each of shape (1, 3, H, W)) for one class.
#     """
#
#     ####### PLOT 1: SUPPORT IMAGES ########
#     num_classes = len(support_images)
#     if num_classes == 0:
#         print("No support images provided!")
#         return
#     # Assumiamo che ogni classe abbia lo stesso numero di shot
#     shots = len(support_images[0])
#
#     fig_support, axes_support = plt.subplots(num_classes, shots,
#                                              figsize=(5 * shots, 5 * num_classes))
#
#     # Assicuriamoci che axes_support sia un array 2D, indipendentemente da num_classes e shots
#     if num_classes == 1 and shots == 1:
#         axes_support = np.array([[axes_support]])
#     elif num_classes == 1:
#         axes_support = np.atleast_2d(axes_support)  # Diventa (1, shots)
#     elif shots == 1:
#         axes_support = axes_support[:, np.newaxis]  # Diventa (num_classes, 1)
#
#     # Itera per ogni classe e per ogni shot della classe
#     for i, class_supports in enumerate(support_images):
#         for j, support_img in enumerate(class_supports):
#             # Convertiamo il tensore in immagine (H, W, 3)
#             np_support = support_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             np_support = np.clip(np_support, 0, 1)
#             axes_support[i, j].imshow(np_support)
#             axes_support[i, j].set_title(f"Class {i + 1} - Shot {j + 1}")
#             axes_support[i, j].axis("off")
#
#     fig_support.suptitle("Support Images", fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#
#     ####### PLOT 2: QUERY PREDICTIONS ########
#     num_queries = len(preds)
#     if num_queries == 0:
#         print("No query predictions provided!")
#         return
#
#     # Se c'è una sola query, trattiamola in modo speciale
#     if num_queries == 1:
#         fig_query, ax = plt.subplots(1, 1, figsize=(5, 5))
#         np_pred = preds[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
#         np_pred = np.clip(np_pred, 0, 1)
#         ax.imshow(np_pred)
#         ax.set_title("Query 1")
#         ax.axis("off")
#     else:
#         fig_query, axes_query = plt.subplots(1, num_queries, figsize=(5 * num_queries, 5))
#         # Se plt.subplots restituisce un array 1D per una singola riga, iteriamo direttamente
#         for i, pred in enumerate(preds):
#             np_pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
#             np_pred = np.clip(np_pred, 0, 1)
#             axes_query[i].imshow(np_pred)
#             axes_query[i].set_title(f"Query {i + 1}")
#             axes_query[i].axis("off")
#
#     fig_query.suptitle("Query Predictions", fontsize=16)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#
#     # Mostra entrambi i plot
#     plt.show()

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
    print(f"Loading model from snapshot: {_config['snapshot']}")
    model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()

    files = './spal' #'./demo' #
    input_folder = os.path.join(files, 'queries')
    support_base_dir = os.path.join(files, 'support')


    annotation_path = os.path.join(support_base_dir, 'support_annotation.json')

    # net = _config['net']
    n_shot = _config['n_shots']
    # n_ways = _config['n_ways']

    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]

    query_dataset = QueryDataset(image_paths, _config['input_size'])
    query_loader = DataLoader(query_dataset, batch_size=_config['task']['n_queries'], shuffle=False)

    support_dataset = SupportDataset(
        support_base_dir, # support_classes_dir,
        annotation_path,
        _config['input_size'],
        to_tensor=ToTensorNormalize()
    )
    support_loader = DataLoader(support_dataset, batch_size=_config['task']['n_shots'], shuffle=False)


    with torch.no_grad():

        support_samples = []
        support_images = []
        support_masks = []
        class_ids = []
        mask_classes = []
        for i, support_sample in enumerate(support_loader):
            if i == n_shot:
                break
            support_samples.append(support_sample)
            support_images.append(support_sample['image'].cuda())
            support_masks.append(list(support_sample['label'].values()))
            # class_ids.append(list(support_sample.get('label').keys())[i])
            # if len(class_ids[i]) != n_ways:
            #     raise ValueError(f"Expected {n_ways} classes, but got {len(class_ids)}")

        for mask in support_masks: #è lista di maschere per ogni support sample li devo mettere su device i tensori
            for i, m in enumerate(mask):
                mask[i] = m.cuda()

        # lista contenente liste di classi per ogni support sample
        class_ids = [list(support_sample['label'].keys()) for i, support_sample in enumerate(support_samples)]
        for i, class_id in enumerate(class_ids):
            for shot in support_masks[i]:
                masks = []
                for j in range(len(class_id)):
                    mask = getMask(shot, class_id[j], class_id)
                    masks.append(mask)
                mask_classes.append(masks)

        support_fg_masks = []
        support_bg_masks = []
        for shot in mask_classes:
            support_fg_mask = []
            support_bg_mask = []
            for way in shot:
                support_fg_masks.append(way['fg_mask'].squeeze(1))
                support_bg_masks.append(way['bg_mask'].squeeze(1))
            # support_fg_masks.append(support_fg_mask)
            # support_bg_masks.append(support_bg_mask)


        # print('mask:', support_fg_mask[0].shape)
        # print('mask:', support_bg_mask[0].shape)

        support_images = [
            list(support_image.split(1, dim=0)) for support_image in support_images
        ]

        for i in range(len(support_images)):
            support_masks[i] = list(support_masks[i][0].split(1, dim=0))

        support_fg_masks = [
            list(support_fg_mask.split(1, dim=0)) for support_fg_mask in support_fg_masks
        ]

        support_bg_masks = [
            list(support_bg_mask.split(1, dim=0)) for support_bg_mask in support_bg_masks
        ]
        # support_fg_masks = list(support_fg_masks.split(1, dim=0))
        # support_bg_masks = list(support_bg_masks.split(1, dim=0))

        for i, query_images in enumerate(query_loader):
            query_images = query_images.cuda() #N x [B x 3 x H x W], tensors ( N is # of queries x batch)

            # Passa i supporti nel formato corretto al modello
            query_preds, _ = model(support_images, support_fg_masks, support_bg_masks, [query_images])
            query_preds = torch.where(query_preds > 0.5, torch.ones_like(query_preds), torch.zeros_like(query_preds))
            if type(query_images) == list:
                query_images = query_images[0]

            query_images = denormalize_tensor(query_images)
            query_preds = apply_mask_overlay(query_images, query_preds.argmax(dim=1))
            plot_image(query_preds, desc=f'Query Image {i+1}')

        # for images, masks in zip(support_images, support_masks):
        #     for j, (img, mask) in enumerate(zip(images, masks)):
        #         img = denormalize_tensor(img)
        #         img = apply_mask_overlay(img, mask.squeeze(0))
        #         plot_image(img, desc=f"Support Image {j+1}")


    print("Inference completed!")

