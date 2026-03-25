import os
import random
import cv2
import math
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .encoder import DataEncoder
from .configuration import SystemConfig, TrainerConfig, DataloaderConfig 

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count


def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, batch_size_to_set=DataloaderConfig.batch_size):
    """ Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, num_workers=num_workers_to_set)
    trainer_config = TrainerConfig(device=device, epoch_num=epoch_num_to_set, progress_bar=True)
    return dataloader_config, trainer_config


def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


def resize(img, boxes, size, max_size=1000):
    '''Resize the input cv2 image to the given size.

    Args:
      img: (cv2) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (cv2) resized image.
      boxes: (tensor) resized boxes.
    '''
    height, width, _ = img.shape
    if isinstance(size, int):
        size_min = min(width, height)
        size_max = max(width, height)
        scale_w = scale_h = float(size) / size_min
        if scale_w * size_max > max_size:
            scale_w = scale_h = float(max_size) / size_max
        new_width = int(width * scale_w + 0.5)
        new_height = int(height * scale_h + 0.5)
    else:
        new_width, new_height = size
        scale_w = float(new_width) / width
        scale_h = float(new_height) / height

    return cv2.resize(img, (new_height, new_width)), \
           boxes * torch.Tensor([scale_w, scale_h, scale_w, scale_h])


def random_flip(img, boxes):
    '''Randomly flip the given cv2 Image.

    Args:
        img: (cv2) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (cv2) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        width = img.shape[1]
        xmin = width - boxes[:, 2]
        xmax = width - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
    return img, boxes


def read_img_labels(img_name, label_path, output = 'tensor'):
    list_file_path = os.path.join(label_path, img_name.replace('.jpg', '.txt'))
    boxes = []
    labels = []
    label_names = []

    with open(list_file_path) as f:
        lines = f.readlines()

    # print(f" image name: {img_name} - # bounding boxes: {len(lines)}")
    for line in lines:
        splited = line.strip().split()
        xmin = splited[3]
        ymin = splited[4]
        xmax = splited[5]
        ymax = splited[6]
        boxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
        labels.append(int(1))        
        label_names.append("Reg_Plate")
    
    if output == 'tensor':
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels, dtype = torch.int64)
    else:
        boxes = np.array(boxes)
        labels = np.array(labels, dtype=np.int64)
        
    return boxes, labels, label_names

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_DUPLEX,
          pos=(0, 0),
          font_scale=0.4,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    # print(x,text_w, y,text_h, font_scale)
    cv2.rectangle(img, (x, y - (text_h+4)), (x + text_w, y - 1), text_color_bg, -1)
    cv2.putText(img, text, (x, y - 3), font, font_scale, text_color, font_thickness)
    return text_size

    
def display_training_sample(trn_sample, title = None, figsize = (8,8), rect_th=1, text_size=0.4, text_th=1):
    """
    display an training sample from the training dataset
    parameters:
      - trn_sample: An instance returned from the training dataset
      - rect_th   : Thickness of bounding box
      - text_size : Size of the class label text
      - text_th   : Thickness of the text
    """
    img, boxes, labels, label_names = trn_sample
    img = np.transpose(img.numpy(), (1,2,0))
    for box, label_name in  zip(boxes, label_names):
        print("detected boxes: ")
        print(box, label_name)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=rect_th)
        draw_text(img,label_name, pos=(int(box[0]), int(box[1])), font_thickness = 1, font_scale= 0.3)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(img)
    plt.show()


def draw_bboxes(exp, rows, columns, trainer_config: TrainerConfig):
    # load the best model
    if trainer_config.model_save_best:
        exp.model.load_state_dict(
            torch.load(os.path.join(trainer_config.model_dir, exp.model.__class__.__name__) + '_best.pth')
        )
    # or use the last saved
    exp.model = exp.model.eval()

    std = (0.229, 0.224, 0.225)
    mean = (0.485, 0.456, 0.406)

    std = torch.Tensor(std)
    mean = torch.Tensor(mean)

    fig, ax = plt.subplots(
        nrows=rows, ncols=columns, figsize=(17, 17), gridspec_kw={
            'wspace': 0,
            'hspace': 0.05
        }
    )

    for axi in ax.flat:
        index = random.randrange(len(exp.loader_test.dataset))

        image, gt_boxes, labels, label_names = exp.loader_test.dataset[index]

        device = torch.device(trainer_config.device)
        image = image.to(device).clone()

        loc_preds, cls_preds = exp.model(image.unsqueeze(0))

        with torch.no_grad():
            img = image.cpu()
            img.mul_(std[:, None, None]).add_(mean[:, None, None])
            img = torch.clamp(img, min=0.0, max=1.0)
            img = img.numpy().transpose(1, 2, 0)

            img = (img * 255.).astype(np.uint8)
            gt_img = img.copy()
            pred_img = img.copy()

            for box in gt_boxes:
                gt_img = cv2.rectangle(
                    gt_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0),
                    thickness=2
                )

            encoder = DataEncoder((img.shape[1], img.shape[0]))
            samples = encoder.decode(loc_preds, cls_preds)
            c_dets = samples[0][1]  # detections for class == 1

            if c_dets.size > 0:
                boxes = c_dets[:, :4]
                for box in boxes:
                    pred_img = cv2.rectangle(
                        pred_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                        thickness=4
                    )

            merged_img = np.concatenate((gt_img, pred_img), axis=1)
            axi.imshow(merged_img)
            axi.axis('off')
    fig.show()    


def run_inference(exp, image_ids = None, rows = None, columns = 3, cls_threshold=0.1, nms_threshold=0.3):

    std = torch.Tensor((0.229, 0.224, 0.225))
    mean = torch.Tensor((0.485, 0.456, 0.406))

    if isinstance(image_ids, int):
        rng = np.random.default_rng()
        image_ids = rng.integers(low=0, high=len(exp.loader_test.dataset), size=image_ids)  
    print(f"image_ids = {image_ids}")
    
    if rows is None: 
        rows = math.ceil(len(image_ids)/columns)
        
    fig, ax = plt.subplots(
        nrows=rows, ncols=columns, figsize=(columns*10, rows*10), 
        gridspec_kw={
            'wspace': 0.05,
            'hspace': 0.05
        }
    )
    
    print(f" rows: {rows}  columns: {columns}  ax: {type(ax)}")
    input_size = exp.dataset_test.input_size
    encoder = DataEncoder((input_size, input_size))
    device = exp.inference_config.device
    ax_list = [ax] if isinstance(ax, matplotlib.axes._axes.Axes) else  [ax for ax in  iter(ax.flat)]
    
    for img_id , (index, axi) in enumerate(zip(image_ids, ax_list)):

        # index = image_ids[img_id]
        # axi = next(iter_ax_flat)
        # print(f" {img_id} -  image dataset index: {index}")
        
        image, gt_boxes, labels, label_names = exp.loader_test.dataset[index]
        image = image.to(device).clone()
        loc_preds, cls_preds = exp.model(image.unsqueeze(0))
        
        # print(f" \timage    : {image.shape}  dtype: {image.dtype}   min: {image.min()}  max: {image.max()} ")
        # print(f" \tgt_boxes : {gt_boxes}")
        # print(f" \tlabels   : {labels}")
        # print(f" \tlabels   : {label_names}")
        # print(f" \tloc_preds: {loc_preds.shape}"  \tcls_preds: {cls_preds.shape}")
        
        with torch.no_grad():
            img = image.cpu()
            # print(f" \t1- img:   : {img.shape}  dtype: {img.dtype}   min: {img.min()}  max: {img.max()} ")
            img.mul_(std[:, None, None]).add_(mean[:, None, None])
            # print(f" \t2- img:   : {img.shape}  dtype: {img.dtype}   min: {img.min()}  max: {img.max()} ")
            img = torch.clamp(img, min=0.0, max=1.0)
            # print(f" \t3- img:   : {img.shape}  dtype: {img.dtype}   min: {img.min()}  max: {img.max()} ")
            img = img.numpy().transpose(1, 2, 0)
            # print(f" \t4- img:   : {img.shape}  dtype: {img.dtype}   min: {img.min()}  max: {img.max()} ")
            img = (img * 255.).astype(np.uint8)
            # print(f" \t4- img:   : {img.shape}  dtype: {img.dtype}   min: {img.min()}  max: {img.max()} \n")

            pred_img = img.copy()
            axi.imshow(pred_img)
            axi.set_title(f" index: {index}")

            samples = encoder.decode(loc_preds, cls_preds, 
                                     cls_threshold=cls_threshold, 
                                     nms_threshold=nms_threshold)
            # print(f" \tsamples: {type(samples)}   {len(samples)}  {len(samples[0])}  {[type(i) for i in samples[0]]}  samples[0][1]: {samples[0][1].shape}")
            # print(f" \tencoder: {type(encoder)}   {img.shape[1]}  {img.shape[0]}")
            # for tmp in samples[0][1]:
            #     print(f" \t       : {tmp}")
        
            # c_dets = samples[0][1]  # detections for class == 1
            # print(f" \tc_dets : {type(c_dets)}   shape: {c_dets.shape}  {samples[0][1].shape}     size: {c_dets.size}  {samples[0][1].size} ")
            if samples[0][1].size > 0:
                for c_dets_row in samples[0][1]:
                    left, bot, right, top, confidence = c_dets_row
                    x, y, w, h = [val for val in [left, bot, right - left, top - bot]]
                    # print(f"  \t    :  left: {left}  bot: {bot}  right: {right}  top: {top}")
                    # print(f"  \t    :  X: {x}  Y: {y}  W: {w}  H: {h}")
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')       
                    axi.add_patch(rect)
                    axi.text(x, y, "{} {:.0f}%".format('Reg_Plate', confidence*100), bbox=dict(facecolor='white', alpha=0.5))                        
            axi.axis('off')
    fig.show()


# def display_image(img_name, image_path = None, label_path = None, threshold=0.5, rect_th=1, text_size=0.4, text_th=1):
#     """
#     isplay
#     parameters:
#       - img_path - path of the input image
#       - threshold - threshold value for prediction score
#       - rect_th - thickness of bounding box
#       - text_size - size of the class label text
#       - text_th - thickness of the text
#     method:
#       - prediction is obtained from get_prediction method
#       - for each prediction, bounding box is drawn and text is written 
#         with opencv
#       - the final image is displayed
#     """
#     boxes = []
#     img_path = os.path.join(image_path, img_name)
#     boxes, labels, label_names = read_img_labels(img_name, label_path)
#     print(f"boxes: {boxes}, labels: {labels}, label_names: {label_names}"   )
#     img = cv2.imread(img_path)
#     print(f" image name: {img_name} - image shape: {img.shape}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     for box, label_name in  zip(boxes, label_names):
#         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=rect_th)
#         draw_text(img,label_name, pos=(int(box[0]), int(box[1])), font_thickness = 1, font_scale= 0.3)
#         # cv2.putText(img, label,  (int(box[0])+2, int(box[1])+10), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
#     plt.figure(figsize=(20,30))
#     plt.imshow(img)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
