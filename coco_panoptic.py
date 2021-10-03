# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask

from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

from .coco import make_coco_transforms
import traceback


class CocoPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_folder /  ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # sanity check

        self.image_id_annotation_map = {}
        self.image_id_image_obj_map = {}
        self.image_id_to_be_processed_list = []

        for image_obj in self.coco['images']:
            self.image_id_image_obj_map[image_obj['id']] = image_obj
        for annotation_obj in self.coco['annotations']:
            image_id = annotation_obj['image_id']
            if image_id not in self.image_id_to_be_processed_list:
                self.image_id_to_be_processed_list.append(image_id)
            if image_id in self.image_id_annotation_map:
                annotation_obj_list = self.image_id_annotation_map[image_id]
                annotation_obj_list.append(annotation_obj)
            else:
                self.image_id_annotation_map[image_id] = [annotation_obj]  



        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        #print("idx: " , idx)
        #image_obj = self.coco['images'][idx]
        #image_id = image_obj['id']
        image_id = self.image_id_to_be_processed_list[idx]
        image_obj = self.image_id_image_obj_map[image_id]
        
        #print("image_id: " , image_id)
        annotation_obj_list = self.image_id_annotation_map[image_id]

        img_path = Path(self.img_folder) / image_obj['file_name']
        #print("img_path " ,img_path)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        i = 0
        masks = None
        category_list = []
        area_list = []
        crowd_list = []
        mask_list = []
        for annotation_obj in annotation_obj_list:
            i += 1
            #print("annotation_obj: ", annotation_obj)
            if isinstance(annotation_obj['segment_map'], str):
                #print("segment_map is str")
                segment_map = self.get_mask_from_seg_map(annotation_obj['segment_map'])
                #print("str: segment_map:", segment_map)
            else:
                #print("segment_map is not str")
                segment_map_temp = annotation_obj['segment_map']
                segment_map = mask.frPyObjects(segment_map_temp, segment_map_temp.get('size')[0], segment_map_temp.get('size')[1])
                #print("non str: segment_map:", segment_map)
            try:
                decoded_map = mask.decode(segment_map)
            except:
                decoded_map = None
                print("Exception while decoding")
            if decoded_map is None:
                continue
            decoded_map_array = np.array(decoded_map)
            if annotation_obj['category_id']<= 0 or annotation_obj['category_id'] > 48:
                #print("adding annotation_id: ", annotation_obj['id'], " category_id: ", annotation_obj['category_id'], " shape: ",  decoded_map_array.T.shape)
                mask_list.append(decoded_map_array.T)
            else:
                #print("adding nnotation_id: ", annotation_obj['id'], " category_id: ", annotation_obj['category_id'], " shape: ",  decoded_map_array.shape)
                mask_list.append(decoded_map_array)
            category_list.append(annotation_obj['category_id'])
            area_list.append(annotation_obj['area'])
            #crowd_list.append(annotation_obj['crowd'])
            crowd_list.append(annotation_obj['iscrowd'])

        #print(" size of mask_list =", len(mask_list))
        try:
            if len(mask_list) == 1:
                masks = np.array(mask_list)
            else:
                masks = np.vstack(mask_list)
        except:
            traceback.print_exc()
            print(" Exception encountered: mask size: ", len(mask_list), " image_id: ", image_id)
        masks = masks.reshape(-1, w, h)
        #print("Final masks=", masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        #print("Final torch masks=", masks)

        #labels = torch.tensor([annotation_obj['category_id'] for annotation_obj in annotation_obj_list ], dtype=torch.int64)
        labels = torch.tensor(category_list, dtype=torch.int64)

        #print("labels=", labels)
        target = {}
        target['image_id'] = torch.tensor([image_id])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        target["boxes"] = masks_to_boxes(masks)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['iscrowd'] = torch.tensor(crowd_list)
        target['area'] = torch.tensor(area_list)

        #for name in ['iscrowd', 'area']:
        #    target[name] = torch.tensor([annotation_obj[name] for annotation_obj in annotation_obj_list])

        if self.transforms is not None:
            #print(" inside transforms")
            img, target = self.transforms(img, target)
        #print("before return img:", img)
        #print("before return  target :" , target)
        return img, target

    def __len__(self):
        return len(self.image_id_to_be_processed_list)	


    def get_height_and_width(self, idx):
        img_info = self.coco['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width

    def get_mask_from_seg_map(self, seg_map):
        if 'counts' not in seg_map or 'size' not in seg_map:
            return []
        if seg_map.index('size') < seg_map.index('counts'):
            format = 1
            #print(' size is before counts: format=1')
        else:
            format = 2
            #print(' counts is before size: format=2')
		
        counts_pattern = "'counts': ["
        seg_without_curly = seg_map[1: len(seg_map) - 1]
        result_dict = None
    
        if format == 1:
            index1 = seg_without_curly.index("'size': [")
            index2 = seg_without_curly.index("]", index1 + len("'size': [")  )
            element1 = seg_without_curly[index1 + len("'size': ["): index2]
            element1_list_str =  element1.split(",")
            element1_list_int =  [int(elem.strip()) for elem in element1_list_str]

            index3 = seg_without_curly.index(counts_pattern)
            #index4 = len(seg_without_curly) - 2
            index4 = seg_without_curly.rindex("]", index3 + len(counts_pattern))
            element2 = seg_without_curly[index3 + len(counts_pattern): index4]
            element2_list_str =  element2.split(",")
            element2_list_int =  [int(elem.strip()) for elem in element2_list_str]

            result_dict = {'counts' : element2_list_int, 'size' : element1_list_int  }
        elif format == 2:
            index1 = seg_without_curly.index(counts_pattern)
            index2 = seg_without_curly.index("]", index1 + len(counts_pattern))
            element1 = seg_without_curly[index1 + len(counts_pattern): index2]
            element1_list_str =  element1.split(",")
            element1_list_int =  [int(elem.strip()) for elem in element1_list_str]
        
            index3 = seg_without_curly.index("'size': [")
            index4 = seg_without_curly.index("]", index3 + len("'size': ["))
            element2 = seg_without_curly[index3 + len("'size': ["): index4]
        
            element2_list_str =  element2.split(",")
            element2_list_int =  [int(elem.strip()) for elem in element2_list_str]

            result_dict = {'counts' : element1_list_int, 'size' : element2_list_int  }
        
        compressed_rle = mask.frPyObjects(result_dict, result_dict.get('size')[0], result_dict.get('size')[1])
        return compressed_rle

    def remove_back_slash(self, input):
        output = ''
        if input is None or input == '' :
            return output
        last_char = ''
        filtered_chars = []
        for ch in input:
            if ch == '\\' and last_char == '\\':
                last_char= ch
                continue
            else:
                filtered_chars.append(ch)
                last_char= ch
        output = ''.join(filtered_chars)
        return output
		
		
def build(image_set, args):
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_panoptic_path)
    assert img_folder_root.exists(), f'provided COCO path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided COCO path {ann_folder_root} does not exist'
    mode = 'panoptic'
    PATHS = {
        "train": ("train2017", Path("annotations") / f'{mode}_train2017.json'),
        "val": ("val2017", Path("annotations") / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    #ann_folder = ann_folder_root / f'{mode}_{img_folder}'		
    ann_folder = ann_folder_root

    dataset = CocoPanoptic(img_folder_path, ann_folder, ann_file,
                           transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset
  		
