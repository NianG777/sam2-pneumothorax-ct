from pycocotools import mask
from skimage import measure
import json
import shutil
import itertools
import numpy as np
from simplification.cutil import simplify_coords_vwp
import os, cv2, copy
from distinctipy import distinctipy


def init_coco(dataset_folder, image_names, categories, coco_json_path):
    coco_json = {
        "info": {
            "description": "SAM Dataset",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "Sam",
            "date_created": "2021/07/01",
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for i, category in enumerate(categories):
        coco_json["categories"].append(
            {"id": i, "name": category, "supercategory": category}
        )
    for i, image_name in enumerate(image_names):
        # 强制以3通道方式读取，忽略PNG透明通道，保证宽高获取一致
        im = cv2.imread(os.path.join(dataset_folder, image_name), cv2.IMREAD_COLOR)
        coco_json["images"].append(
            {
                "id": i,
                "file_name": image_name,
                "width": im.shape[1],
                "height": im.shape[0],
            }
        )
    with open(coco_json_path, "w") as f:
        json.dump(coco_json, f)


def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans


def unbunch_coords(coords):
    return list(itertools.chain(*coords))


def bounding_box_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def parse_mask_to_coco(image_id, anno_id, image_mask, category_id, poly=False):
    start_anno_id = anno_id
    x, y, width, height = bounding_box_from_mask(image_mask)
    if poly == False:
        fortran_binary_mask = np.asfortranarray(image_mask)
        encoded_mask = mask.encode(fortran_binary_mask)
    if poly == True:
        contours = measure.find_contours(image_mask, 0.5)
    annotation = {
        "id": start_anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": [],
    }
    if poly == False:
        annotation["segmentation"] = encoded_mask
        annotation["segmentation"]["counts"] = str(
            annotation["segmentation"]["counts"], "utf-8"
        )
    if poly == True:
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            sc = bunch_coords(segmentation)
            sc = simplify_coords_vwp(sc, 2)
            sc = unbunch_coords(sc)
            if len(sc) > 4:
                annotation["segmentation"].append(sc)
    return annotation


class DatasetExplorer:
    def __init__(self, dataset_folder, categories=None, coco_json_path=None):
        self.dataset_folder = dataset_folder
        
        # 递归获取images文件夹下的所有图像文件
        self.image_names = self._get_all_image_files()
        
        self.coco_json_path = coco_json_path
        if not os.path.exists(coco_json_path):
            self.__init_coco_json(categories)
        with open(coco_json_path, "r") as f:
            self.coco_json = json.load(f)

        self.categories = [category["name"] for category in self.coco_json["categories"]]
        self.annotations_by_image_id = {}
        for annotation in self.coco_json["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(annotation)

        self.global_annotation_id = len(self.coco_json["annotations"])
        self.category_colors = distinctipy.get_colors(len(self.categories))
        self.category_colors = [
            tuple([int(255 * c) for c in color]) for color in self.category_colors
        ]

    def _get_all_image_files(self):
        """
        递归获取images文件夹下所有图像文件的相对路径
        支持子文件夹结构
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        images_folder = os.path.join(self.dataset_folder, "images")
        
        if not os.path.exists(images_folder):
            return []
        
        for root, dirs, files in os.walk(images_folder):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    # 计算相对于images文件夹的路径
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, images_folder)
                    image_files.append(relative_path)
        
        return image_files

    def __init_coco_json(self, categories):
        # 为COCO JSON格式添加images/前缀
        appended_image_names = [
            os.path.join("images", name) for name in self.image_names
        ]
        init_coco(
            self.dataset_folder, appended_image_names, categories, self.coco_json_path
        )

    def get_colors(self, category_id):
        return self.category_colors[category_id]
    
    def get_categories(self):
        return self.categories

    def get_num_images(self):
        return len(self.image_names)

    def get_image_data(self, image_id):
        image_name = self.image_names[image_id]
        
        # 构建图像路径（支持子文件夹）
        image_path = os.path.join(self.dataset_folder, "images", image_name)
        
        # 构建embedding路径（支持子文件夹）
        # 将图像文件扩展名替换为.npy
        embedding_name = os.path.splitext(image_name)[0] + ".npy"
        embedding_path = os.path.join(self.dataset_folder, "embeddings", embedding_name)
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None, None, None, None
            
        if not os.path.exists(embedding_path):
            print(f"Warning: Embedding file not found: {embedding_path}")
            return None, None, None, None
        
        # 读取图像
        # 以3通道方式读取，避免PNG透明通道导致下游处理异常
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"Error: Failed to load image: {image_path}")
            return None, None, None, None
            
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 读取embedding
        try:
            image_embedding = np.load(embedding_path)
        except Exception as e:
            print(f"Error: Failed to load embedding: {embedding_path}, {e}")
            return None, None, None, None
        
        # 获取图像对应的annotations
        annotations = self.annotations_by_image_id.get(image_id, [])
        
        return image, image_embedding, annotations, image_name

    def __add_to_our_annotation_dict(self, annotation):
        image_id = annotation["image_id"]
        if image_id not in self.annotations_by_image_id:
            self.annotations_by_image_id[image_id] = []
        self.annotations_by_image_id[image_id].append(annotation)
    
    def __delet_to_our_annotation_dict(self, image_id):
        self.annotations_by_image_id[image_id].pop(-1)

    def get_annotations(self, image_id, return_colors=False):
        if image_id not in self.annotations_by_image_id:
            return [], []
        cats = [a["category_id"] for a in self.annotations_by_image_id[image_id]]
        colors = [self.category_colors[c] for c in cats]
        if return_colors:
            return self.annotations_by_image_id[image_id], colors
        return self.annotations_by_image_id[image_id]

    def add_annotation(self, image_id, category_id, mask, poly=True):
        if mask is None:
            return
        annotation = parse_mask_to_coco(
            image_id, self.global_annotation_id, mask, category_id, poly=poly
        )
        self.__add_to_our_annotation_dict(annotation)
        self.coco_json["annotations"].append(annotation)
        self.global_annotation_id += 1

    def delet_annotation(self, image_id):
        self.__delet_to_our_annotation_dict(image_id)
        self.coco_json["annotations"].pop(-1)
        self.global_annotation_id -= 1

    def save_annotation(self):
        with open(self.coco_json_path, "w") as f:
            json.dump(self.coco_json, f)
