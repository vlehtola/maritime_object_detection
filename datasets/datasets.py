"""
Tinae Yehuala 2025. Mostly modified from MMDetection @https://github.com/open-mmlab/mmdetection.
Modification are for the purpose of  making it fit to Singapore and ABOship marine dataset. 
"""
import copy
from typing import List, Sequence, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset
from mmdet.datasets.dataset_wrappers import ConcatDataset

# category meta data from singapore dataset
METAINFO_SMD = {
        "classes" : ("objects", 
                     "Boat", 
                     "Buoy", 
                     "Ferry", 
                     "Flying bird-plane", 
                     "Kayak", 
                     "Other", 
                     "Sail boat", 
                     "Speed boat", 
                     "Vessel-ship"),
        "palette" : [(220, 20, 60), 
                     (119, 11, 32), 
                     (0, 0, 142), 
                     (0, 0, 230), 
                     (106, 0, 228), 
                     (0, 60, 100), 
                     (0, 80, 100), 
                     (0, 0, 70), 
                     (0, 0, 192), 
                     (250, 170, 30)]}

# category mapping between ABOships and Singapore
MAPPING_SMD2ABO =  {1 : 1, 
                    2 : 9, 
                    3 : 9, 
                    4 : 3, 
                    5 : 9, 
                    6 : 1, 
                    7 : 6, 
                    8 : 8, 
                    9 : 9, 
                    10 : 7, 
                    11: 2}

# category meta data for ABOship dataset
METAINFO_ABO = { "classes" : ("boat", 
                              "cargoship", 
                              "cruiseship", 
                              "ferry", 
                              "militaryship", 
                              "miscboat", 
                              "miscellaneous", 
                              "motorboat", 
                              "passengership", 
                              "sailboat", 
                              "seamark"),
                  "palette" :[(220, 20, 60), 
                            (119, 11, 32), 
                            (0, 0, 142), 
                            (0, 0, 230), 
                            (106, 0, 228), 
                            (0, 60, 100), 
                            (0, 80, 100), 
                            (0, 0, 70), 
                            (0, 0, 192), 
                            (250, 170, 30),
                            (250, 170, 30)]}

# dataset class on original datasets
@DATASETS.register_module()
class SingaporeOrig(CocoDataset):
    METAINFO = METAINFO_SMD

@DATASETS.register_module()
class AboOrig(CocoDataset):
    METAINFO = METAINFO_ABO
    
# make sure the label for each categories gets some label in both datasets
class CommonDataset(CocoDataset):
    def load_data_list(self):
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501

        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        # the order of category ids in cat_ids will determine the label of the class, so sort the ideas to make it consistant
        self.cat_ids = sorted(self.coco.get_cat_ids(cat_names=self.metainfo['classes']))    
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

@DATASETS.register_module()
class SingaporeDataset(CommonDataset):
      METAINFO = METAINFO_SMD

# mapping of ABOship dataset to Singapore done here
@DATASETS.register_module()
class ABOshipsDataset(CocoDataset):
    # meta data of ABOship dataset
    METAINFO = METAINFO_ABO

    # meta data of singapore dataset
    METAINFO_SMD = METAINFO_SMD
    
    # mapping of category ids between ABOships to Singapore
    MAPPING = MAPPING_SMD2ABO
   
    def load_data_list(self):
        data_list = super().load_data_list()
        # print(f"aboship cat_ids: {self.cat_ids}")
        # print(f"aboship: cat2label: {self.cat2label}")
        self.change_cat2label()
        self.change_cat_ids()
        self.change_cat_img_map()
        # print(f"after : {self.cat2label}")
        # print(f"after : {self.cat_ids}")
        return data_list
    
    def change_cat_ids(self):
        new_category_ids = []

        for category_id in self.cat_ids:
            new_category_ids.append(self.MAPPING[category_id])
        self.cat_ids = new_category_ids

    def change_cat2label(self):
        new_cat2label  = dict()
        new_label      = {i:i for i, _ in enumerate(self.METAINFO_SMD["classes"])} 
        for cat_id, _ in self.cat2label.items():
            if self.MAPPING[cat_id] in new_cat2label.keys():
                continue
            new_cat2label[self.MAPPING[cat_id]] = new_label[self.MAPPING[cat_id]]
        self.cat2label = new_cat2label

    def change_cat_img_map(self):
        new_cat_img_map = {}

        for cat_id, imgs in self.cat_img_map.items():
            new_cat_img_map[self.MAPPING[cat_id]] = imgs
        
        self.cat_img_map = new_cat_img_map

    def parse_data_info(self, raw_data_info):
        return super().parse_data_info(raw_data_info)
    
    def filter_data(self):
        return super().filter_data()
    
@DATASETS.register_module()
class ConcatMarineDataset(ConcatDataset):
    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 lazy_init: bool = False,
                 ignore_keys: Union[str, List[str], None] = None):
        self.datasets: List[BaseDataset] = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                self.datasets.append(DATASETS.build(dataset))
            elif isinstance(dataset, BaseDataset):
                self.datasets.append(dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')
        if ignore_keys is None:
            self.ignore_keys = []
        elif isinstance(ignore_keys, str):
            self.ignore_keys = [ignore_keys]
        elif isinstance(ignore_keys, list):
            self.ignore_keys = ignore_keys
        else:
            raise TypeError('ignore_keys should be a list or str, '
                            f'but got {type(ignore_keys)}')

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # if the metainfo of multiple datasets are the same, use metainfo
        # of the first dataset, else the metainfo is a list with metainfo
        # of all the datasets
        is_all_same = True
        self._metainfo_first = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    is_all_same = False
                    break
                if self._metainfo_first[key] != dataset.metainfo[key]:
                    is_all_same = True # TODO: originaly it is false but creates incompatablity with CocoMeterics. 
                    break

        if is_all_same:
            self._metainfo = self.datasets[0].metainfo
        else:
            self._metainfo = [dataset.metainfo for dataset in self.datasets]

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

            if is_all_same:
                self._metainfo.update(
                    dict(cumulative_sizes=self.cumulative_sizes))
            else:
                for i, dataset in enumerate(self.datasets):
                    self._metainfo[i].update(
                        dict(cumulative_sizes=self.cumulative_sizes))
          