#coding=utf-8
import os
import json
import re

from tqdm import tqdm
import glob
#from glob import glob
import six
import numpy as np
from .dataset import Dataset
from .video import Video

class UAV123_10FPSVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(UAV123_10FPSVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

class UAV123_10FPSDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'UAV123', 'UAV20L'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(UAV123_10FPSDataset, self).__init__(name, dataset_root)
        # with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
        #     meta_data = json.load(f)
        self.root_dir = dataset_root
        self.version = name
        assert name == 'UAV123_10fps'
        self._check_integrity(dataset_root)


        # self.anno_files = sorted(glob.glob(
        #     os.path.join(dataset_root, '*/*_gt.txt')))
        # self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        # self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        
        # sequence meta information
        meta_file = os.path.join(
            os.path.dirname(__file__), 'uav123_10fps.json')
        with open(meta_file) as f:
            self.seq_metas = json.load(f)

        # sequence and annotation paths
        self.anno_files = sorted(glob.glob(
            os.path.join(self.root_dir, 'anno/%s/*.txt' % self.version)))
        self.seq_names = [
            os.path.basename(f)[:-4] for f in self.anno_files]
        self.seq_dirs = [os.path.join(
            self.root_dir, 'data_seq/UAV123_10fps/%s' % \
                self.seq_metas[self.version][n]['folder_name'])
            for n in self.seq_names]
        # self.seq_names = [re.sub(r'_.*$', "", n) for n in self.seq_names]

        meta_data=dict()

        for  i in range(len(self.seq_names)):
            start_frame = self.seq_metas[self.version][self.seq_names[i]]['start_frame']
            end_frame = self.seq_metas[self.version][self.seq_names[i]]['end_frame']
            img_files = [os.path.join(self.seq_dirs[i], '%06d.jpg' % f)for f in range(start_frame, end_frame + 1)]
            # img_names = sorted(glob.glob(os.path.join(self.seq_dirs[i], '*.jpg')))
            img_names = [x.split('/UAV123_10fps/')[-1] for x in img_files]
            gt_rect = np.loadtxt(self.anno_files[i], delimiter=',')
            data=dict()
            data['video_dir']=self.seq_dirs[i]
            data['init_rect']=gt_rect[0]
            data['img_names']=img_names
            data['gt_rect']=gt_rect
            data['attr']=0
            meta_data[self.seq_names[i]]=data

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for  video in pbar:
           #pbar.set_postfix_str(video)
            self.videos[video] = UAV123_10FPSVideo(video,
                                          dataset_root+'/data_seq/UAV123_10fps',
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])
        # set attr  
        # attr = []
        # for x in self.videos.values():
        #     attr += x.attr
        # attr = set(attr)
        # self.attr = {}
        # self.attr['ALL'] = list(self.videos.keys())
        # for x in attr:
        #     self.attr[x] = []
        # for k, v in self.videos.items():
        #     for attr_ in v.attr:
        #         self.attr[attr_].append(k)

    def _check_integrity(self, root_dir, version='UAV123_10fps'):
        # sequence meta information
        meta_file = os.path.join(
            os.path.dirname(__file__), 'uav123_10fps.json')
        with open(meta_file) as f:
            seq_metas = json.load(f)
        seq_names = list(seq_metas[version].keys())

        if os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 3:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(
                    root_dir, 'data_seq/UAV123_10fps/%s' % \
                        seq_metas[version][seq_name]['folder_name'])
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

if __name__=='__main__':
    dataset = UAV123_10FPSDataset(name='UAV123_10fps', dataset_root='/home/huyan/disk1/dataset/UAV123_10fps')
