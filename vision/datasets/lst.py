import numpy as np
import pandas as pd
import logging
from pathlib import Path
import cv2
import psutil
import warnings

class LstDataset:

    def __init__(
            self, 
            lst_file,
            class_names=None,
            transform=None, 
            target_transform=None
            ):
        self.lstfile= lst_file
        self.transform = transform
        self.target_transform = target_transform
        print('Reading file to infer stuff...')
        with open(lst_file, 'r') as f:
            max_label = 0
            self.ids = []
            for i, line in enumerate(f):
                line = line.split('\t')
                if i==0:
                    self.len_of_header, len_of_label \
                        = [int(x) for x in line[1:3]]
                    #len_of_header is arbitrary, 
                    #len_of_label is not
                    assert len_of_label == 5
                max_label = max(
                    [int(float(x)) for x in \
                        line[(self.len_of_header+1):-1:5]
                ])
                self.no_of_lines = i
        print('Stuff inferred!')
        # Define class_dict
        if class_names is None:
            self.class_dict = {f'class_{i}': i for i in range(max_label)}
        else:
            with open(class_names, 'r') as f:
                names = f.readlines()
                if (len(names)-1) < max_label:
                    raise ValueError('Class names not compatible with lst_file')
            self.class_dict = {name: i for i, names in enumerate(names)}

    def __len__(self):
        return self.no_of_lines
    
    def __getitem__(self, idx):
        line = self._read_line
        boxes, labels = self._get_annotation(line)
        image = self._read_image(line)
        if not self.transform is None:
            image, boxes, labels = self.transform(
                image, boxes, labels
                )
        if not self.target_transform is None:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def _read_line(self, idx):
        with open(self.lstfile, 'r') as f:
            line = f.readline(idx).split('\t')
            f.seek(0)
        return line

    def _get_annotation(self, line):
        annot = line[(self.len_of_header+1):-1]
        boxes = [annot[x+1:x+5] for x in range(0, len(annot), 5)]
        labels = [int(float(x)) for x in annot[::5]]
        return (
            np.array(boxes, dtype=np.float32), 
            np.array(labels, dtype=np.int64)
            )
    def _read_image(self, line):
        image = cv2.imread(line[-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image