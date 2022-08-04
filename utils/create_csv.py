#!/usr/bin/env

""" Create csv file for UCF101 dataset
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2020'


import csv
import os.path as ops
import os

path_to_videos = '/BS/xian18/work/data/ucf101/video'

dataset_output = '/BS/kukleva/work/data/ucf/sp1m_kt_spat9060'
os.makedirs(ops.join(dataset_output), exist_ok=True)

csv_path = ops.join('/BS/kukleva/work/data/ucf/data_splits', 'feature_extract.csv')

rows = [['root_path', 'file_name', 'label']]
counter = 0
for f_root, f_dirs, files in os.walk(path_to_videos):
    for filename in files:
        if not filename.endswith('avi'):
            continue
        counter += 1
        if counter % 100 == 0:
            print(counter)

        label = f_root.split('/')[-1]
        rows.append([f_root, filename, label])

with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print('Written file %s' % csv_path)
print('rows %d' % len(rows))
