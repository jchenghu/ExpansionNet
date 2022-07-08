import csv
import h5py
import numpy as np
import argparse
import base64
import sys

csv.field_size_limit(sys.maxsize)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Features generation')
    parser.add_argument('--tsv_features_path', type=str, default='./trainval/')
    parser.add_argument('--hdf5_dst_path', type=str, default='./github_ignore_material/raw_data/mscoco2014_features.hdf5')
    args = parser.parse_args()

    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'bboxes', 'features']

    src_file_path = args.tsv_features_path
    files_path = [ src_file_path + 'karpathy_test_resnet101_faster_rcnn_genome.tsv', #    1.7 GB
                   src_file_path + 'karpathy_train_resnet101_faster_rcnn_genome.tsv.0', #     19.7 GB
                   src_file_path + 'karpathy_train_resnet101_faster_rcnn_genome.tsv.1', #     19.8 GB
                   src_file_path + 'karpathy_val_resnet101_faster_rcnn_genome.tsv'    # 1.7 GB
                  ]
    hdf5_dst_file = h5py.File(args.hdf5_dst_path, 'w')

    for file_path in files_path:
        how_many_items = 0
        with open(file_path, "rt") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:

                img_id = int(item['image_id'])
                num_boxes = int(item['num_boxes'])

                feature = np.frombuffer(base64.standard_b64decode(item['features']),
                                        dtype=np.float32).reshape(num_boxes, -1)

                hdf5_dst_file.create_dataset(name=str(item['image_id']) + '_features',
                                             data=np.array(feature),
                                             dtype=np.float32)

                how_many_items += 1
                if how_many_items % 1000 == 0:
                    print("Converted: " + str(how_many_items))

        print("Num items in " + str(file_path) + " " + str(how_many_items))

