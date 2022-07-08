import json
import numpy as np
import random
import cv2
import h5py
import copy
import torch
from time import time

from data.object_detection_classes import ObjClass
from utils import language_utils
from data.transparent_data_loader import TransparentDataLoader


class MsCocoDatasetKarpathy:

    TrainSet_ID = 1
    ValidationSet_ID = 2
    TestSet_ID = 3

    def __init__(self,
                 images_path,
                 mscoco_annotations_path,
                 train2014_bboxes_annotations_path,
                 val2014_bboxes_annotations_path,
                 detected_bboxes_hdf5_filepath,
                 limited_num_train_images=None,
                 limited_num_val_images=None,
                 limited_num_test_images=None,
                 dict_min_occurrences=5,
                 verbose=True
                 ):
        super(MsCocoDatasetKarpathy, self).__init__()

        if images_path is None:
            print("Warning: images path set to None, the program will run, but some method in " + str(__file__) + \
                  "may lead to program termination.")
            self.images_path = ""
        else:
            self.images_path = images_path

        start_time = time()

        self.karpathy_train_set = dict()
        self.karpathy_val_set = dict()
        self.karpathy_test_set = dict()

        with open(mscoco_annotations_path, 'r') as f:
            json_file = json.load(f)['images']

        num_train_captions = 0
        num_val_captions = 0
        num_test_captions = 0
        if verbose:
            print("Initializing dataset... ", end=" ")
        for json_item in json_file:
            new_item = dict()
            new_item['img_path'] = self.images_path + json_item['filepath'] + '/img/' + json_item['filename']
            new_item_captions = [item['raw'] for item in json_item['sentences']]
            new_item['img_id'] = json_item['cocoid']
            new_item['captions'] = new_item_captions

            if json_item['split'] == 'train' or json_item['split'] == 'restval':
                self.karpathy_train_set[json_item['cocoid']] = new_item
                num_train_captions += len(json_item['sentences'])
            elif json_item['split'] == 'test':
                self.karpathy_test_set[json_item['cocoid']] = new_item
                num_test_captions += len(json_item['sentences'])
            elif json_item['split'] == 'val':
                self.karpathy_val_set[json_item['cocoid']] = new_item
                num_val_captions += len(json_item['sentences'])

        if verbose:
            print("Initializing boundig boxes...", end=" ")
        self.add_bboxes(train2014_bboxes_annotations_path)
        self.add_bboxes(val2014_bboxes_annotations_path)
        if verbose:
            print("Done.")

        list_train_set = []
        list_val_set = []
        list_test_set = []
        for key in self.karpathy_train_set.keys():
            list_train_set.append(self.karpathy_train_set[key])
        for key in self.karpathy_val_set.keys():
            list_val_set.append(self.karpathy_val_set[key])
        for key in self.karpathy_test_set.keys():
            list_test_set.append(self.karpathy_test_set[key])
        self.karpathy_train_list = list_train_set
        self.karpathy_val_list = list_val_set
        self.karpathy_test_list = list_test_set

        self.train_num_images = len(self.karpathy_train_list)
        self.val_num_images = len(self.karpathy_val_list)
        self.test_num_images = len(self.karpathy_test_list)

        if limited_num_train_images is not None:
            self.karpathy_train_list = self.karpathy_train_list[:limited_num_train_images]
            self.train_num_images = limited_num_train_images
        if limited_num_val_images is not None:
            self.karpathy_val_list = self.karpathy_val_list[:limited_num_val_images]
            self.val_num_images = limited_num_val_images
        if limited_num_test_images is not None:
            self.karpathy_test_list = self.karpathy_test_list[:limited_num_test_images]
            self.test_num_images = limited_num_test_images

        if verbose:
            print("Num train images: " + str(self.train_num_images))
            print("Num val images: " + str(self.val_num_images))
            print("Num test images: " + str(self.test_num_images))

        # Pre-processing part
        tokenized_captions_list = []
        for i in range(self.train_num_images):
            for caption in self.karpathy_train_list[i]['captions']:
                tmp = language_utils.lowercase_and_clean_trailing_spaces([caption])
                tmp = language_utils.add_space_between_non_alphanumeric_symbols(tmp)
                tmp = language_utils.remove_punctuations(tmp)
                tokenized_caption = ['SOS'] + language_utils.tokenize(tmp)[0] + ['EOS']
                tokenized_captions_list.append(tokenized_caption)

        counter_dict = dict()
        for i in range(len(tokenized_captions_list)):
            for word in tokenized_captions_list[i]:
                if word not in counter_dict:
                    counter_dict[word] = 1
                else:
                    counter_dict[word] += 1

        less_than_min_occurrences_set = set()
        for k, v in counter_dict.items():
            if v < dict_min_occurrences:
                less_than_min_occurrences_set.add(k)
        if verbose:
            print("tot tokens " + str(len(counter_dict)) +
                  " less than " + str(dict_min_occurrences) + ": " + str(len(less_than_min_occurrences_set)) +
                  " remaining: " + str(len(counter_dict) - len(less_than_min_occurrences_set)))

        # Sort the dictionary for the sake of repeatibility
        self.num_caption_vocab = 4
        self.max_seq_len = 0
        discovered_words = ['PAD', 'SOS', 'EOS', 'UNK']
        for i in range(len(tokenized_captions_list)):
            caption = tokenized_captions_list[i]
            if len(caption) > self.max_seq_len:
                self.max_seq_len = len(caption)
            for word in caption:
                if (word not in discovered_words) and (not word in less_than_min_occurrences_set):
                    discovered_words.append(word)
                    self.num_caption_vocab += 1

        discovered_words.sort()
        self.caption_word2idx_dict = dict()
        self.caption_idx2word_list = []
        for i in range(len(discovered_words)):
            self.caption_word2idx_dict[discovered_words[i]] = i
            self.caption_idx2word_list.append(discovered_words[i])
        if verbose:
            print("Maximum seq len: " + str(self.max_seq_len))
            print("There are " + str(self.num_caption_vocab) + " vocabs in dict")
            print("Example Dict symbols: ")
            for i in range(10):
                print(self.caption_idx2word_list[i], end=" ")

        self.detected_bboxes_hdf5_filepath = detected_bboxes_hdf5_filepath

        if verbose:
            print("dataset creation time: " + str(time() - start_time) + " sec")

    def add_bboxes(self, annotations_path):
        with open(annotations_path, 'r') as f:
            annotation_dicts = json.load(f)['annotations']
        for entry in annotation_dicts:
            img_id = entry['image_id']
            if img_id in self.karpathy_val_set.keys():
                dict_reference = self.karpathy_val_set
            elif img_id in self.karpathy_test_set.keys():
                dict_reference = self.karpathy_test_set
            else:  # if img_id in train_caption_ids:
                dict_reference = self.karpathy_train_set
            bbox = entry['bbox']
            x, y, weight, height = bbox
            new_format_bbox = (int(x), int(y), int(x + weight), int(y + height))
            if 'bboxes' not in dict_reference[img_id].keys():
                dict_reference[img_id]['bboxes'] = [new_format_bbox]
            else:
                # otherwise add caption to the list of captions
                dict_reference[img_id]['bboxes'].append(new_format_bbox)

    def show_random_train_samples(self, how_many_samples=2):
        hdf5_file = h5py.File(self.detected_bboxes_hdf5_filepath, 'r')

        random_idxes = random.sample(range(self.train_num_images), how_many_samples)
        for idx in random_idxes:
            print("rand index " + str(random_idxes) + " : " + self.karpathy_train_list[idx]['img_path'])
            img = cv2.imread(self.karpathy_train_list[idx]['img_path'])

            img_id = self.karpathy_train_list[idx]['img_id']
            detected_bboxes = hdf5_file['%d_boxes' % img_id][()]
            detected_class = hdf5_file['%d_cls_prob' % img_id][()]
            detected_class = np.argmax(detected_class, axis=1)

            print(self.karpathy_train_list[idx]['captions'])
            print(detected_class)

            for bboxes_idx in range(len(detected_bboxes)):
                label = ObjClass(detected_class[bboxes_idx]).name
                x1, y1, x2, y2 = detected_bboxes[bboxes_idx]
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                green_color = (0, 255, 0)
                cv2.rectangle(img, pt1, pt2, green_color, 1)
                cv2.putText(img, label, (int(x1), int(y1)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 1)
            cv2.imshow(self.karpathy_train_list[idx]['img_path'], img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_random_validation_samples(self, how_many_samples=2):
        hdf5_file = h5py.File(self.detected_bboxes_hdf5_filepath, 'r')

        random_idxes = random.sample(range(self.val_num_images), how_many_samples)
        for idx in random_idxes:
            print("rand index " + str(random_idxes) + " : " + self.karpathy_val_list[idx]['img_path'])
            img = cv2.imread(self.karpathy_val_list[idx]['img_path'])
            print(self.karpathy_val_list[idx]['captions'])

            img_id = self.karpathy_val_list[idx]['img_id']
            detected_bboxes = hdf5_file['%d_boxes' % img_id][()]
            detected_class = hdf5_file['%d_cls_prob' % img_id][()]
            detected_class = np.argmax(detected_class, axis=1)
            for bboxes_idx in range(len(detected_bboxes)):
                label = ObjClass(detected_class[bboxes_idx]).name
                x1, y1, x2, y2 = detected_bboxes[bboxes_idx]
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                green_color = (0, 255, 0)
                cv2.rectangle(img, pt1, pt2, green_color, 1)
                cv2.putText(img, label, (int(x1), int(y1)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 1)
            cv2.imshow(self.karpathy_val_list[idx]['img_path'], img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_image_samples(self, img_idx, dataset_split, show_bounding_boxes=False):
        hdf5_file = h5py.File(self.detected_bboxes_hdf5_filepath, 'r')

        if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
            img_path = self.karpathy_test_list[img_idx]['img_path']
            captions = self.karpathy_test_list[img_idx]['captions']
            img_id = self.karpathy_test_list[img_idx]['img_id']
        elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
            img_path = self.karpathy_val_list[img_idx]['img_path']
            captions = self.karpathy_val_list[img_idx]['captions']
            img_id = self.karpathy_val_list[img_idx]['img_id']
        else:
            img_path = self.karpathy_train_list[img_idx]['img_path']
            captions = self.karpathy_train_list[img_idx]['captions']
            img_id = self.karpathy_train_list[img_idx]['img_id']

        print(str(dataset_split) + " idx: " + str(img_idx) + " path: " + img_path)
        img = cv2.imread(img_path)
        detected_bboxes = hdf5_file['%d_boxes' % img_id][()]
        detected_class = hdf5_file['%d_cls_prob' % img_id][()]
        detected_class = np.argmax(detected_class, axis=1)

        if show_bounding_boxes:
            for bboxes_idx in range(len(detected_bboxes)):
                label = ObjClass(detected_class[bboxes_idx]).name
                x1, y1, x2, y2 = detected_bboxes[bboxes_idx]
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                green_color = (0, 255, 0)
                cv2.rectangle(img, pt1, pt2, green_color, 1)
                cv2.putText(img, label, (int(x1), int(y1) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 1)
        cv2.imshow(img_path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return captions

    def get_image_samples(self, img_idx, dataset_split):
        hdf5_file = h5py.File(self.detected_bboxes_hdf5_filepath, 'r')

        if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
            img_path = self.karpathy_test_list[img_idx]['img_path']
            captions = self.karpathy_test_list[img_idx]['captions']
            img_id = self.karpathy_test_list[img_idx]['img_id']
        elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
            img_path = self.karpathy_val_list[img_idx]['img_path']
            captions = self.karpathy_val_list[img_idx]['captions']
            img_id = self.karpathy_val_list[img_idx]['img_id']
        else:
            img_path = self.karpathy_train_list[img_idx]['img_path']
            captions = self.karpathy_train_list[img_idx]['captions']
            img_id = self.karpathy_train_list[img_idx]['img_id']

        detected_bboxes = hdf5_file['%d_boxes' % img_id][()]
        detected_class = hdf5_file['%d_cls_prob' % img_id][()]
        detected_class = np.argmax(detected_class, axis=1)
        list_of_bboxes = []
        list_of_labels = []
        for bboxes_idx in range(len(detected_bboxes)):
            label = ObjClass(detected_class[bboxes_idx]).name
            x1, y1, x2, y2 = detected_bboxes[bboxes_idx]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            list_of_bboxes.append((pt1, pt2, x1, y1, x2, y2))
            list_of_labels.append(label)

        image_data = {'img_id': img_id, 'captions': captions, 'img_path': img_path,
                      'bboxes_coord': list_of_bboxes, 'bboxes_label': list_of_labels}

        return image_data

    def get_all_images_captions(self, dataset_split):
        all_image_references = []

        if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
            dataset = self.karpathy_test_list
        elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
            dataset = self.karpathy_val_list
        else:
            dataset = self.karpathy_train_list

        for img_idx in range(len(dataset)):
            all_image_references.append(dataset[img_idx]['captions'])
        return all_image_references

    def get_eos_token_idx(self):
        return self.caption_word2idx_dict['EOS']

    def get_sos_token_idx(self):
        return self.caption_word2idx_dict['SOS']

    def get_pad_token_idx(self):
        return self.caption_word2idx_dict['PAD']

    def get_unk_token_idx(self):
        return self.caption_word2idx_dict['UNK']

    def get_eos_token_str(self):
        return 'EOS'

    def get_sos_token_str(self):
        return 'SOS'

    def get_pad_token_str(self):
        return 'PAD'

    def get_unk_token_str(self):
        return 'UNK'


class MsCocoDataLoader(TransparentDataLoader):

    """
    DataLoader for MSCOCO

    num_procs: number of processes, this data loader support multi gpu processing in the
        1 process - 1 gpu paradigm, normally a distributed trained would require separate dataloader
        for each process/gpu, in this case we want to split
        the batch size along multiple gpus and having them working concurrently on the same virtual batch
    rank: process and gpu identifier
    array_of_init_seeds: a seed for each epoch
    """
    NOT_DEFINED = -1

    def __init__(self, mscoco_dataset,
                       array_of_init_seeds,
                       batch_size, rank=0, num_procs=1,
                       dataloader_mode='caption_wise',
                       verbose=False):
        super(TransparentDataLoader, self).__init__()
        assert (dataloader_mode == 'caption_wise' or dataloader_mode == 'image_wise'), \
            "dataloader_mode must be either caption_wise or image_wise"

        self.mscoco_dataset = mscoco_dataset

        self.dataloader_mode = dataloader_mode

        self.num_procs = num_procs
        self.rank = rank

        self.epoch_it = 0
        # multiply array of init_seeds for safety reason
        self.array_of_init_seeds = array_of_init_seeds * 10
        self.max_num_epoch = len(array_of_init_seeds)

        self.max_num_regions = None

        self.batch_size = batch_size

        self.num_procs = num_procs
        self.num_batches = MsCocoDataLoader.NOT_DEFINED
        self.batch_it = []
        self.image_idx_x = []
        self.caption_y = []
        for idx_proc in range(num_procs):
            self.batch_it.append(0)
            self.image_idx_x.append([])
            self.caption_y.append([])

        self.hdf5_features_file = h5py.File(self.mscoco_dataset.detected_bboxes_hdf5_filepath, 'r')

        self.set_epoch_it(epoch=0, verbose=verbose)

    def init_epoch(self, epoch_it, verbose=False):
        init_timer_start = time()

        # set the seed
        random.seed(self.array_of_init_seeds[epoch_it])

        if self.dataloader_mode == 'caption_wise':
            self.batch_it = []
            self.image_idx_x = []
            self.caption_y = []
            for idx_proc in range(self.num_procs):
                self.batch_it.append(0)
                self.image_idx_x.append([])
                self.caption_y.append([])

            img_idx_caption_id_pair_list = []
            for img_idx in range(self.mscoco_dataset.train_num_images):
                num_captions = len(self.mscoco_dataset.karpathy_train_list[img_idx]['captions'])
                for caption_id in range(num_captions):
                    img_idx_caption_id_pair_list.append((img_idx, caption_id))
            random.shuffle(img_idx_caption_id_pair_list)

            # the number of pairs must be multiple of batch_size * world_size
            # so every processor has the same number of batches
            tailing_elements = len(img_idx_caption_id_pair_list) % (self.batch_size * self.num_procs)
            if tailing_elements != 0:
                img_idx_caption_id_pair_list = img_idx_caption_id_pair_list[:-tailing_elements]

            image_idx_batch = []
            caption_y_batch = []
            for idx_proc in range(self.num_procs):
                image_idx_batch.append([])
                caption_y_batch.append([])
            i = 0
            while i < len(img_idx_caption_id_pair_list):
                for idx_proc in range(self.num_procs):
                    img_idx, caption_id = img_idx_caption_id_pair_list[i]
                    image_idx_batch[idx_proc].append(img_idx)
                    preprocessed_caption = self.preprocess(self.mscoco_dataset.karpathy_train_list[img_idx]['captions'][caption_id])
                    caption_y_batch[idx_proc].append(preprocessed_caption)
                    i += 1
                if i % self.batch_size == 0:
                    for idx_proc in range(self.num_procs):
                        self.image_idx_x[idx_proc].append(image_idx_batch[idx_proc])
                        self.caption_y[idx_proc].append(caption_y_batch[idx_proc])
                        image_idx_batch[idx_proc] = []
                        caption_y_batch[idx_proc] = []

            self.num_batches = len(self.image_idx_x[0])

            for idx_proc in range(self.num_procs):
                self.batch_it[idx_proc] = 0
        else:  # image_wise
            self.batch_it = []
            self.image_idx_x = []
            for idx_proc in range(self.num_procs):
                self.batch_it.append(0)
                self.image_idx_x.append([])

            img_idxes_list = list(range(self.mscoco_dataset.train_num_images))
            random.shuffle(img_idxes_list)

            tailing_elements = len(img_idxes_list) % (self.batch_size * self.num_procs)
            if tailing_elements != 0:
                img_idxes_list = img_idxes_list[:-tailing_elements]

            image_idx_batch = []
            for idx_proc in range(self.num_procs):
                image_idx_batch.append([])
            i = 0
            while i < len(img_idxes_list):
                for idx_proc in range(self.num_procs):
                    img_idx = img_idxes_list[i]
                    image_idx_batch[idx_proc].append(img_idx)
                    i += 1
                if i % self.batch_size == 0:
                    for idx_proc in range(self.num_procs):
                        self.image_idx_x[idx_proc].append(image_idx_batch[idx_proc])
                        image_idx_batch[idx_proc] = []
            self.num_batches = len(self.image_idx_x[0])
            for idx_proc in range(self.num_procs):
                self.batch_it[idx_proc] = 0

        if verbose:
            print(str(self.rank) + "] " + __name__ + ") Dataset epoch initialization " + str(
                time() - init_timer_start) + " s elapsed")
            print(str(self.rank) + "] " + __name__ + ") How many batches " + str(self.num_batches))

    def get_next_batch(self, verbose=False, get_also_image_idxes=False, get_also_image_path=False, get_also_obj_labels=False):

        if self.batch_it[self.rank] >= self.num_batches:
            if verbose:
                print("Proc: " + str(self.rank) + " re-initialization")
            self.epoch_it += 1
            if self.epoch_it >= len(self.array_of_init_seeds):
                raise Exception("Please increase number of random seed in the array of initialization seed.")

            self.init_epoch(epoch_it=self.epoch_it, verbose=verbose)

        img_id_batch = []
        img_idx_batch = self.image_idx_x[self.rank][self.batch_it[self.rank]]
        for i in range(len(img_idx_batch)):
            img_idx = img_idx_batch[i]
            img_id_batch.append(self.mscoco_dataset.karpathy_train_list[img_idx]['img_id'])
        batch_x, \
        batch_x_num_pads = self.get_PADDED_img_batch(img_id_batch)

        if self.dataloader_mode == 'caption_wise':
            # convert captions to indexes
            batch_caption_y_as_string = copy.copy(self.caption_y[self.rank][self.batch_it[self.rank]])
            batch_caption_y_encoded = language_utils. \
                convert_allsentences_word2idx(batch_caption_y_as_string,
                                              self.mscoco_dataset.caption_word2idx_dict)
            batch_y, \
            batch_y_num_pads = language_utils. \
                add_PAD_according_to_batch(batch_caption_y_encoded,
                                           self.mscoco_dataset.get_pad_token_idx())
            batch_y = torch.tensor(batch_y)
        else:  # image_wise
            batch_y = [self.mscoco_dataset.karpathy_train_list[img_idx]['captions'] for img_idx in img_idx_batch]

        if verbose:
            mean_src_len = int(
                sum([(len(batch_x[i]) - batch_x_num_pads[i]) for i in range(len(batch_x))]) / len(batch_x))
            if self.dataloader_mode == 'caption_wise':
                mean_trg_len = int(
                    sum([(len(batch_y[i]) - batch_y_num_pads[i]) for i in range(len(batch_y))]) / len(batch_y))
            else:  # image_wise
                mean_trg_len = \
                    sum([(len(cap.split(' '))) for captions in batch_y for cap in captions]) // sum(
                        [len(captions) for captions in batch_y])
            print(str(self.rank) + "] " + __name__ + ") batch " + str(self.batch_it[self.rank]) + " / " +
                  str(self.num_batches) + " batch_size: " + str(len(batch_x)) + " epoch: " + str(self.epoch_it) +
                  " avg_src_seq_len: " + str(mean_src_len) +
                  " avg_trg_seq_len: " + str(mean_trg_len))

        self.batch_it[self.rank] += 1

        file_path_batch_x = []
        if get_also_image_path:
            for i in range(len(img_idx_batch)):
                idx = img_idx_batch[i]
                file_path_batch_x.append(self.mscoco_dataset.karpathy_train_list[idx]['img_path'])

            if self.dataloader_mode == 'caption_wise':
                return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads, file_path_batch_x
            else:
                return batch_x, batch_y, batch_x_num_pads, file_path_batch_x

        if get_also_obj_labels:
            boxes_labels = self.get_bboxes_labels(img_id_batch)

            if self.dataloader_mode == 'caption_wise':
                return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads, boxes_labels
            else:
                return batch_x, batch_y, batch_x_num_pads, boxes_labels

        if get_also_image_idxes:
            if self.dataloader_mode == 'caption_wise':
                return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads, img_idx_batch
            else:
                return batch_x, batch_y, batch_x_num_pads, img_idx_batch

        if self.dataloader_mode == 'caption_wise':
            return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads
        else:
            return batch_x, batch_y, batch_x_num_pads

    def get_random_samples(self, dataset_split, num_samples=1):
        batch_captions_y_as_string = []
        img_id_batch = []

        if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
            img_idx_batch = random.sample(range(self.mscoco_dataset.test_num_images), num_samples)
        elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
            img_idx_batch = random.sample(range(self.mscoco_dataset.val_num_images), num_samples)
        else:
            img_idx_batch = random.sample(range(self.mscoco_dataset.train_num_images), num_samples)

        for i in range(len(img_idx_batch)):
            img_idx = img_idx_batch[i]

            if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
                caption_id = random.randint(a=0, b=len(self.mscoco_dataset.karpathy_test_list[img_idx]['captions'])-1)
                caption = self.mscoco_dataset.karpathy_test_list[img_idx]['captions'][caption_id]
            elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
                caption_id = random.randint(a=0, b=len(self.mscoco_dataset.karpathy_val_list[img_idx]['captions'])-1)
                caption = self.mscoco_dataset.karpathy_val_list[img_idx]['captions'][caption_id]
            else:
                caption_id = random.randint(a=0, b=len(self.mscoco_dataset.karpathy_train_list[img_idx]['captions'])-1)
                caption = self.mscoco_dataset.karpathy_train_list[img_idx]['captions'][caption_id]

            preprocessed_caption = self.preprocess(caption)

            if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
               batch_captions_y_as_string.append(preprocessed_caption)
               img_id_batch.append(self.mscoco_dataset.karpathy_test_list[img_idx]['img_id'])
            elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
               batch_captions_y_as_string.append(preprocessed_caption)
               img_id_batch.append(self.mscoco_dataset.karpathy_val_list[img_idx]['img_id'])
            else:
               batch_captions_y_as_string.append(preprocessed_caption)
               img_id_batch.append(self.mscoco_dataset.karpathy_train_list[img_idx]['img_id'])

        batch_x, \
        batch_x_num_pads = self.get_PADDED_img_batch(img_id_batch)

        batch_caption_y_encoded = language_utils. \
            convert_allsentences_word2idx(batch_captions_y_as_string,
                                          self.mscoco_dataset.caption_word2idx_dict)
        batch_y, \
        batch_y_num_pads = language_utils. \
            add_PAD_according_to_batch(batch_caption_y_encoded,
                                       self.mscoco_dataset.get_pad_token_idx())
        batch_y = torch.tensor(batch_y)

        return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads, img_idx_batch

    def get_PADDED_img_batch(self, img_id_list, verbose=False):
        start_time = time()

        list_of_bboxes_tensor = []
        list_of_num_bboxes = []
        for img_id in img_id_list:
            bboxes_numpy_tensor = self.hdf5_features_file['%d_features' % img_id][()]
            bboxes_tensor = torch.tensor(bboxes_numpy_tensor)
            list_of_bboxes_tensor.append(bboxes_tensor)
            list_of_num_bboxes.append(len(bboxes_numpy_tensor))

        padded_batch_of_bboxes_tensor = torch.nn.utils.rnn.pad_sequence(list_of_bboxes_tensor, batch_first=True)

        list_of_num_pads = []
        max_seq_len = max([length for length in list_of_num_bboxes])
        for i in range(len(list_of_num_bboxes)):
            list_of_num_pads.append(max_seq_len - list_of_num_bboxes[i])

        if verbose:
            time_spent_batching = (time() - start_time)
            print("Time spent batching: " + str(time_spent_batching) + " s")

        return padded_batch_of_bboxes_tensor, list_of_num_pads

    def get_bboxes_by_idx(self, img_idx, dataset_split):
        if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
            img_id = self.mscoco_dataset.karpathy_test_list[img_idx]['img_id']
            bboxes_tensor = torch.tensor(self.hdf5_features_file['%d_features' % img_id][()])
        elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
            img_id = self.mscoco_dataset.karpathy_val_list[img_idx]['img_id']
            bboxes_tensor = torch.tensor(self.hdf5_features_file['%d_features' % img_id][()])
        else:
            img_id = self.mscoco_dataset.karpathy_train_list[img_idx]['img_id']
            bboxes_tensor = torch.tensor(self.hdf5_features_file['%d_features' % img_id][()])
        return bboxes_tensor

    def get_all_image_captions_by_idx(self, img_idx, dataset_split):
        if dataset_split == MsCocoDatasetKarpathy.TestSet_ID:
            caption_list = self.mscoco_dataset.karpathy_test_list[img_idx]['captions']
        elif dataset_split == MsCocoDatasetKarpathy.ValidationSet_ID:
            caption_list = self.mscoco_dataset.karpathy_val_list[img_idx]['captions']
        else:
            caption_list = self.mscoco_dataset.karpathy_train_list[img_idx]['captions']

        return caption_list

    def get_bboxes_labels(self, img_id_list):
        detected_classes = []
        for i in range(len(img_id_list)):
            img_id = img_id_list[i]
            detected_class = self.hdf5_features_file['%d_cls_prob' % img_id][()]
            detected_classes.append(list(np.argmax(detected_class, axis=1)))

        return detected_classes

    def set_epoch_it(self, epoch, verbose=False):
        assert (epoch < len(self.array_of_init_seeds)), "requested epoch higher than the maximum: " + str(len(self.array_of_init_seeds))
        self.epoch_it = epoch
        self.init_epoch(epoch_it=self.epoch_it, verbose=verbose)

    def get_epoch_it(self):
        return self.epoch_it

    def get_num_epoch(self):
        return self.max_num_epoch

    def get_num_batches(self):
        return self.num_batches

    def set_batch_it(self, batch_it):
        self.batch_it[self.rank] = batch_it

    def get_batch_it(self):
        return self.batch_it[self.rank]

    def change_batch_size(self, batch_size, verbose):
        self.batch_size = batch_size
        self.set_epoch_it(epoch=0, verbose=verbose)
        self.set_batch_it(batch_it=0)

    def get_batch_size(self):
        return self.batch_size

    def save_state(self):
        return {'batch_it': self.batch_it[self.rank],
                'epoch_it': self.epoch_it,
                'batch_size': self.batch_size,
                'array_of_init_seed': self.array_of_init_seeds}

    def load_state(self, state):
        self.array_of_init_seeds = state['array_of_init_seed']
        self.batch_size = state['batch_size']
        self.set_epoch_it(state['epoch_it'])
        self.batch_it[self.rank] = state['batch_it']

    def preprocess(self, caption):
        caption = language_utils.lowercase_and_clean_trailing_spaces([caption])
        caption = language_utils.add_space_between_non_alphanumeric_symbols(caption)
        caption = language_utils.remove_punctuations(caption)
        caption = [self.mscoco_dataset.get_sos_token_str()] + language_utils.tokenize(caption)[0] + \
                  [self.mscoco_dataset.get_eos_token_str()]
        # replace oovs with [UNK]
        preprocessed_tokenized_caption = []
        for word in caption:
            if word not in self.mscoco_dataset.caption_word2idx_dict.keys():
                preprocessed_tokenized_caption.append(self.mscoco_dataset.get_unk_token_str())
            else:
                preprocessed_tokenized_caption.append(word)
        return preprocessed_tokenized_caption

    def preprocess_list(self, caption_list):
        for i in range(len(caption_list)):
            caption_list[i] = self.preprocess(caption_list[i])
        return caption_list

