import codecs
import csv

import numpy as np
import scipy.io as scio
import xlwt as xlwt
from sklearn.model_selection import train_test_split

class LoadMyData(object):
    def __init__(self, path_of_data, this_subject, other_subjects):
        self.path = path_of_data

        data_and_label = scio.loadmat(path_of_data + this_subject)
        self.this_subject_data = data_and_label['data']
        self.this_subject_label = data_and_label['label'].T

        other_subjects_data, other_subjects_label = self.load_other_subjects(other_subjects)
        self.other_subjects_data = other_subjects_data
        self.other_subjects_label = other_subjects_label
    def __len__(self):
        return len(self.label)

    def get_one_data(self):
        return self.this_subject_data, self.this_subject_label
    def get_others_data(self):
        return self.other_subjects_data, self.other_subjects_label

    def load_other_subjects(self, other_subjects):
        other_subjects_data = list()
        other_subjects_label = list()
        for index_of_subject, subject in enumerate(other_subjects):
            tmp_this_subject_data = scio.loadmat(self.path + subject)

            tmp_this_subject_data_list = tmp_this_subject_data['data'].tolist()

            tmp_this_subject_label_list = tmp_this_subject_data['label'].T.tolist()

            if(not other_subjects_data):
                other_subjects_data = tmp_this_subject_data_list
            else:
                other_subjects_data.extend(tmp_this_subject_data_list)

            if(not other_subjects_label):
                other_subjects_label = tmp_this_subject_label_list
            else:
                other_subjects_label.extend(tmp_this_subject_label_list)
        return np.array(other_subjects_data), np.array(other_subjects_label)

    def split_train_test_data(self, data, label_of_data, split_partition):
        x_train, x_test, y_train, y_test = train_test_split(data, label_of_data, test_size=split_partition)
        return x_train, y_train, x_test, y_test
