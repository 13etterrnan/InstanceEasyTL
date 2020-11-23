import scipy.stats
import numpy as np
import time

from InstanceEasyTL import InstanceEasyTL
from LoadMyData import LoadMyData

if __name__ == "__main__":

    for times in range(10):
        # subjects' name
        subjects = np.array(
            ['CASSTE', 'SALSTE', 'ANZALE', 'ARCALE', 'BORFRA', 'BORGIA', 'CALGIO', 'CILRAM', 'CULLEO', 'DESTER',
             'DIFANT', 'MARFRA',
             'SCAEMI', 'VALNIC', 'VITSIM'])

        num_of_subjects = len(subjects)
        windows = 0.5
        num_of_iter = 30
        split_rate = 0.3
        # path of subjects
        path_processed_subjects = '' 

        for index_of_subject in range(num_of_subjects):
            # get train subjects and test subject 
            this_subject = subjects[index_of_subject]
            print('\n\n=================' + this_subject + '\t start====================\n')
            other_subjects = []
            for index_of_another in range(num_of_subjects):
                another_subject = subjects[index_of_another]
                if (not (id(another_subject) == id((this_subject)))):
                    other_subjects.append(another_subject)

            # load data
            load_module = LoadMyData(path_processed_subjects, this_subject, other_subjects)
            [one, label_of_one] = load_module.get_one_data()
            [others, label_of_others] = load_module.get_others_data()

            # z-score
            one = one / np.tile(np.sum(one, axis=1).reshape(-1, 1), [1, one.shape[1]])
            one = scipy.stats.mstats.zscore(one)
            others = others / np.tile(np.sum(others, axis=1).reshape(-1, 1), [1, others.shape[1]])
            others = scipy.stats.mstats.zscore(others)

            # get Tsd Ttd and S
            [Ttd, Ttd_label, S, S_label] = load_module.split_train_test_data(one, label_of_one, split_rate)
            Tsd = others
            Tsd_label = label_of_others

            # classify
            t0 = time.time()
            my_model = InstanceEasyTL(num_of_iter)
            result_label = my_model.fit_predict(Ttd, Tsd, Ttd_label, Tsd_label, S)
            t1 = time.time()
            dur = t1 - t0
            print("running time：\t{0:.0f} minutes: {1:.0f}seconds\n ".format(dur // 60, dur % 60))

            # indicators
            from sklearn import metrics

            S_label = S_label.reshape(S_label.shape[0])
            Accuracy = metrics.accuracy_score(S_label, result_label)
            Recall = metrics.recall_score(S_label, result_label, average='binary')
            Precision = metrics.precision_score(S_label, result_label, average='binary')
            F1score = metrics.f1_score(S_label, result_label, average='binary')
            print("Accuracy:\t {0:.2%}\nRecall:\t{1:.2%}\nPrecision:\t{2:.2%}\nF1score:\t{3:.2%}\n".format(Accuracy,
                                                                                                           Recall,
                                                                                                           Precision,                                                                                                          Precision,
                                                                                                           F1score))

