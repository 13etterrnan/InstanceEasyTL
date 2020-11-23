import numpy as np
from sklearn import tree

from myEasyTL import myEasyTL


class InstanceEasyTL:
    def __init__(self, iters=10):
        self.iters = iters

    def fit_predict(self, Ttd, Tsd, Ttd_label, Tsd_label, test):
        N = self.iters
        trans_data = np.concatenate((Tsd, Ttd), axis=0)
        trans_label = np.concatenate((Tsd_label, Ttd_label), axis=0)

        row_Tsd = Tsd.shape[0]
        row_Ttd = Ttd.shape[0]
        row_S = test.shape[0]

        test_data = np.concatenate((trans_data, test), axis=0)

        # init weight
        weights_A = np.ones([row_Tsd, 1]) / row_Tsd
        weights_S = np.ones([row_Ttd, 1]) / row_Ttd
        weights = np.concatenate((weights_A, weights_S), axis=0)

        bata = 1 / (1 + np.sqrt(2 * np.log(row_Tsd / N)))


        bata_T = np.zeros([1, N])
        result_label = np.ones([row_Tsd + row_Ttd + row_T, N])

        predict = np.zeros([row_T])

        # print ('params initial finished.')
        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')
        test_data = np.asarray(test_data, order='C')

        for i in range(N):
            P = self.calculate_P(weights, trans_label)

            result_label[:, i] = self.train_classify(trans_data, trans_label,
                                                test_data, P)
            # print ('result,', result_label[:, i], row_Tsd, row_Ttd, i, result_label.shape)

            error_rate = self.calculate_error_rate(Ttd_label, result_label[row_Tsd:row_Tsd + row_Ttd, i],
                                            weights[row_Tsd:row_Tsd + row_Ttd, :])
            # print ('Error rate:', error_rate)
            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                if(i != 0):
                    N = i
                    break  # Prevent overfitting
                # error_rate = 0.001

            bata_T[0, i] = error_rate / (1 - error_rate)

            # Adjust the weight of Ttd
            for j in range(row_Ttd):
                weights[row_Tsd + j] = weights[row_Tsd + j] * np.power(bata_T[0, i],
                                                                (-np.abs(result_label[row_Tsd + j, i] - Ttd_label[j])))

            # Adjust the weight of Tsd
            for j in range(row_Tsd):
                weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - Tsd_label[j]))
        # print bata_T
        for i in range(row_S):
            # Skip the training data label
            left = np.sum(
                result_label[row_Tsd + row_Ttd + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
            right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

            if left >= right:
                predict[i] = 1
            else:
                predict[i] = 0

        return predict


    def calculate_P(self, weights, label):
        total = np.sum(weights)
        return np.asarray(weights / total, order='C')


    def train_classify(self, trans_data, trans_label, test_data, P):
        predict_label = myEasyTL(trans_data, trans_label, test_data, P, "raw")
        return predict_label


    def calculate_error_rate(self, label_R, label_H, weight):
        total = np.sum(weight)
        label_R = label_R.flatten()
        return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))