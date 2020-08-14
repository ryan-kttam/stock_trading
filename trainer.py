from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class learner(object):

    def __init__(self, symbol):
        self.symbol = symbol
        pass


    def load_scale_data(self, data, split_days):
        test_days = split_days
        test_y = data[-test_days:]['y']
        train_y = data[:-test_days]['y']
        self.all_y = data['y']
        test_x = data[-test_days:].drop('y', axis=1)
        train_x = data[:-test_days].drop('y', axis=1)

        scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_x)
        self.scaler = scaler

        scaled_train = scaler.transform(train_x)
        scaled_test = scaler.transform(test_x)
        self.all_x = scaler.transform(data.drop('y', axis=1))

        self.scaled_train = scaled_train
        self.scaled_test = scaled_test
        self.test_y = test_y
        self.train_y = train_y


    def grid_search(self, tree_range=(50,210), tree_increment=20, depth_range=(3,21), depth_increment=1):
        num_trees = [i for i in range(tree_range[0], tree_range[1], tree_increment)]
        depths = [i for i in range(depth_range[0], depth_range[1],depth_increment)]

        best_out = 0
        best_in = 0
        n_tree = 0
        n_depth = 0
        for tree in num_trees:
            for d in depths:
                rf = RandomForestClassifier(n_estimators=tree, max_depth=d)
                rf.fit(X=self.scaled_train, y=self.train_y)

                in_sample = pd.Series(rf.predict(self.scaled_train), index=self.train_y.index)
                subset = (self.train_y != 0) | (in_sample != self.train_y)
                in_sample_accuracy = sum(in_sample[subset] == self.train_y[subset]) / len(in_sample[subset])

                out_sample = pd.Series(rf.predict(self.scaled_test), index=self.test_y.index)
                subset = (self.test_y != 0) | (out_sample != self.test_y)
                out_sample_accuracy = sum(out_sample[subset] == self.test_y[subset]) / len(out_sample[subset])

                if out_sample_accuracy >= best_out:
                    if in_sample_accuracy > best_in:
                        print([tree, d, in_sample_accuracy, out_sample_accuracy])
                        best_out = out_sample_accuracy
                        best_in = in_sample_accuracy
                        n_tree = tree
                        n_depth = d
        self.best_tree = n_tree
        self.best_depth = n_depth
        self.rf = RandomForestClassifier(n_estimators=n_tree, max_depth=n_depth)
        self.rf.fit(X=self.all_x, y=self.all_y)



