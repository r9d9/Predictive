import features.prep
import pandas as pd
import numpy as np
import trees
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

data = pd.read_csv("../data/housing.csv", sep=";")
# print(data.head())
data = features.prep.dropnans_newidx(data)
# print(data.shape)
data = features.prep.convert_cats(data, "ocean_proximity")

#####################
data = data.sample(1000)
data["median_house_value"] = data["median_house_value"] / 100000
# data = data[["latitude","longitude","median_income","median_house_value"]]
###################

train_X, val_X, test_X, train_y, val_y, test_y = features.prep.split_train_val_test(data, "median_house_value",
                                                                                    [0.6, 0.2, 0.2])
# print(train_X.shape)
# print(val_X.shape)
# print(test_X.shape)
# train_X, test_X, train_y, test_y = features.prep.split_train_test(data, "median_house_value")
# scaling is not necessary for decision trees
small_dTree = DecisionTreeRegressor(max_depth=2)
small_dTree.fit(train_X[["median_income"]], train_y)
small_dTree_pred = small_dTree.predict(train_X[["median_income"]])
trees.tree_to_code(small_dTree, ['median_income'])
print(f" feature: {small_dTree.tree_.feature}")
print(f" child left: {small_dTree.tree_.children_left}")
print(f" child left idx 4: {small_dTree.tree_.children_left[4]}")
print(f" child right: {small_dTree.tree_.children_right}")
print(f" tree value idx 2: {small_dTree.tree_.value[2]}")
print(f" ")
rss = np.sum((train_y - small_dTree_pred) ** 2)
print(f" rss: {rss}")
thresh_node = small_dTree.tree_.threshold[1]
samples_in_leaves = small_dTree.apply(
    train_X[["median_income"]])  # gibt pro Zeile die Zugehörigkeit zum terminal node aus
print(samples_in_leaves)
print(pd.value_counts((samples_in_leaves)))
# use impurity measure?
print(small_dTree.tree_.children_left[2])
print(f" child left idx 2: {small_dTree.tree_.children_left[2]}")
print(f" tree leaf: {_tree.TREE_LEAF}")
trees.depth_first(small_dTree.tree_, 0)
print(f" node count: {small_dTree.tree_.node_count}")  # total nr of nodes
print(f" node impurity idx 0: {small_dTree.tree_.impurity[0]}")
print(f" impurity * n samples at idx 0: {small_dTree.tree_.impurity[0]*small_dTree.tree_.n_node_samples[0]}")
rss_mean = np.sum((train_y - np.mean(train_y)) ** 2)
print(f" pred error of mean: {rss_mean}")
print(f" n node samples idx 2: {small_dTree.tree_.n_node_samples[2]}")
print(f" n node samples idx 1: {small_dTree.tree_.n_node_samples[1]}")
print(f" n node samples idx 3: {small_dTree.tree_.n_node_samples[3]}")

