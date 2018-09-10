import pandas as pd
import copy
import viz
import numpy as np
import sklearn.linear_model as lm
import evaluation
import features
import features.prep
import trees
from matplotlib import pyplot as plt
import classifier
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import _tree

pd.set_option('display.max_columns',30)


data = pd.read_csv("data/housing.csv", sep=";")
# print(data.head())
data = features.prep.dropnans_newidx(data)
# print(data.shape)
data = features.prep.convert_cats(data, "ocean_proximity")

#####################
data = data.sample(3000)
data = data[["latitude", "longitude", "median_income", "median_house_value"]]
###################
train_X, val_X, test_X, train_y, val_y, test_y = features.prep.split_train_val_test(data, "median_house_value",[0.6,0.2,0.2])

small_dTree = DecisionTreeRegressor(max_depth=2)
small_dTree.fit(train_X[["median_income"]], train_y)
small_dTree_pred = small_dTree.predict(train_X[["median_income"]])
trees.tree_to_code(small_dTree, ['median_income'])
trees.depth_first(small_dTree.tree_, 0)
print(trees._calc_impurity(small_dTree.tree_,0))
t = small_dTree.tree_
imp = 0
for leaf in [0,1,2,3,4,5,6]:
    imp = (t.n_node_samples[leaf] * t.impurity[leaf])
    print(f"leaf {leaf}: imp: {imp}")

