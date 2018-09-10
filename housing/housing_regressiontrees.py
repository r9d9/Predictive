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

if __name__ == '__main__':
    data = pd.read_csv("../data/housing.csv",sep=";")
    #print(data.head())
    data = features.prep.dropnans_newidx(data)
    #print(data.shape)
    data = features.prep.convert_cats(data,"ocean_proximity")

    #####################
    data = data.sample(1000)
    data = data[["latitude","longitude","median_income","median_house_value"]]
    ###################

    train_X, val_X, test_X, train_y, val_y, test_y = features.prep.split_train_val_test(data, "median_house_value",[0.6,0.2,0.2])
    #print(train_X.shape)
    #print(val_X.shape)
    #print(test_X.shape)
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
    samples_in_leaves = small_dTree.apply(train_X[["median_income"]]) # gibt pro Zeile die Zugehörigkeit zum terminal node aus
    print(samples_in_leaves)
    print(pd.value_counts((samples_in_leaves)))
    #use impurity measure?
    print(small_dTree.tree_.children_left[2])
    print(f" child left idx 2: {small_dTree.tree_.children_left[2]}")
    print(f" tree leaf: {_tree.TREE_LEAF}")
    trees.depth_first(small_dTree.tree_, 0)
    print(f" node count: {small_dTree.tree_.node_count}") # total nr of nodes
    print(f" node impurity idx 0: {small_dTree.tree_.impurity[0]}")
    print(f" impurity * n samples at idx 0: {small_dTree.tree_.impurity[0]*small_dTree.tree_.n_node_samples[0]}")
    rss_mean = np.sum((train_y - np.mean(train_y))**2)
    print(f" pred error of mean: {rss_mean}")
    print(f" n node samples idx 2: {small_dTree.tree_.n_node_samples[2]}")

    print(f" n node samples idx 1: {small_dTree.tree_.n_node_samples[1]}")
    print(f" n node samples idx 3: {small_dTree.tree_.n_node_samples[3]}")

    # finding alpha --> best subtree from pruning

    dTree = DecisionTreeRegressor(min_samples_leaf=3)
    dTree.fit(train_X, train_y)

    tree_array = [dTree]

    num_nodes = dTree.tree_.capacity
    index = 0
    alpha = 0
    k = 1

    alpha_array = [alpha]
    num_nodes_array = [num_nodes]

    while num_nodes > 1:
        tree_array.append(copy.deepcopy(tree_array[k - 1]))

        min_node_idx, min_gk = trees.determine_alpha(tree_array[k].tree_)
        alpha_array.append(min_gk)
        trees.prune(tree_array[k].tree_, min_node_idx)

        num_nodes = sum(1 * (tree_array[k].tree_.n_node_samples != 0))
        num_nodes_array.append(num_nodes)
        k += 1
    print("alpha array:")
    print(alpha_array)
    print("numnodes array:")
    print(num_nodes_array)
    error_rates = pd.DataFrame(columns=["tree", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])
    error_rates_tr = pd.DataFrame(columns=["tree", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])

    for t in tree_array:
        pred = t.predict(val_X)
        pred_training = t.predict(train_X)
        idx = tree_array.index(t)
        new_row = pd.concat([pd.Series(idx,name="tree"),evaluation.save_errors(val_y, pred)],axis=1)
        new_row_tr = pd.concat([pd.Series(idx,name="tree"),evaluation.save_errors(train_y, pred_training)],axis=1)
        error_rates = error_rates.append(new_row,sort=False)
        error_rates_tr = error_rates_tr.append(new_row_tr,sort=False)
    print("sorted error rates for val:\n")
    err_sorted = error_rates.sort_values(["R2","RMSE"],ascending=[False,True])
    print(err_sorted)
    best_tree_nr = err_sorted.iloc[0,0]

    best_tree = tree_array[best_tree_nr]
    fig2, axes2 = plt.subplots(3,2,figsize=(10,12))
    ax2 = axes2.flatten()
    for col in error_rates.iloc[:,1:6].columns.values:
        axnr = error_rates.columns.get_loc(col) -1
        ax2[axnr].set_title(col)
        ax2[axnr].scatter(alpha_array, error_rates[col],label="val")
        ax2[axnr].scatter(alpha_array, error_rates_tr[col],label="train")
        ax2[axnr].legend()
        ax2[axnr].set_xlim([0, 0.6e+12])
        ax2[axnr].set_xlabel("alpha")

    fig2.tight_layout()
    a = viz.ScrollableWindow(fig2)

    fig3, axes3 = plt.subplots(3, 2, figsize=(10, 12))
    ax3 = axes3.flatten()
    for col in error_rates.iloc[:, 1:6].columns.values:
        axnr = error_rates.columns.get_loc(col) - 1
        ax3[axnr].set_title(col)
        ax3[axnr].set_xlabel("subtree")
        ax3[axnr].scatter(error_rates["tree"],error_rates[col],label="val")
        ax3[axnr].scatter(error_rates_tr["tree"], error_rates_tr[col],label="train")
        ax3[axnr].legend()

    fig3.tight_layout()
    b = viz.ScrollableWindow(fig3)



    random_dTree = RandomForestRegressor(min_samples_leaf=3)
    random_dTree.fit(train_X, train_y)
    rand_pred_val = random_dTree.predict(val_X)
    rand_pred_tr = random_dTree.predict(train_X)
    rand_error_rates_val = pd.DataFrame(columns=["type", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])
    rand_error_rates_tr = pd.DataFrame(columns=["type", "MSE", "RMSE", "R2", "RMSE % of mean", "Cali"])

    rand_new_row = pd.concat([pd.Series("val",name="type"),evaluation.save_errors(val_y, rand_pred_val)],axis=1)
    rand_new_row_tr = pd.concat([pd.Series("train",name="type"),evaluation.save_errors(train_y, rand_pred_tr)],axis=1)
    rand_error_rates = rand_error_rates_val.append(rand_new_row,sort=False)
    rand_error_rates_tr = rand_error_rates_tr.append(rand_new_row_tr,sort=False)
    print("pruned error rates val:")
    print(error_rates.loc[error_rates["tree"] == best_tree_nr,:])
    print("pruned error rates train:")
    print(error_rates_tr.loc[error_rates["tree"] == best_tree_nr,:])
    print("rand error rates val:")
    print(rand_error_rates)
    print("rand error rates train:")
    print(rand_error_rates_tr)


    fig5, axes5 = plt.subplots(3, 2, figsize=(10, 12))
    ax5 = axes5.flatten()
    for col in error_rates.iloc[:, 1:6].columns.values:
        axnr = error_rates.columns.get_loc(col) - 1
        ax5[axnr].set_title(col)
        ax5[axnr].set_xlabel("subtree")
        ax5[axnr].scatter("pruned",error_rates.loc[error_rates["tree"] == best_tree_nr,col],label="pruned val")
        ax5[axnr].scatter("pruned", error_rates_tr.loc[error_rates["tree"] == best_tree_nr,col],label="pruned train")
        ax5[axnr].scatter("rand", rand_error_rates[col], label="rand val")
        ax5[axnr].scatter("rand", rand_error_rates_tr[col], label="rand train")

        ax5[axnr].legend()

    fig5.tight_layout()
    b = viz.ScrollableWindow(fig5)

"""
    fig4, axes4 = plt.subplots(3, 2, figsize=(10, 12))
    ax4 = axes4.flatten()
    for col in error_rates.iloc[:, 1:6].columns.values:
        axnr = error_rates.columns.get_loc(col) - 1
        ax4[axnr].set_title(col)
        ax4[axnr].set_xlabel("num nodes")
        ax4[axnr].scatter(num_nodes_array,error_rates[col])

    fig4.tight_layout()
    c = viz.ScrollableWindow(fig4)


    #dForest = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)



    
    fig, axes = plt.subplots(10,2,figsize=(10, 30))
    ax = axes.flatten()
    ax[0].set_title('Real Test Values')
    ax[0].scatter(x=test_X['longitude'], y=test_X['latitude'], c=test_y, s=10, cmap='hot');
    error_rates = pd.DataFrame(columns =["max_depth","MSE","RMSE","R2","RMSE % of mean","Cali"])

    
    for max_depth in range(1,20):
        dTree = DecisionTreeRegressor(max_depth=max_depth)
        dTree.fit(train_X, train_y)
        pred = dTree.predict(test_X)
        title = "Tree Prediction for max_depth = " + str(max_depth)
        ax[max_depth].set_title(title)
        ax[max_depth].scatter(x=test_X['longitude'], y=test_X['latitude'], c=pred, s=10, cmap='hot');
        new_row = pd.concat([pd.Series(max_depth,name="max_depth"),evaluation.save_errors(test_y, pred)],axis=1)
        error_rates = error_rates.append(new_row,sort=False)
    fig.tight_layout()
    a = viz.ScrollableWindow(fig)
    print(error_rates)
    plt.close("all")

    fig2, axes2 = plt.subplots(3,2,figsize=(10,12))
    ax2 = axes2.flatten()
    for col in error_rates.iloc[:,1:6].columns.values:
        axnr = error_rates.columns.get_loc(col) -1
        ax2[axnr].set_title(col)
        ax2[axnr].set_xlabel("max_depth")
        ax2[axnr].plot(error_rates["max_depth"],error_rates[col])

    fig2.tight_layout()
    b = viz.ScrollableWindow(fig2)
    #plt.show()
    plt.close('all')
    """
