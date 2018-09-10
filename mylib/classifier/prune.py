import sys

from sklearn.tree import _tree


def prune(inner_tree, index):
    """
    The given tree object is modified to resemble a version pruned at index
    @:param inner_tree: sk-learn tree object
    @:param index: the index of the node at which inner_tree is pruned
    """

    # wenn es 'children' gibt besuche ich die 'children'
    if inner_tree.children_left[index] != _tree.TREE_LEAF:
        prune(inner_tree, inner_tree.children_left[index])

        prune(inner_tree, inner_tree.children_right[index])

        # set node to leaf
        idx_left = inner_tree.children_left[index]
        idx_right = inner_tree.children_right[index]

        inner_tree.children_left[index] = _tree.TREE_LEAF
        inner_tree.children_right[index] = _tree.TREE_LEAF

        inner_tree.n_node_samples[idx_left] = 0
        inner_tree.n_node_samples[idx_right] = 0


    else:
        # wenn es keine 'children' gibt kann ich prunen
        inner_tree.n_node_samples[index] = 0


def determine_alpha(tree):
    """
    Given a regression tree, the relevant penalty scalars gk are determined for pruning. Every
    inner node of the tree is visit to evaluate the penalty scalar gk that would make pruning in each node reasonable.
    The minimum gk is returned
    @:param tree: sk-learn tree object
    @:returns the index and corresponding values of the minimal gk found in the tree (alpha)
    """
    min_gk = sys.maxsize
    min_node_idx = tree.node_count

    # traverse all inner nodes in to find min_gk
    for node_idx in range(tree.node_count):
        # if node is a leaf node, skip node
        if tree.children_left[node_idx] == _tree.TREE_LEAF:
            continue

        # inner node
        node_impurity = tree.n_node_samples[node_idx] * tree.impurity[node_idx]
        subtree_impurity, subtree_leafs = _calc_impurity(tree, node_idx)

        print('IDX: ', node_idx, 'NI: ', node_impurity, 'STI: ', subtree_impurity, 'STL: ', subtree_leafs)

        gk = (node_impurity - subtree_impurity) / (subtree_leafs - 1.)

        if gk < min_gk:
            min_node_idx = node_idx
            min_gk = gk

    return min_node_idx, min_gk


def _calc_impurity(tree, index):
    """
    Calc_impurity is a recursive function for calculating the absolute impurity of any subtree.
    The absolute impurity is calculated by the impurity of every leaf-node scaled with the number of samples per node.
    @:param tree: sk-learn tree object
    @:param index: the index of the root node of the subtree
    @:returns impurity and leaf count of subtree
    """

    # print("index: ", index, " impurity: ", d_tree.tree_.n_node_samples[index] * tree.impurity[index] / 10000000)
    # wenn es 'children' besuche die 'children'
    if tree.children_left[index] != _tree.TREE_LEAF:
        impurity_left, leafs_left = _calc_impurity(tree, tree.children_left[index])
        impurity_right, leafs_right = _calc_impurity(tree, tree.children_right[index])

        return impurity_left + impurity_right, leafs_left + leafs_right

    # wenn es keine 'children' gibt bin ich ein leaf Knoten
    else:
        # print("index: ", index, " cost: ", d_tree.tree_.n_node_samples[index] * tree.impurity[index]/10000000)
        return tree.n_node_samples[index] * tree.impurity[index], 1


