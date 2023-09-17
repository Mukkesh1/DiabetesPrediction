from random import randrange
import pandas as pd
import math


# Probability of the accuracy, takes actual values and the predicted values of the test set 
def accuracy_calculation(actual_values, predicted_values):
    value = 0
    # for all the entries in the test set
    for i in range(len(actual_values)):
        if actual_values[i] == predicted_values[i]:
            value += 1
    return value / float(len(actual_values)) * 100.0


# Split the data set into two nodes
def node_split(index, value, dataset):
    left_node, right_node = list(), list()
    for each_entry in dataset:
        #if value of the each entry is less then append to left node else to right node
        if each_entry[index] < value:
            left_node.append(each_entry)
        else:
            right_node.append(each_entry)
    return left_node, right_node


#calculate gini index for each node and calculate the weighted average of child nodes
def gini_index(node_entries, class_values):
    # the total entries present in the nodes
    data_entries = float(sum([len(node) for node in node_entries]))
    gini_score = 0.0
    for node in node_entries:
        node_size = float(len(node))
        # avoid dividing the count of entries by zero
        if node_size == 0:
            continue
        score = 0.0
        for class_val in class_values:
            # probability of each class in each child node 
            probability = [entry[-1] for entry in node].count(class_val) / node_size
            # square the probability
            score += probability * probability
        #calculate the weighted average of the child nodes
        gini_score += (1.0 - score) * (node_size / data_entries)
    return gini_score


def entropy(node_entries, class_values):
    data_entries = float(sum([len(node) for node in node_entries]))
    #entropy_of_parent + = -[len(group) for group in groups * math.log(len(group) for group in groups)]
    entropy = 0.0
    for node in node_entries:
        node_size = float(len(node))
        # avoid dividing the count of entries by zero
        if node_size == 0:
            continue
        score = 0.0
        for class_val in class_values:
            # probability of each class in each child node 
            probability = [entry[-1] for entry in node].count(class_val) / node_size
            # avoid calculating probability of zero 
            if(probability == 0):
                score += 0
            else:
                score += - (probability * math.log(probability))
        # weighted entropy of the child nodes
        entropy += score * (node_size / data_entries)   
    return entropy


#decides the node split based on entropy and gini
def decision_node(train,m_impurity):
    # class values for train data [Positive, Negative]
    class_values = list(set(row[-1] for row in train))
    #Assign prior values and store the index , the value and the node data
    column_index, column_value, data_score, node_data = 999, 999, 999, None
    if( m_impurity == 'gini'):
        # run for all columns except the last 
        for index in range(len(train[0])-1):
            for row in train:
                #get left and right groups 
                groups = node_split(index, row[index], train)
                # get the gini value for each row entry 
                gini = gini_index(groups, class_values)
                #compare the gini with previous entry and if less update the values
                if gini < data_score:
                    column_index, column_value, data_score, node_data = index, row[index], gini, groups
        #return the dictionary with index value and the groups
        return {'index':column_index, 'value':column_value, 'groups':node_data}
    elif (m_impurity =='entropy'):
        for index in range(len(train[0])-1):
            for row in train:
                groups = node_split(index, row[index], train)
                # get the entropy value for each row entry 
                entropy_value = entropy(groups, class_values)
                #compare the entropy with previous entry and if less update the values
                if entropy_value < data_score:
                    column_index, column_value, data_score, node_data = index, row[index], entropy_value, groups
        #return the dictionary with index value and the groups
        return {'index':column_index, 'value':column_value, 'groups':node_data}


#setting the value for leaf node based on the maximum key values
def leaf_node(data_entries):
    leaf = [entry[-1] for entry in data_entries]
    return max(set(leaf), key=leaf.count)


# splits the nodes based on the depth and prune the data to overcome overfit
def split_node(node_data,m_impurity, depth_of_tree, min_data_points, depth):
    #splits the data into left and right nodes
    left_node, right_node = node_data['groups']
    del(node_data['groups'])
    #if there is no requirement of split then create the leaf node
    if not left_node or not right_node:
        node_data['left'] = node_data['right'] = leaf_node(left_node + right_node)
        return
    #validation on the maximum depth a tree can have and if it is equal to 
    #maximum depth a tree can have stop the splitting and create the leaf nodes.
    if depth >= depth_of_tree:
        node_data['left'], node_data['right'] = leaf_node(left_node), leaf_node(right_node)
        return
    # check for the left node and if the data entries are less then the node can have
    # end the split and create a leaf node
    # this condition is added to pre prune the data to over come the overfit as 
    #we are passing values after cross validation
    if len(left_node) <= min_data_points:
        node_data['left'] = leaf_node(left_node)
    else:
        #else split the data 
        node_data['left'] = decision_node(left_node, m_impurity)
        split_node(node_data['left'], m_impurity, depth_of_tree, min_data_points, depth+1)
    # same procedure for right nodes 
    if len(right_node) <= min_data_points:
        node_data['right'] = leaf_node(right_node)
    else:
        node_data['right'] = decision_node(right_node,m_impurity)
        split_node(node_data['right'],m_impurity,depth_of_tree, min_data_points, depth+1)


#Predict the class value for each entry in the test set
def predict_value(tree_node, entry):
    if entry[tree_node['index']] < tree_node['value']:
        if isinstance(tree_node['left'], dict):
            return predict_value(tree_node['left'], entry)
        else:
            return tree_node['left']
    else:
        if isinstance(tree_node['right'], dict):
            return predict_value(tree_node['right'], entry)
        else:
            return tree_node['right']


def decision_algorithm(train,test,m_impurity, depth_of_tree, min_data_points):
    decision_tree = decision_node(train,m_impurity)
    split_node(decision_tree,m_impurity, depth_of_tree, min_data_points, 1)
    predictions = list()
    for entry in test:
        prediction = predict_value(decision_tree, entry)
        predictions.append(prediction)
    return (predictions, decision_tree)

def calculate(filename, cv_fold, depth_of_tree, min_data_points, impurity):
    file=pd.read_csv(filename)
    data=file.values.tolist()
    k_splits = list()
    data_copy = list(data)
    length_of_each_set = int(len(data) / cv_fold)
    for i in range(cv_fold):
        set_list = list()
        while len(set_list) < length_of_each_set:
            index_number = randrange(len(data_copy))
            set_list.append(data_copy.pop(index_number))
        k_splits.append(set_list)
    accuracies = list()
    for split in k_splits:
        train = list(k_splits)
        train.remove(split)
        train = sum(train, [])
        test = list()
        best_tree = []
        best_accuracy = 0
        for row in split:
            data_entry = list(row)
            test.append(data_entry)
            data_entry[-1] = None
        predicted_values, tree = decision_algorithm(train, test, impurity, depth_of_tree, min_data_points)
        actual_values = [row[-1] for row in split] 
        accuracy = accuracy_calculation(actual_values, predicted_values)
        accuracies.append(accuracy)
        if best_accuracy < accuracy:
            best_tree = tree
    return (sum(accuracies)/float(len(accuracies)))

def predict_diabetes(age,gender,polyuria,polydipsia,sudden_weight_loss,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability,delayed_healing,partial_paresis,muscle_stiffness, alopecia, obesity):
    new_entry=list()
    new_entry.append(age)
    new_entry.append(gender)
    new_entry.append(polyuria)
    new_entry.append(polydipsia)
    new_entry.append(sudden_weight_loss)
    new_entry.append(weakness)
    new_entry.append(polyphagia)
    new_entry.append(genital_thrush)
    new_entry.append(visual_blurring)
    new_entry.append(itching)
    new_entry.append(irritability)
    new_entry.append(delayed_healing)
    new_entry.append(partial_paresis)
    new_entry.append(muscle_stiffness)
    new_entry.append(alopecia)
    new_entry.append(obesity)
    best_tree = get_best_tree()
    prediction = predict_value(best_tree,new_entry)
    return prediction

def get_best_tree():
    file=pd.read_csv('/Users/charitrapy/Downloads/diabetes_data_upload.csv')
    data=file.values.tolist()
    #cross validating the whole dataset into k folds 
    sets = 5
    # depth of the decision tree
    depth_of_tree = 6
    # the minimum number of data entries in leaf node to prune the data 
    min_data_points = 5
    # Cross validation of the dataset 
    k_splits = list()
    #copying the dataset
    data_copy = list(data)
    #devide the datsets into k folds 
    length_of_each_set = int(len(data) / sets)
    for i in range(sets):
        set_list = list()
        while len(set_list) < length_of_each_set:
            #randonmly select the index and arrange them into k sets
            index_number = randrange(len(data_copy))
            set_list.append(data_copy.pop(index_number))
        # k_splits has the list of datasets devided into k sets 
        k_splits.append(set_list)

    impurity_measure=['gini','entropy']
    for m_impurity in impurity_measure:
        datasets = k_splits
        accuracies = list()
        #for all the dataset of k folds choose test set and train sets to address the problem of overfitting
        for dataset in datasets:
            # all k folds are assigned as training data set 
            train = list(datasets)
            # removing one data set to choose that has the testing set 
            train.remove(dataset)
            train = sum(train, [])
            test = list()
            # the test set is created without the class label value
            best_tree = []
            best_accuracy = 0
            for row in dataset:
                data_entry = list(row)
                test.append(data_entry)
                data_entry[-1] = None
            #prediction runs for k folds  
            predicted_values, tree = decision_algorithm(train,test,m_impurity, depth_of_tree, min_data_points)
            # actual values for the test set 
            actual_values = [row[-1] for row in dataset]
            #accuracy of each test set 
            accuracy = accuracy_calculation(actual_values, predicted_values)
            # accuracies of k test sets
            if best_accuracy < accuracy:
                best_tree = tree
    return best_tree
    

