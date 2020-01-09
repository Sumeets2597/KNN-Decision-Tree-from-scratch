# Implementing KNN & Decision Tree from scratch

The task is to classify images based on which direction the image is oriented in. We need to use different types of classifiers to achieve this task. We have implemented Neural Networks, Decision Tree and K Nearest Neighbors to classify the images given.


## Decision Tree Classifier:
Decision Tree Classifier uses conditions based on a specific feature and a threshold for that feature to makee decisions. For this, it uses ID3. In this algorithm, we calculate the information that we can get by splitting the data by a feature. For this, we calculate the entropy of the splits and then find the difference between the entropy and the total expected information. In our approach, to reduce the computational time, instead of finding the maxium information gain, we compared the entropies of the splits. Since the expected information is a constant value, we could get the same result by comparing the entropies. The split value for the feature was selected by taking 3 random values from the feature and then taking the one that gave the maximum accuracy. The datastructure we used is "Dictionary". The deciding condition was was made the key of the dictionary. The value given to the key is a list. The list stores the children of that node. A recursive call is made to the tree-building function. The recursive depth is decided by the 3 stopping conditions:
i)Max depth reached or only 1 sample in the split or the split has only one class label. 
ii)The class label is decided when the terminating condition is reached. 
iii)The class label that occurs the most of the time in the split is passed as a decision.

The variation of accuracy with the maximum depth is as follows:

| SNo | Max Depth  | Accuracy    | 
|-----|------------|-------------|
| 1   | 3          | 31          | 
| 2   | 5          | 60          | 
| 3   | 7          | 60          | 
| 4   | 10         | 52          | 

For calculating the threshold, we tried different number of random values for a maximum depth of 7. The variation of accuracy is as follows:

| SNo | Random no. | Accuracy    | 
|-----|------------|-------------|
| 1   | 3          | 60          | 
| 2   | 5          | 57          | 
| 3   | 10         | 56          | 
| 4   | 20         | 56          | 

For classifying a given image, we passed the tuple and compared with the root node condition of the decision tree. The check for the condtions goes on until we get a value which is a class label and not a dictionary.

## K Nearest Neighbor Classifier:

The K-nearest neighbors is a non-parametric algorithm which means it makes less assumptions about the data but is computationally slower. It can be used for both classification and regression problems. It lies on only one assumption that things which are similar lies close to each other in space. First, while training the KNN groups all the data points based on the class they belong to. A value of k is provided by the user which can be anything. Whenever a new data point comes while testing, it maps that point to the same space. Now based on the Euclidian distance, it considers the k-closest neighbors around the data point. Now the class to which all these k-neighbors belong is calculated and the class having maximum number is selected as the predicted class.
We tried approach with all list implementation but it was consuming too much time so we changed the approach a bit, used dictionary instead. As knn is lazy algorithm and it trains while testing itself, i.e it calculates the distance of each test point from all the training points. So we made a replica of training data into the model_nearest file. Our code takes 316 seconds i.e approx 5 min to run the entire testing code and generating output file.

The variation of accuracy with respect to number of nearest neighbors (k) is as follows:

| SNo | Value of K | Accuracy       | 
|-----|------------|----------------|
| 1   | 2          | 67.23          | 
| 2   | 4          | 69.67          | 
| 3   | 6          | 69.77          | 
| 4   | 8          | 69.35          | 
| 5   | 10         | 70.41          |

## Instruction to use:
For training: ./code.py train train_file.txt model_file.txt [model]

For testing:  ./code.py test test_file.txt model_file.txt [model]

model can be any one of 'nearest' or 'tree'
