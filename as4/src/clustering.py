import numpy as np
from random import randint


class KMeans():
    """
    KMeans. Class for building an unsupervised clustering model
    """

    def __init__(self, k, max_iter=20):

        """
        :param k: the number of clusters
        :param max_iter: maximum number of iterations
        """

        self.k = k
        self.max_iter = max_iter

    def init_center(self, x):
        """
        initializes the center of the clusters using the given input
        :param x: input of shape (n, m)
        :return: updates the self.centers
        """

        self.centers = np.zeros((self.k, x.shape[1]))

        ################################
        #      YOUR CODE GOES HERE     #
        ################################
        for i in range(self.k): #Loop through the number of clusters
            temp = randint(0, x.shape[0])   #Create a random number (index) from 0 to 7352   
            self.centers[i] = x[temp]       #The ith center becomes the "temp" example
            #self.centers will now contain k number of 'rows' from X.
            
    def revise_centers(self, x, labels):
        """
        it updates the centers based on the labels
        :param x: the input data of (n, m)
        :param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
        :return: updates the self.centers
        """

        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)
            self.centers[i, :] = x[wherei, :].mean(0)

    def predict(self, x):
        """
        returns the labels of the input x based on the current self.centers
        :param x: input of (n, m)
        :return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
        """
        labels = np.zeros((x.shape[0]), dtype=int) #this contains 7352 0's - one for each example.
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        lowest = np.inf
        for i in range(x.shape[0]):
            for j in range(self.k):
                 e = np.linalg.norm(self.centers[j] - x[i])
                 if e < lowest:
                    lowest = e
                    labels[i] = j
            lowest = np.inf
        return labels

    def get_sse(self, x, labels):
        """
        for a given input x and its cluster labels, it computes the sse with respect to self.centers
        :param x:  input of (n, m)
        :param labels: label of (n,)
        :return: float scalar of sse
        """
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        sse = 0.
        for i in range(x.shape[0]):
            error_i = np.linalg.norm(self.centers[labels[i]] - x[i])
            sse += error_i
        return sse

    def get_purity(self, x, y):
        """
        computes the purity of the labels (predictions) given on x by the model
        :param x: the input of (n, m)
        :param y: the ground truth class labels
        :return:
        """
        labels = self.predict(x)
        purity = 0
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        
        for i in range(self.k):
            points_in_cluster = []
            for j in range(len(labels)):
                if labels[j] == i:
                    points_in_cluster.append(y[j])
            temp = np.bincount(points_in_cluster)
            purity += max(temp)

        return (purity / x.shape[0])


    def fit(self, x):
        """
        this function iteratively fits data x into k-means model. The result of the iteration is the cluster centers.
        :param x: input data of (n, m)
        :return: computes self.centers. It also returns sse_veersus_iterations for x.
        """

        # intialize self.centers
        self.init_center(x)

        sse_vs_iter = []
        for iter in range(self.max_iter):
            # finds the cluster index for each x[i] based on the current centers
            labels = self.predict(x)

            # revises the values of self.centers based on the x and current labels
            self.revise_centers(x, labels)

            # computes the sse based on the current labels and centers.
            sse = self.get_sse(x, labels)

            sse_vs_iter.append(sse)

        return sse_vs_iter
