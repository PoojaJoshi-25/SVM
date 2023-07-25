from sklearn.datasets import load_breast_cancer    #sklearn library of ml --->>open source data analysis library
import matplotlib.pyplot as plt       # graphical plotting library
from sklearn.inspection import DecisionBoundaryDisplay  #sklearn.inspection--> design a better model
from sklearn.svm import SVC  # svm for clasisfication of tasks

#Load the dataset
cancer = load_breast_cancer()
X = cancer.data[:, :2]  # :-> select all rows, :2 selects the first two columns
y = cancer.target  # cancer.target would contain the corresponding class labels (0s and 1s)

#Build the model
svm = SVC(kernel="rbf", gamma='scale', C=1.0)
#svm = SVC(kernel='linear')
#svm = SVC(kernel ='poly', degree = 4)

#Training the model
svm.fit(X, y)

#Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
		svm,
		X,
		response_method="predict",
		cmap=plt.cm.Spectral,
		alpha=0.8,
		xlabel=cancer.feature_names[0],
		ylabel=cancer.feature_names[1],
	)

#Scatter plot
plt.scatter(X[:, 0], X[:, 1],
			c=y,
			s=20, edgecolors="k")
plt.show()


#   plt.scatter: This function is used to create a scatter plot, where each point is represented by a dot on the graph.
#
# X[:, 0] and X[:, 1]: These are NumPy array indexing expressions. X is assumed to be a two-dimensional array (or a matrix) representing the input features (input data) for the scatter plot. X[:, 0] selects all rows from the first column of X, and X[:, 1] selects all rows from the second column of X. This means that the scatter plot will use the first column of X as the x-coordinates and the second column of X as the y-coordinates for each data point.
#
# c=y: The c parameter specifies the color of each point in the scatter plot. It sets the color of the points based on the y variable, which contains the target values or class labels. Each unique value in y will be assigned a different color, allowing you to visualize the different classes or groups in the scatter plot.
#
# s=20: The s parameter sets the size of the markers (dots) in the scatter plot. In this case, the size of each dot will be 20.
#
# edgecolors="k": The edgecolors parameter specifies the color of the edges of the markers. Here, "k" represents the color black, so the edges of the dots will be black.