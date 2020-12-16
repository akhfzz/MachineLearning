import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn
from sklearn import datasets, neighbors
from sklearn.neighbors import KNeighborsClassifier
from openpyxl import Workbook, load_workbook
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

class SupervisedLearning:
	def __init__(self):
		self.number = None
		self.names = None
		self.file = None

	def Explanation(self,*names):
		self.names=names
		#create and load file
		create_file_excel = Workbook()
		create_file_excel.save("Data_iris.xlsx")

		file = datasets.load_iris()

		#Explanatory analys
		print(type(file))
		print(file.keys())
		print(type(file.data), type(file.target))
		print(file.data.shape)
		print(file.target_names)
		print("_--^^"*15)

		#Convert datasets to dataframe
		columns = ["Panjang sepal", "lebar sepal", "panjang petal", "lebar petal", "class"]
		df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=columns)

		df.to_excel("Data_iris.xlsx")

		X_train = file["data"]
		y_train = file["target"]
		call_knn = KNeighborsClassifier(n_neighbors=3,weights="uniform", algorithm="auto",metric="euclidean")
		call_knn.fit(X_train,y_train)
		data_pred = [list(self.names)] #data will be prediction euclidian
		y_pred = call_knn.predict(data_pred)
		input ("0:setosa, 1:versicolor, 2:virginica")
		print("Result from : the character no-{}".format(y_pred))
		return self.names

	def ValueError(self,file):
		self.file = file

		# name_columns = ["Panjang sepal", "lebar sepal", "panjang petal", "lebar petal", "target"]
		# df = pd.read_csv(self.file, names=name_columns)
		df = pd.read_excel("Data_iris.xlsx")
		X = df.iloc[:, 1:4].values
		Y = df.iloc[:, 5].values
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=42)

		encoding = LabelEncoder()
		encoding.fit(y_train)
		y_train = encoding.transform(y_train)
		y_test = encoding.transform(y_test)

		standarisasi = StandardScaler()
		standarisasi.fit(X_train) 
		X_train = standarisasi.transform(X_train)
		X_test = standarisasi.transform(X_test)

		error = []
		for i in range(1, 40):
			algorithm = KNeighborsClassifier(n_neighbors=i)
			algorithm.fit(X_train, y_train)
		 
			pred_range = algorithm.predict(X_test)
			error.append(np.mean(pred_range != y_test))
		plt.figure()
		plt.plot(range(1, 40), error, color='grey', marker='o', 
		        markerfacecolor='gold', markersize=10)
		plt.title('Average Error with K Variable')
		plt.xlabel('Value K')
		plt.ylabel('Average Error')
		plt.show()

		classifier = KNeighborsClassifier(n_neighbors=4)
		classifier.fit(X_train, y_train)
	
		y_pred = classifier.predict(X_test)
		print(classification_report(y_test, y_pred, target_names= encoding.classes_))
		return self.file

	def VisualizationClassific(self,number):
		self.number = number
		file = datasets.load_iris()

		X = file.data[:,:self.number]
		Y = file.target
		light_clmap = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
		bold_clmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
		nearest = neighbors.KNeighborsClassifier(n_neighbors=6, weights="uniform")
		nearest.fit(X,Y)
		x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
		y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
		double_x, double_y = np.meshgrid(np.arange(x_min, x_max),
										np.arange(y_min, y_max))
		dist = nearest.predict(np.c_[double_x.ravel(), double_y.ravel()])
		dist = dist.reshape(double_x.shape)
		plt.figure()
		plt.pcolormesh(double_x, double_y, dist, cmap=light_clmap)
		plt.scatter(X[:,0], X[:,1], c=Y, cmap=bold_clmap, edgecolor="k", s=20)
		plt.xlim(double_x.min(), double_x.max())
		plt.ylim(double_y.min(), double_y.max())
		plt.title("Irish Flower Classification")
		plt.show()
		return self.number

panjang_sepal = input("berapa panjang sepal : ")
lebar_sepal = input("berapa lebar sepal : ")
panjang_petal = input("berapa panjang petal : ")
lebar_petal = input("berapa lebar petal : ")
# predict(panjang_petal, lebar_sepal, panjang_petal, lebar_petal)
# ValueError("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
# VisualizationClassific(2)
obj = SupervisedLearning()
obj.ValueError("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
print(obj.file)
obj.Explanation(panjang_petal, lebar_sepal, panjang_petal, lebar_petal)
print(obj.names)