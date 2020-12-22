import pandas as pd 
from pandas import ExcelFile
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def Firstly(file):
	#Explanation data
	defile = pd.read_excel(file)
	defile.isnull().sum()
	defile.set_index("Panjang sepal")
	defile.keys()

	#convert and showing graphyic
	tampung = defile["class"].values.tolist()
	conv = set(tampung)

	a = defile.iloc[:,5].values

	x_gr = defile["Panjang sepal"]
	y_gr = defile["lebar sepal"]
	plt.scatter(x_gr, y_gr)
	plt.xlabel("Panjang Sepal")
	plt.ylabel("Lebar Sepal")
	plt.title("Grafik dari Klaster Bunga Iris")
	plt.show()
Firstly("Data_iris.xlsx")

def activationDBscan(komponen):
	df = pd.read_excel("Data_iris.xlsx")
	data = df.iloc[:,1:5]

	algorithm = DBSCAN()
	
	algorithm.fit(data)

	dekomposit = PCA(n_components=komponen).fit(data)

	pca = dekomposit.transform(data)
	label = {"Irish-Setosa": "red", "Irish-Versicolor": "gold", "Irish-Virgnica":"green"}

	for ryz in range(0,pca.shape[0]):
		if algorithm.labels_[ryz] == 0:
			Cls1 = plt.scatter(pca[ryz,0], pca[ryz,1], c="r", marker="^")

		elif algorithm.labels_[ryz] == 1:
			Cls2 = plt.scatter(pca[ryz,0], pca[ryz,1],c="g", marker="o")

		elif algorithm.labels_[ryz] == -1:
			Cls3 = plt.scatter(pca[ryz,0], pca[ryz,1], c="b", marker="X")

	plt.legend([Cls1, Cls2, Cls3],["Cluster1", "Cluster2", "Cluster3"])
	plt.title("Kluster Bunga Iris dengan DBscan")
	plt.show()
int_input = int(input("How many component which u want no more from 4: "))
activationDBscan(int_input)


