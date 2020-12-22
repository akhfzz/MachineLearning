import pandas as pd 
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import seaborn as sns

def Cluster(k):
	#Explanatory,wrangling etc
	roar = pd.read_excel("gesampel.xls")
	roar.corr()
	select = roar.filter(items=["Sales","Profit"])
	select.head(12)
	df = select.isnull().sum()

	# Convert to array from datasets and represent
	toarray = np.asarray(select)
	plt.scatter(toarray[:,0], toarray[:,1], label="True Position")
	plt.xlabel("Sales")
	plt.ylabel("Profit")
	plt.title("Rekayasa Pendapatan")
	plt.show()

	#Algorithm active
	algorith = KMeans(n_clusters=k)
	algorith.fit(toarray)
	algorith.cluster_centers_

	#After algorith was active, u can showing visual data
	plt.scatter(toarray[:,0],toarray[:,1],c=algorith.labels_,
				cmap="rainbow")
	plt.scatter(algorith.cluster_centers_[:,0],
		algorith.cluster_centers_[:,1], color="black")
	plt.xlabel("Sales")
	plt.ylabel("Profit")
	plt.title("Cluster Pendapatan")
	plt.show()
K_input = int(input("how many K for centroid :"))
Cluster(K_input)