from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd 
from pandas import ExcelFile

class NaiveBayes:
	def __init__(self, *inputtest):
		self.inputtest = inputtest
	def GNB(self):
		#Explanatory
		file = pd.read_excel("Data_iris.xlsx", sheet_name="Sheet1")
		data = file.iloc[:,1:5].values
		target = file.iloc[:, 5].values
		# print(file.head(10))
		# print(target.shape)
		# print(set(target))

		#Select training and testing set
		x_train, x_test, y_train, y_test = train_test_split(data, target)
		call_NB = GaussianNB()
		call_NB.fit(data,target)

		pred_trng_set = call_NB.predict(data)

		data_test = [list(self.inputtest)]
		y_pred = call_NB.predict(data_test)
		return "Based on this description, what is kind of iris? : {}".format(y_pred)
		return "Accuracy = %0.2f" % accuracy_score(y_test, call_NB.predict(x_test))
		return classification_report(y_test,call_NB.predict(x_test))

panjang_sepal = float(input("berapa panjang sepal : "))
lebar_sepal = float(input("berapa lebar sepal : "))
panjang_petal = float(input("berapa panjang petal : "))
lebar_petal = float(input("berapa lebar petal : "))
result = NaiveBayes(panjang_sepal,lebar_sepal, panjang_petal, lebar_petal)
print(result.GNB())