import pandas as pd 
from pandas import ExcelFile 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def FileAnalys(items):
	read_file = pd.read_excel("gesampel.xls",sheet_name="Orders")
	clean = read_file.loc[read_file[items]>= 0.1] 
	select = clean.filter(items=["Discount","Sales"])
	item = select.head(20)

	x = item.iloc[:,:-1].values #independent/predict or discount
	y = item.iloc[:,1].values #dependent/response or sales
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
	regress =  LinearRegression()
	regress.fit(x_train,y_train) #fitting has been ready
	y_predic = regress.predict(x_test)
	y_predic #your data have two variables, that is y_test and y_predict
	#y_test is data fact from observation from sales, and y_pred is data prediction from model sales thinked ML

	#visualization data real
	plt.scatter(select["Discount"], select['Sales'])
	plt.xlabel("Discount")
	plt.ylabel("Harga Penjualan")
	plt.title("Grafik Penjualan Pabrik")
	plt.show()

	#visualization data predict
	plt.figure(figsize=(10,8))
	plt.scatter(x_train,y_train,color="blue")
	plt.plot(x_train, regress.predict(x_train), color="red")
	plt.title("Biaya Promosi terhadap Penjualan(training set)")
	plt.xlabel("Discount")
	plt.ylabel("Harga Penjualan")
	plt.show()

	#visualization data predict from test data
	plt.figure(figsize=(10,8))
	plt.scatter(x_test,y_test, color="green")
	plt.plot(x_train,regress.predict(x_train),color="gold")
	plt.title("Biaya terhadap promo (training set)")
	plt.xlabel("Discount")
	plt.ylabel("Harga")
	plt.show()
print(FileAnalys("Discount"))
