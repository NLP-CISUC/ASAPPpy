import keras as k
import pandas as pd
import cufflinks as cf
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import  Embedding
from keras.layers import Dense,SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.recurrent import LSTM
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.models import Model, load_model
from sklearn.utils import resample
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
#naive bayes, svm, random forest



#https://github.com/susanli2016/NLP-with-Python/blob/master/Multi-Class%20Text%20Classification%20LSTM%20Consumer%20complaints.ipynb

#1---Empresa na Hora
#2---Marca na hora e marca na hora online
#3---Formas jurídicas
#4---Cartão de Empresa/Cartão de Pessoa Coletiva
#5---Criação da Empresa Online
#6---Certificados de Admissibilidade
#7---Inscrições Online
#8---Certidões Online
#9---Gestão da Empresa Online
#10---RJACSR
#11---Alojamento Local

def escala(X,y):
	from imblearn.over_sampling import RandomOverSampler
	sampler = RandomOverSampler(sampling_strategy='not majority',random_state=0)
	X_train, Y_train = sampler.fit_sample(X, y)
	return X_train,Y_train

def treina(model_name):

	df = pd.read_csv("dataset/divididossopcomlegendas.txt",sep='§',header=0)
	df.info()

	max_len = 0
	for value in df.Perguntas:
		if(len(value)>max_len):
			#print(value)
			max_len = len(value)
	max_words = 0
	for value in df.Perguntas:
		word_count = len(value.split(" "))
		if(word_count>max_words):
			#print(word_count)
			max_words = word_count
	#print("---------")
	#print(max_words)
	#print(df.Class.value_counts().sort_values())
	g = df.groupby('Class')
	g= pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
	print(df)

	#df =  pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
	#print("###")
	#print(df.shape)
	#print(df.head(10))
	#df = g



	#X_conveted = pd.get_dummies(df["Perguntas"])
	#Divisão em treino e teste
	X_train, X_test, Y_train, Y_test = train_test_split(df["Perguntas"],df["Class"], test_size = 0.3, random_state = 42)
	#print(X_train.shape)
	#print(Y_train.shape)
	#print(X_test.shape)
	#print(Y_test.shape)



	vect = TfidfVectorizer().fit(X_train)

	with open("vect", 'wb') as fid:
		pickle.dump(vect, fid)	

	X_train_vectorized = vect.transform(X_train)

	X_train_vectorized, Y_train = escala(X_train_vectorized, Y_train)

	clf = svm.SVC(gamma="scale",kernel='linear', degree=16)
	clf.fit(X_train_vectorized,Y_train)

	with open(model_name, 'wb') as fid:
		pickle.dump(clf, fid)	

	preds = clf.predict(vect.transform(X_test))
	score = classification_report(Y_test, preds)
	print(score)

	return vect


def corre(modelo,v):
	labels = ["Empresa na Hora", "Marca na hora e marca na hora online", "Formas jurídicas", "Cartão de Empresa/Cartão de Pessoa Coletiva", "Criação da Empresa Online","Certificados de Admissibilidade", "Inscrições Online","Certidões Online", "Gestão da Empresa Online","RJACSR","Alojamento Local", "Fora do Domínio", "Fora do Domínio", "Fora do Domínio", "Fora do Domínio", "Fora do Domínio"]

	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("vect", 'rb') as fid:
		v = pickle.load(fid)
	a = 0

	while (a==0):
		print("Entrada.")
		entrada = input()
		entrada = [entrada]
		#entrada = pd.DataFrame([entrada])
		print("Entrada recebida.")
		transformada = v.transform(entrada)
		preds = clfrNB.predict(transformada)
		print(preds[0])
		print(labels[preds[0]])

def svm_classifier(frase):

	labels = ["Empresa na Hora", 1, "Formas jurídicas", "Cartão de Empresa/Cartão de Pessoa Coletiva", "Criação da Empresa Online", "Certificados de Admissibilidade", "Inscrições Online", "Certidões Online", "Gestão da Empresa Online", 2, 3, "Fora do Domínio", "Fora do Domínio", "Fora do Domínio", "Fora do Domínio", 4]

	with open("trained_models/svm_linear_kernel.pickle", 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("vectors/vect", 'rb') as fid:
		v = pickle.load(fid)

	entrada = [frase]
	transformada = v.transform(entrada)
	preds = clfrNB.predict(transformada)

	return labels[preds[0]]


if __name__ == '__main__':
	#_lr_0.03
	#modelo = "lstm_com_balanceamento_varias_camadas_500_lr_0.03.h5"
	#modelo = "lstm_com_balanceamento_varias_camadas_200_lr_0.03.h5"
	# vect = treina("svm_linear_kernel.pickle")
	# corre("svm_linear_kernel.pickle", "")
	svm_classifier("ola")




