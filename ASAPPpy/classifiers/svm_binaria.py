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

	df = pd.read_csv("divididobinario.txt",sep='§',header=0)
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
	#print(g)

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

	with open("vect_bin", 'wb') as fid:
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


def corre_para_frase_1(modelo,frase):
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("classifiers/vectors/vect_bin", 'rb') as fid:
		v = pickle.load(fid)
	sentences = [frase]
	print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)


	############################################
	#Parte para alimentar o outro classificador
	############################################
	in_domain = []
	for index,value in enumerate(preds):
		print(value)
		if(value==1):
			#frase = str(sentences[index])+"§"+str(classe_original[index])
			in_domain.append(frase)
	return in_domain

def corre_para_frase(frase):
	with open("classifiers/trained_models/svm_binaria_v3.pickle", 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("classifiers/vectors/vect_bin", 'rb') as fid:
		v = pickle.load(fid)
	sentences = [frase]
	print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)

	'''
	############################################
	#Parte para alimentar o outro classificador
	############################################
	in_domain = []
	for index,value in enumerate(preds):
		print(value)
		if(value==1):
			#frase = str(sentences[index])+"§"+str(classe_original[index])
			in_domain.append(frase)
	return in_domain
	'''
	return preds[0]

def corre_para_testes(modelo,ficheiro):
	print("##############")
	nome = ficheiro.replace("out_","")
	nome = nome.replace(".txt","")
	print(nome.upper())
	print("##############")
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("vect_bin", 'rb') as fid:
		v = pickle.load(fid)
	sentences = []
	true_results = []
	classe_original = []
	with open(ficheiro,"r") as f:
		for line in f:
			line = line.replace("\n","")
			line = line.split("§")
			sentences.append(line[0])
			classe_original.append(int(line[1]))
			if(int(line[1])<15):
				true_results.append(1)
			else:
				true_results.append(15)
	print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)

	score = classification_report(true_results, preds,labels=[1,15])
	print(score)


	############################################
	#Parte para alimentar o outro classificador
	############################################
	in_domain = []
	for index,value in enumerate(preds):
		if(value==1):
			frase = str(sentences[index])+"§"+str(classe_original[index])
			in_domain.append(frase)
	return in_domain

def corre(modelo,v):
	labels = ["In the domain","In the domain","Formas jurídicas","Cartão de Empresa/Cartão de Pessoa Coletiva","Criação da Empresa Online","Certificados de Admissibilidade","Inscrições Online","Certidões Online","Gestão da Empresa Online","RJACSR","Alojamento Local","Out of Domain","Out of Domain","Out of Domain","Out of Domain","Out of Domain","Out of Domain"]
	
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("vect_bin", 'rb') as fid:
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
		print(preds)
		if(preds[0]==1):
			print("ID")
		else:
			print("OOD")



if __name__ == '__main__':
	#_lr_0.03
	#modelo = "lstm_com_balanceamento_varias_camadas_500_lr_0.03.h5"
	#modelo = "lstm_com_balanceamento_varias_camadas_200_lr_0.03.h5"
	#vect = treina("modelos/svm_binaria.pickle")
	#corre("modelos/svm_binaria.pickle","")
	#corre_para_testes("modelos/svm_binaria.pickle","inputs/out_vg1_legendas_original.txt")
	print(corre_para_frase("modelos/svm_binaria.pickle","frase de teste"))
	print(corre_para_frase("modelos/svm_binaria.pickle","akhjsd jkasndkjasnjdknaskj nda"))




