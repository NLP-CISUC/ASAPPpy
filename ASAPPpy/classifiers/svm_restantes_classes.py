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
from ASAPPpy.classifiers.svm_binaria import *
#from svm_binaria import *
import matplotlib.pyplot as plt

#from sklearn.svm._classes import *
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

	df = pd.read_csv("input/multiclass_only_train.txt",sep='§',header=0)
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

	with open("Models/vect_tresclasses", 'wb') as fid:
		pickle.dump(vect, fid)	

	X_train_vectorized = vect.transform(X_train)

	X_train_vectorized, Y_train = escala(X_train_vectorized, Y_train)

	clf = svm.SVC(gamma="scale",kernel='linear', degree=16,probability=True)
	clf.fit(X_train_vectorized,Y_train)

	with open(model_name, 'wb') as fid:
		pickle.dump(clf, fid)	

	preds = clf.predict(vect.transform(X_test))
	score = classification_report(Y_test, preds,target_names = ["Espaço Empresa","Apoios Sociais","RJACSR","Alojamento Local"])
	print(score)

	return vect

def corre_para_testes_restantes(modelo,frases,tresh_hold):
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("Models/vect_tresclasses", 'rb') as fid:
		v = pickle.load(fid)

	sentences = []
	true_results = []
	for line in frases:
		line = line.replace("\n","")
		line = line.split("§")
		sentences.append(line[0])
		true_results.append(int(line[1]))
	print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)
	preds_probs = clfrNB.predict_proba(transformada)

	#score = classification_report(true_results, preds,labels=[1,10,11])
	#print(score)
	'''
	all_classes = []
	all_valores = []
	max_prob_correto = []
	max_prob_errado = []
	for index, elem in enumerate(sentences):
		current_prob = preds_probs[index]
		if(preds[index]==true_results[index]):
			max_prob_correto.append(current_prob[current_prob.argmax()])
		else:
			max_prob_errado.append(current_prob[current_prob.argmax()])
		classes = []
		valores = []
		for index_2,elem in enumerate(current_prob):
			if(elem>tresh_hold):
				classes.append(index_2)
				valores.append(elem)
		frase = sentences[index] + "Prediction: "
		for elem in classes:
			frase+=" "+str(elem)+" "
		frase+=" probability " 
		for elem in valores:
			frase+=" "+str(elem)
		all_classes.append(classes)
		all_valores.append(valores)
		#print(frase)
	'''
	score = classification_report(true_results, preds)
	#target_names = ["Espaço Empresa","Apoios Sociais","RJACSR","Alojamento Local"]
	print(score)
	#hist = np.histogram([max_prob], bins=100)
	#print(max_prob)
	#plt.hist([max_prob_correto,max_prob_errado], bins=100)
	#plt.show()
	return preds

def corre_para_frase_restantes(modelo,frases,tresh_hold):
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("Models/vect", 'rb') as fid:
		v = pickle.load(fid)

	sentences = []
	true_results = []
	sentences.append(frases[0].replace("\n",""))
	print("Entrada recebida.")
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)
	preds_probs = clfrNB.predict_proba(transformada)

	max_prob_correto = []
	max_prob_errado = []
	for index, elem in enumerate(sentences):
		current_prob = preds_probs[index]
		classes = []
		valores = []
		for index_2,elem in enumerate(current_prob):
			if(elem>tresh_hold):
				classes.append(index_2)
				valores.append(elem)
	return classes,valores

def corre(modelo,v):
	#RJACSR,AL,PE
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("Models/vect_tresclasses", 'rb') as fid:
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
		preds_probs = clfrNB.predict_proba(transformada)
		print(preds)
		if(preds[0]==1):
			print("PE")
		elif(preds[0]==10):
			print("RJACSR")
		elif(preds[0]==11):
			print("AL")
		else:
			print("OOD")

def corre_modelo_real(modelo,frases,tresh_hold):
	
	with open(modelo, 'rb') as fid:
		clfrNB = pickle.load(fid)

	with open("Models/vect_tresclasses", 'rb') as fid:
		v = pickle.load(fid)

	sentences = frases
	transformada = v.transform(sentences)
	preds = clfrNB.predict(transformada)
	preds_probs = clfrNB.predict_proba(transformada)
	max_prob_correto = []
	max_prob_errado = []
	for index, elem in enumerate(sentences):
		current_prob = preds_probs[index]
		classes = []
		valores = []
		for index_2,elem in enumerate(current_prob):
			#print(elem)
			if(elem>tresh_hold):
				classes.append(index_2)
				valores.append(elem)
	return classes,valores
	'''
	for index,elem in enumerate(preds_probs):
		#print(sentences[index] + " "+ str(elem))
		print(elem)
	return preds_probs
	'''

def corre_modelo_completo(modelo_binario,modelo_conjunto,frase,tresh_hold):
	frases_validas = corre_para_frase(modelo_binario,frase)
	return corre_modelo_real(modelo_conjunto,frases_validas,tresh_hold)

if __name__ == '__main__':
	##########
	#Descomentar para treinar modelo multiclasses (1-4)
	##########
	#vect = treina("Models/svm_muticlass.pickle")

	#########
	#Coisas para testes
	#########
	'''
	frases = []
	print("##################VERSAO VG1#################")
	with open("Input/TestMulticlass/VG1.txt","r") as f:
		for line in f:
			frases.append(line)
			#print(line)
	corre_para_testes_restantes("Models/svm_muticlass.pickle",frases,0.8)

	frases = []
	print("##################VERSAO VG2#################")
	with open("Input/TestMulticlass/VG2.txt","r") as f:
		for line in f:
			frases.append(line)
			#print(line)
	corre_para_testes_restantes("Models/svm_muticlass.pickle",frases,0.8)

	frases = []
	print("##################VERSAO VUC#################")
	with open("Input/TestMulticlass/VUC.txt","r") as f:
		for line in f:
			frases.append(line)
			#print(line)
	corre_para_testes_restantes("Models/svm_muticlass.pickle",frases,0.8)

	frases = []
	print("##################VERSAO VIN#################")
	with open("Input/TestMulticlass/VIN.txt","r") as f:
		for line in f:
			frases.append(line)
			#print(line)
	corre_para_testes_restantes("Models/svm_muticlass.pickle",frases,0.8)
	'''
	'''
	frases = []
	print("##################VERSAO VIN#################")
	with open("Input/TestMulticlass/total.txt","r") as f:
		for line in f:
			frases.append(line)
			#print(line)
	corre_para_testes_restantes("Models/svm_muticlass.pickle",frases,0.8)
	'''

	#########
	#Correr binário seguido de normal
	#########
	tresh_hold = 0.9
	for i in range(10000):
		for elem in (corre_modelo_completo("Models/svm_binaria_v3.pickle","Models/svm_muticlass.pickle","frase de teste",tresh_hold)):
			print(elem)



	