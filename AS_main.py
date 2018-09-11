'''
TÍTULO: Método Computacional de Análise de Sentimento por meio deAprendizado de Máquina
AUTOR: Edilton Torres de Andrade
LICENÇA DO SOFTWARE: GNU General Public License (GPLv3)
Código fonte de Classificador Naive Bayes usando a Biblioteca TextBlob.

Parametros:
a) Treinamento (quantidade de resenhas)
b) Teste (posicação na lista, quantidade de resenhas
c) URL para teste, usar permalink do site https://www.imdb.com

	Exemplos:
	Filme - Transformers: The Last Knight - Estrelas 1/10
	https://www.imdb.com/review/rw3818482/?ref_=tt_urv

	Filme - Interstellar - Estrelas 10/10
	https://www.imdb.com/review/rw3117999/?ref_=tt_urv

	Filme - Mother - Estrelas 10/10
	https://www.imdb.com/review/rw4214905/?ref_=tt_urv

'''

#Bibliotecas
import webbrowser
import numpy as np
import re
import time
import progressbar
import pandas as pd
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from urllib.request import urlopen
from bs4 import BeautifulSoup

#função para limpar texto do html
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

#remover ligações
def remove_stopwords(sentence, stopwords):
	sentencewords = sentence.split()
	resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
	result = ' '.join(resultwords)
	return result

# 
if __name__ == "__main__":

	start = time.time() #captura hora de inicio
	print('\n')
	print('Método Computacional de Análise de Sentimento por meio de Aprendizado de Máquina')
	print('Edilton Torres de Andrade - GNU General Public License (GPLv3)')
	print('Classificador Naive Bayes usando a Biblioteca TextBlob','\n')
	print ("****** Iniciando Classificação Naive Bayes ******")

	bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength) #inicia barra de progresso

	i = int(time.time()-start)
	bar.update(i)
	
	#Recuperando os dados de treinamento

	name="imdb_tr.csv" #nome arquivo base de dados
	
	data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')

	i = int(time.time()-start)
	bar.update(i)

	qtd_treinamento=100 # qtd de dados para treinamento. Valor será o dobro.

	datatop=data.head(qtd_treinamento) #exemplos do inicio do arquivo
	databottom=data.tail(qtd_treinamento) #exemplos do fim do arquivo

	datamerge1=datatop.append(databottom) #unifica dados do inicio e do fim. Terá qtd_treinamento x2
	
	datamerge1['polarity2'] = np.where(datamerge1['polarity']==1,'pos',np.where(datamerge1['polarity']==0,'neg','n/a')) #converte 1 para "pos" e 0 para "neg"

	train2=list(zip(*[datamerge1[c].values.tolist() for c in ['text', 'polarity2']])) # cria tupla 

	data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')

	qtd_teste=20 #qtd de dados de teste. Valor será o dobro.

	inicio_teste=200 #registro de inicio da base de teste

	datatop=data.head(inicio_teste).tail(qtd_teste)

	databottom=data.tail(inicio_teste).head(qtd_teste)

	datamerge2=datatop.append(databottom) #unifica dados do inicio e do fim. Terá qtd_teste x2

	datamerge2['polarity2'] = np.where(datamerge2['polarity']==1,'pos',np.where(datamerge2['polarity']==0,'neg','n/a'))

	test2=list(zip(*[datamerge2[c].values.tolist() for c in ['text', 'polarity2']]))

	#atualiza barra de progresso
	i = int(time.time()-start)
	bar.update(i)

	#inicia classificador treinamento
	cl = NaiveBayesClassifier(train2)

	#atualiza barra de progresso	
	i = int(time.time()-start)
	bar.update(i)
	
	#registra acuracia treinamento x teste
	actest = cl.accuracy(test2) 

	#atualiza barra de progresso		
	i = int(time.time()-start)
	bar.update(i)

	#treina com a base de teste
	cl.update(test2)

	#atualiza barra de progresso	
	i = int(time.time()-start)
	bar.update(i)
	
	#registra acuracia teste vs treinamento + este
	actest2 = cl.accuracy(test2) 

	print( '\n')
	print ("Descrição dos Dados de Treinamento:")
	print(datamerge1.describe())
	print( '\n')
	print ("Descrição dos Dados de Teste:")
	print(datamerge2.describe())
	print( '\n')
	print ("Acurácia (base de teste versus vs base de treinamento): ", actest )
	print( '\n')
	print("Acurácia revisada (base de teste vs treinamento + teste): ", actest2)
	print( '\n')
	print("Top 10 - Características mais relevantes:")
	cl.show_informative_features(10)
	print('\n')
	print("****** Tempo total de execução", int(time.time()-start), "segundos ******")
	print( '\n')
	print("****** Teste customizado ******")
	print('\n')
	print("Insira um Permalink (URL) de uma resenha em inglês do site (https://www.imdb.com)")
	
	urldefault="https://www.imdb.com/review/rw3117999/?ref_=tt_urv"
	print('Default = Interestellar - Estrelas 10/10 -', urldefault)

	userurl=input("<Enter> Default ou entre com o Permalink(URL):")
	
	if userurl == '':
		userurl = urldefault

	webbrowser.open(userurl) #abre browser com review

	response = urlopen(userurl)
	
	html = response.read()

	soup = BeautifulSoup(html, features="html5lib")
	
	review = str(soup.find_all("div", "text show-more__control"))

	stopwords = open("stopwords.en.txt", 'r' , encoding="ISO-8859-1").read()
	stopwords = stopwords.split("\n")

	review = remove_stopwords(review, stopwords)

	review=cleanhtml(review)[1 :-1]

	print ('\n')
	print ('Texto da resenha:')

	blob = TextBlob (review, classifier=cl)

	classificacao = blob.classify()

	print(blob)
	print("Classificação:", classificacao)
	print('\n')
	useravaliacao =classificacao

	useravaliacao = input("<Enter> para manter ou digite a sua classificação (pos/neg):")

	new_data = [(review, classificacao)]

	if useravaliacao == "pos":
		new_data = [(review, 'pos')]

	elif useravaliacao == "neg":
		new_data = [(review, 'neg')]


	cl.update(new_data)

	print(blob)
	print("Classificação Revisada:", blob.classify())