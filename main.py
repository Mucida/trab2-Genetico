import copy 
import gym
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


qtdDeGeracoes = 100 #quantas vezes irá rodar o genetico
tamanhoPopulacao = 500 #quantidade de indiviudos por geracao
numeroDePassos = 500 #numero de passos total por individuo
topMelhores = 10 #elite a ser mantida na geracao
chanceDeMutacao = 0.4
tamanhoDoBloco = 15 #numero maximo de passos(actions) diferentes por individuo
minTamanhoDoBloco = 5
tamanhoDoBlocoInicial = 8 #numero maximo de passos(actions) iniciais diferentes por individuo
minTamanhoDoBlocoInicial = 2
totalPlotar = [] #usado para plotar o grafico com os melhores de cada geracao
totalMelhorPlotar = []
notas = [] #usado para criar a probabilidade de escolher um pai
voltas = 1 #quantas vezes o mesmo individuo ira simular (valor medio sera calculado)


class Passo:
	def __init__(self, acao, valor):
		self.acao = acao
		self.valor = valor


class Individuo: 
	def __init__(self, passosIniciais, passosFinais, total):
		self.passosFinais = passosFinais
		self.passosIniciais = passosIniciais
		self.total = total


def set_chanceDeMutacao(val):
    global chanceDeMutacao  
    chanceDeMutacao = val

#gera geracao 0 com individuos totalmente aleatorios
def geraPrimeiraGeracao():
	print('Gerando primeira geracao')
	populacao = []
	for k in range(tamanhoPopulacao):
		#gera bloco inicial
		passosIniciais = []
		tamInicial = random.randint(minTamanhoDoBlocoInicial, tamanhoDoBlocoInicial-1)
		for i in range(tamInicial):
			acao = []
			for x in range(4):
				acao.append(random.uniform(-1,1))
			passosIniciais.append(Passo(acao, 0))
		#gera bloco final
		passosFinais = []
		tamFinal = random.randint(minTamanhoDoBloco, tamanhoDoBloco-1)
		for i in range(tamFinal):
			acao = []
			for x in range(4):
				acao.append(random.uniform(-1,1))
			passosFinais.append(Passo(acao, 0))

		robo = Individuo(passosIniciais, passosFinais, 0)
		total = 0
		aux = 0
		for i in range(numeroDePassos):
			if i < len(passosIniciais):
				obs, reward, done, _ = envFake.step(robo.passosIniciais[i].acao)
			else:
				j = (aux+len(passosFinais))%len(passosFinais)
				obs, reward, done, _ = envFake.step(robo.passosFinais[j].acao)
				if done:
					break
			aux += 1
			total+=reward
		robo.total = total
		envFake.reset()
		populacao.append(copy.deepcopy(robo))
	return populacao


#mutacao de troca de passos
def mutacao1(individuo):
	posicao1 = random.randint(0, tamanhoDoBloco-1)
	posicao2 = random.randint(0, tamanhoDoBloco-1)
	aux = individuo.passos[posicao1]
	individuo.passos[posicao1] = individuo.passos[posicao2]
	individuo.passos[posicao2] = aux


#mutacao de troca de gene
def mutacao2(individuo):
	posicaoPasso = random.randint(0, len(individuo.passosIniciais)-1)
	posicaoGene = random.randint(0,3)
	individuo.passosIniciais[posicaoPasso].acao[posicaoGene] = random.uniform(-1,1)
	posicaoPasso = random.randint(0, len(individuo.passosFinais)-1)
	posicaoGene = random.randint(0,3)
	individuo.passosFinais[posicaoPasso].acao[posicaoGene] = random.uniform(-1,1)


#ussado para fazer a probabilidade dos melhores serem escolhidos
def somaPrefixo(melhoreIndividuos):
	menor = melhoreIndividuos[topMelhores-1].total
	if menor < 0:
		menor *= -1
		for i in range(topMelhores):
			notas[i] += menor
	for i in range(topMelhores-1):
		notas[i+1] += notas[i]


#avalias os steps de um individuo no mapa
def avalia(individuo):	
	total = 0
	for v in range(voltas):
		aux = 0
		for i in range(numeroDePassos):
			if i < len(individuo.passosIniciais):
				obs, reward, done, _ = envFake.step(individuo.passosIniciais[i].acao)
			else:
				j = (aux+len(individuo.passosFinais))%len(individuo.passosFinais)
				obs, reward, done, _ = envFake.step(individuo.passosFinais[j].acao)
				if done:
					break
			aux+=1
			total+=reward
		envFake.reset()
	individuo.total = total/voltas

def auxCalculaPosicoes(notas):
	indicePai1 = random.uniform(0,notas[topMelhores-1])
	for m in range(topMelhores):
		if indicePai1 <= notas[m]:
			indicePai1 = m
			break
	indicePai2 = indicePai1
	while indicePai2 == indicePai1:
		indicePai2 = random.uniform(0,notas[topMelhores-1])
		for m in range(topMelhores):
			if indicePai2 <= notas[m]:
				indicePai2 = m
				break
	return indicePai1, indicePai2

#realiza o crossover
def cruzamento():
	print("cruzamento")
	melhoreIndividuos = [] 
	notas.clear()
	#os P melhores filhos da geracao anterior
	for i in range(topMelhores):
		melhoreIndividuos.append(copy.deepcopy(populacao[i]))
		notas.append(copy.deepcopy(populacao[i].total))
	somaPrefixo(melhoreIndividuos)
	populacao.clear()
	#coloca os melhores individuos na nova geracao
	for i in range(topMelhores):
		populacao.append(copy.deepcopy(melhoreIndividuos[i]))

	#gera os filhos atraves de cruzamentos
	for i in range(tamanhoPopulacao-topMelhores):
		filho = Individuo([],[], 0)
		indicePai1, indicePai2 = auxCalculaPosicoes(notas)
		ranInicial=0
		#realiza o cruzamento da parte final
		sorteia = random.randint(1,2)
		if(sorteia == 1):
			ranInicial = len(populacao[indicePai1].passosIniciais)
		else:
			ranInicial = len(populacao[indicePai2].passosIniciais)
		#ranInicial = min(len(populacao[indicePai1].passosIniciais), len(populacao[indicePai2].passosIniciais))
		for m in range(ranInicial):
			passosIniciais = Passo([],0)
			acao = []
			corte = random.randint(1, 2)
			for j in range(4):
				if j < corte:
					if m < len(populacao[indicePai1].passosIniciais):
						acao.append(copy.deepcopy(populacao[indicePai1].passosIniciais[m].acao[j]))
					else: 
						acao.append(copy.deepcopy(populacao[indicePai2].passosIniciais[m].acao[j]))
				else:
					if m < len(populacao[indicePai2].passosIniciais):
						acao.append(copy.deepcopy(populacao[indicePai2].passosIniciais[m].acao[j]))
					else:
						acao.append(copy.deepcopy(populacao[indicePai1].passosIniciais[m].acao[j]))
			for n in range(4):
				passosIniciais.acao.append(acao[n])
			filho.passosIniciais.append(copy.deepcopy(passosIniciais))
		
		#realiza o cruzamento da parte final
		indicePai1, indicePai2 = auxCalculaPosicoes(notas)
		ranFinal=0
		sorteia = random.randint(1,2)
		if(sorteia == 1):
			ranFinal = len(populacao[indicePai1].passosFinais)
		else:
			ranFinal = len(populacao[indicePai2].passosFinais)
		ranFinal = min(len(populacao[indicePai1].passosFinais), len(populacao[indicePai2].passosFinais))
		for m in range(ranFinal):
			passosFinais = Passo([],0)
			acao = []
			corte = random.randint(1, 2)
			for j in range(4):
				if j < corte:
					if m < len(populacao[indicePai1].passosFinais):
						acao.append(copy.deepcopy(populacao[indicePai1].passosFinais[m].acao[j]))
					else: 
						acao.append(copy.deepcopy(populacao[indicePai2].passosFinais[m].acao[j]))
				else:
					if m < len(populacao[indicePai2].passosFinais):
						acao.append(copy.deepcopy(populacao[indicePai2].passosFinais[m].acao[j]))
					else:
						acao.append(copy.deepcopy(populacao[indicePai1].passosFinais[m].acao[j]))
			for n in range(4):
				passosFinais.acao.append(acao[n])
			filho.passosFinais.append(copy.deepcopy(passosFinais))

		#tenta realizar mutacoes
		fator = random.random()
		if fator < chanceDeMutacao:
			sorteia = random.randint(1,10)
			if sorteia < 0:
				mutacao1(filho)
			else:
				mutacao2(filho)	
		avalia(filho)	
		populacao.append(copy.deepcopy(filho))

	#muta elite
	for i in range(topMelhores):
		fator = random.random()
		if fator < chanceDeMutacao*0.8:
			mutacao2(populacao[i])
			avalia(populacao[i])
	

#ordena a populacao em ordem decrescente
def ordenacao(populacao, melhorAtual): 
	print('ordenacao')
	populacao.sort(key=lambda x: x.total, reverse=True)
	cont = 0
	for p in populacao:
		if cont == 0:
			totalPlotar.append(p.total)
			if melhorAtual.total < p.total:
				melhorAtual = copy.deepcopy(p)
			totalMelhorPlotar.append(melhorAtual.total)
		if cont < 5: #mostrar os 5 melhores (para controle)
			print(p.total)
		cont += 1
	return melhorAtual

#prepara um ambiente pra teste
envFake = gym.make('BipedalWalker-v2')
observationFake = envFake.reset()
populacao = geraPrimeiraGeracao()
melhorAtual = Individuo([],[],0)
melhorAtual = ordenacao(populacao, melhorAtual)
#roda o genetico
for i in range(qtdDeGeracoes):
	print('')
	print('GERACAO ', i+1)
	cruzamento()
	melhorAtual = ordenacao(populacao, melhorAtual)
	#set_chanceDeMutacao(chanceDeMutacao - (i+10)/100)
envFake.close()

plt.plot(range(qtdDeGeracoes+1), totalPlotar, label='Melhor por geração')
plt.plot(range(qtdDeGeracoes+1), totalMelhorPlotar, label='Melhor de todos')
plt.ylabel('Reward')
plt.xlabel('Gerações')
plt.legend()
plt.savefig('file')

#apresenta o melhor individuo da ultima geracao na tela
def andaFinal(env, individuo):
	total = 0
	aux = 0
	print(individuo.total)
	for i in range(500):
		env.render()
		if i < len(individuo.passosIniciais):
			obs, reward, done, _ = env.step(individuo.passosIniciais[i].acao)
		else:
			j = (aux+len(individuo.passosFinais))%len(individuo.passosFinais)
			obs, reward, done, _ = env.step(individuo.passosFinais[j].acao)
		aux+=1
	env.reset()

#prepara o ambiente para o melhor robo tentar andar
env = gym.make('BipedalWalker-v2')
observation = env.reset()
input('Aperte para rodar o melhor de todos:') 
while True:
	andaFinal(env, melhorAtual)
	x = input('Andar novamente? (s/n):')
	if x == 'n':
		break
env.close()
