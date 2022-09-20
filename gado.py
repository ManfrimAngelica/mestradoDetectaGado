# ⍺⍺⍺ ---- Rastreamento de Peixes ---- ⍺⍺⍺

# Importa os módulos necessários para o funcionamento do programa
from collections import deque # Módulo de estrutura de dados do tipo lista de indexação rápida
from imutils.video import VideoStream # Módulo para obtenção de vídeo em tempo real
from imutils.video import FPS # Módulo para obtenção dos frames por segundo de execução do vídeo
from object_detection.utils import visualization_utils as vis_util # Módulo de visualização da API de detecção de objetos do TensorFlow
from object_detection.utils import label_map_util # Módulo de manipulação de labels da API de detecção de objetos do TensorFlow
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np # Módulo de operações matemáticas
import pandas as pd # Módulo de manipulação e análise de dados
import dlib # Módulo contendo diversas bibliotecas para cálculos númericos, processamento de imagens e afins
import cv2 # Módulo de ferramentas para visão computacional (OpenCV)
import tensorflow as tf # Módulo de ferramentas de redes neurais (TensorFlow)
import imutils # Módulo de ferramentas para manipulação de imagem/vídeo
import time # Módulo de ferramentas de tempo
import sys # Módulo de ferramentas do sistema
import os # Módulo de ferramentas de interface do sistema
import subprocess
import signal

# Variável de inicialização do modelo do TensorFlow, utilizado para representar um fluxo de dados como um gráfico
modelo = tf.Graph()

# Menu de opções
print('\n---- Contador de Gado ----\n') # Introdução ao programa

print('                                    ..')
print('                                     l  ,')
print('                                     .dk00.')
print('                          .,col;    ;xOxldXo')
print('     .;ldkxdlc;;".  ....,cdolOXNXkokdOxxd00Xo')
print('   :OK0O0NWWWNNWNNNNNNNNK00kddxkxdcdxxxxxx0KXl')
print(' .k0OOOkkOKNNXNNNNNWNXXNN000Oxxxddcxxxxxxddox,"')
print(' dxxxkkkkxkO00KKXXXNNNXXXO00Okxxxxdlxxdxo,')
print(' loxxxxkxxxkkO00KKK0KK00kkkOkkxxxxx:ol.')
print(' ,dxxxddxxxxxkkOO000K0OkkkkOOOkkkxd;"')
print(' .dxkkxxocodxxxxxkkOOOOkkkkkOOOOkxo')
print(' .oxkkd:;,,,coddxxxxxkkkxkkO0OkkxO"')
print(' .xxxxccc;     .";;,,,;;:dkOOxdd;.')
print(' oOkdooo:                .kOOxO.')
print(' ,ox."dd                  kOKKo')
print('. dx  cd                  :kOx.')
print('  lk  .d:                 "0kl')
print('  "d:  ;d;                "K0o')
print('   .,;  ..;.              .OKO"')
print('                           "dd,')

# Opções referentes à seleção da fonte de vídeo
print('\nDeseja-se usar uma fonte de vídeo a partir de:') # Escolha da fonte de vídeo
print('1 - Arquivo') # Fonte de vídeo a partir de um arquivo salvo/gravado
print('2 - Tempo real') # Fonte de vídeo a partir de uma Webcam/USB/CSI
opcaofonte = int(input('>')) # Armazena de fonte de vídeo a opção escolhida

# Tratamento da fonte de vídeo escolhida
if opcaofonte == 1: # Opção de vídeo a partir de um arquivo salvo/gravado
	print('\nDigite o endereço completo do arquivo de vídeo, incluindo seu nome e extensão:') # Mensagem ao usuário solicitando o local do arquivo de vídeo
	localfonte = input('>') # Recebe do usuário o local completo de onde está armazenado o arquivo de vídeo
	vs = cv2.VideoCapture(localfonte) # Armazena o arquivo de vídeo em uma variável
	fpscerto = vs.get(cv2.CAP_PROP_FPS) # Obtêm o valor de FPS do vídeo armazenado, para a geração correta do arquivo de vídeo com as informações de rastreamento
	fpscerto = round(fpscerto, 2) # Realiza o arredondamento do FPS obtido em 2 casas após a vírgula

	# Mensagem solicitando ao usuário o local que será armazenado a captura do vídeo exibido na tela em tempo real
	print('\nDigite o nome (sem a extensão) e o local do arquivo que deseja armazenar o vídeo com os pontos de rastreamento:')
	saidastream = input('>') # Recebe do usuário o local completo de onde será armazenado a captura do vídeo

elif opcaofonte == 2: # Opção de vídeo a partir de uma fonte Webcam/USB/CSI
	print('\nFontes de vídeo disponíveis para a captura:\n') # Mensagem ao usuário listando as fontes de vídeo presentes no dispositivo
	os.system("ls -l /dev/video*") # Lista as fontes de vídeo conectadas e disponíveis
	print('\nCaso encontre problemas com o dispositivo escolhido, talvez seja necessário utilizar ou aplicar outros parâmetros.') # Mensagem informativa sobre a necessidade de parâmtros adicionais
	print('\nDigite qual dispositivo será usado para a captura:') # Mensagem ao usuário solicitando a escolha do dispositivo de vídeo
	dispositivocap = input('>') # Armazena o dispositivo de vídeo escolhido pelo usuário em uma variável
	vs = cv2.VideoCapture(dispositivocap) # Armazena o vídeo em tempo real em uma variável
	print('\nFoi utilizado parâmetros adicionais na escolha do dispositivo? [S/N]') # Mensagem ao usuário questionando se houve a necessidade de inserção de parâmetros extras para a fonte de vídeo
	parametrocam = input('>') # Recebe a resposta do usuário sobre a fonte de vídeo

	# Se não foi necessário parâmetros adicionais, é possível informar mais parâmetros
	if parametrocam == "n" or parametrocam == "N":
		# Mensagem solicitando ao usuário a escolha e definição da resolução de captura e gravação do vídeo em tempo real
		print('\nÉ necessário definir a resolução (em pixels) para a captura. Digite o valor para o comprimento (width):')
		wcaptura = int(input('>')) # Armazena o valor da resolução de comprimento (width) em pixels fornecido pelo usuário
		vs.set(3, wcaptura) # Atribui na fonte de vídeo o valor digitado de resolução (em pixels) de comprimento (width)
		print('\nDigite o valor para a altura (height):')  # Mensagem solicitando ao usuário a escolha e definição da resolução (em pixels) de comprimento (height)
		hcaptura = int(input('>')) # Armazena o valor da resolução (em pixels) de altura (height) fornecido pelo usuário
		vs.set(4, hcaptura) # Atribui na fonte de vídeo o valor digitado de resolução (em pixels) de altura (height)

	# Mensagem solicitando ao usuário o local que será armazenado a captura do vídeo exibido na tela em tempo real
	print('\nDigite o nome (sem a extensão) e o local do arquivo que deseja armazenar o vídeo em tempo real original e com os pontos de rastreamento:')
	saidastream = input('>') # Recebe do usuário o local completo de onde será armazenado a captura do vídeo

else: # Opção digitada de maneira incorreta
	print('Opção inválida!') # Mensagem ao usuário dizendo que a opção digitada foi incorreta
	sys.exit() # Encerra o programa

# Opções referentes ao nome e local do arquivo escolhido para salvar os pontos de rastreamento
# Mensagem ao usuário solicitando o nome e o local do arquivo .csv a ser salvo
print('\nDigite o nome (sem a extensão) e o local do arquivo para salvar os dados de pontos de rastreamento:')
logarquivo = input('>') # Armazena o nome e o local que será salvo os dados de pontos de rastreamento

# Opções referentes à rede neural
print('\nA estrutura das redes do TensorFlow devem ser à seguinte:') # Mensagem demonstrando a estrutura exata que a pasta com a rede neural deve conter
print('|--- RedeGado') # Pasta que contêm os arquivos da rede neural
print('|	|--- classes.pbtxt') # Arquivo de listagem e nomes das classes
print('|	|--- frozen_inference_graph.pb') # Arquivo de pesos

# Seleção da pasta de rede neural escolhida
print('\nIndique a pasta onde está localizado os arquivos da rede neural escolhida:') # Mensagem solicitando ao usuário indicar a pasta da rede neural escolhida
pastarede = input('>') # Armazena o local da rede neural escolhida

# Tratamento de localização dos arquivos necessários referentes à rede neural
# Verifica se o local da pasta informado contêm os arquivos da rede neural
if not (os.path.isfile(pastarede + "/classes.pbtxt") and os.path.isfile(pastarede + "/frozen_inference_graph.pb")):
	# Mensagem informando ao usuário que os arquivos da rede neural não foram encontrados ou não estão no formato estruturado informado
	print('\nArquivos da rede neural não encontrados! Por favor, verifique se o local informado da pasta e a estrutura estão corretos.\n')
	sys.exit() # Encerra o programa
print('\nArquivos da rede neural encontrados!\n') # Mensagem informando ao usuário que os arquivos da rede neural foram localizados

# Carrega os arquivos de labels, pesos e configurações da rede 
rotulos = os.path.sep.join([pastarede, "classes.pbtxt"]) # Carrega o arquivo de nomes/classes
numeroclasses = 89 # Número de classes para o modelo
pesos = os.path.sep.join([pastarede, "frozen_inference_graph.pb"]) # Carrega o arquivo de topologia e pesos

# Torna o modelo como principal/padrão para a execução
with modelo.as_default():
	# Inicializa o gráfico de definições, utilizado para serializar o gráfico computacional do TensorFlow
	defgrafico = tf.GraphDef()

	# Carregamento do arquivo de topologia e pesos
	with tf.gfile.GFile(pesos, "rb") as f:
		graficoserial = f.read() # Realiza a leitura do arquivo de topologia e pesos
		defgrafico.ParseFromString(graficoserial) # Serializa o gráfico computacional
		tf.import_graph_def(defgrafico, name="") # Realiza a importação do gráfico computacional serializado

# Carrega o arquivo de classes e rótulos a partir do disco
maparotulos = label_map_util.load_labelmap(rotulos) # Carrega o arquivo de classes e rótulos
categorias = label_map_util.convert_label_map_to_categories(maparotulos, numeroclasses, use_display_name=True) # Converte o mapa de rótulos e classes, retornando uma lista de dicionários
categoriaid = label_map_util.create_category_index(categorias) # Cria um dicionário codificado pelo ID da categoria

# Opções referentes à seleção da fonte de vídeo
print('Qual execução deseja realizar?') # Escolha da fonte de vídeo
print('1 - Detector + Rastreador (mais veloz, acompanha o objeto)') # Fonte de vídeo a partir de um arquivo salvo/gravado
print('2 - Somente detector (mais lento, pode perder o objeto)') # Fonte de vídeo a partir de uma Webcam/USB/CSI
opcaoexecucao = int(input('>')) # Armazena de fonte de vídeo a opção escolhida

# Espaço em branco para separar os logs
print('')

# Inicializa a lista de pontos de rastreamento
listapontos = deque()

# Inicializa o parâmetro de confiança, visando filtrar detecções fracas (abaixo de 50%)
filtroconfianca = 0.3

# Inicializa o parâmetro de limite (threshold) 
valorlimite = 0.3

# Inicializa uma variável auxiliar que permite a gravação do vídeo
gravarvideo = None
gravarvideorastreio = None

# Inicializa as variáveis de contagem de FPS
aquecimento = 0 # Inicializa uma variável auxiliar de sincronismo
contagemfps = 0 # Inicializa uma variável auxiliar para a contagem de FPS
totalFPS = 0 # Inicializa a variável para a contagem total de FPS
totalframes = 0 # Inicializa a variável para a quantidade de frames lidos e processados

# Inicializações do rastreamento
coordenadasobjetos = OrderedDict() # Inicializa e declara um dicionário de objetos
coordenadasdesaparecidos = OrderedDict() # Inicializa e declara um dicionário de objetos desaparecidos
objetos = OrderedDict() # Inicializa e declara um dicionário de objetos
desaparecidos = OrderedDict() # Inicializa e declara um dicionário de objetos desaparecidos
valordesaparece = 40 # Inicializa o parâmetro que define em quantos frames o objeto será considerado perdido
valordistancia = 100 # Inicializa o parâmetro que define a distância entre objetos
saltoframe = 10 # Inicializa o parâmetro que define de quantos em quantos frames será realizada a detecção
IDobjetoaux = 0 # Inicializa a variável de vinculação da ID do objeto
IDobjetocoordaux = 0 # Inicializa uma variável auxiliar
porcentagens = [] # Inicializa a variável de armazenamento das porcentagens
rastreadores = [] # Inicializa a variável de armazenamento dos rastreadores
caixadetectada = [] # Inicializa a variável de armazenamento das caixas com objetos detectados
flagrastreador = 0 # Flag auxiliar para o rastreador
rotulosrastreados = [] # Inicializa a variável de armazenamento dos rótulos
objetosrastreados = {} # Inicializa a variável de objetos rastreáveis
coordenadasrastreadas = {} # Inicializa a variável de objetos rastreáveis

# Cria a sessão do TensorFlow para realizar o processo de inferência
with modelo.as_default():
	with tf.Session(graph=modelo) as sessao:
		# Inicia o loop
		while True:
			# Inicia a contagem dos frames por segundo utilizando o estimador da biblioteca imutils
			contaframe = FPS().start()

			# Realiza a leitura do frame atual e do próximo frame, para que possa ser executado tanto para vídeo gravado quanto em tempo real
			frame = vs.read()  # Captura o frame atual do vídeo
			frame = frame[1]  # Captura o próximo frame e o armazena

			# Tratamento para verificação de final do vídeo
			if frame is None:
				break

			# Obtêm as dimensões do frame lido
			(altura, comprimento) = frame.shape[:2]

			if opcaofonte == 2:
				# Copia o frame de entrada afim de manter o frame de captura original, sem informações de rastreamento, utilizado apenas para gravações em tempo real
				frameoriginal = frame.copy()

			# Converte o frame atual para o espaço de cores RGB, para que o rastreador do dlib possa operar corretamente
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Inicializa a lista de armazenamento das caixas delimitadoras advindas do detector ou do rastreador
			retangulos = []

			# Tratamento para aplicação da detecção ou rastreamento dependendo da quantidade de frames percorridos
			# Utiliza-se o detector, para obter os objetos e suas localizações
			if (totalframes % saltoframe == 0) or opcaoexecucao == 2:

				# Inicializa a variável de armazenamento dos rastreadores
				rastreadores = []

				imagemtensor = modelo.get_tensor_by_name("image_tensor:0")  # Obtêm a referência da imagem (frame) de entrada
				caixastensor = modelo.get_tensor_by_name("detection_boxes:0")  # Obtêm informações das caixas delimitadoras
				porcentagenstensor = modelo.get_tensor_by_name("detection_scores:0")  # Obtêm informações de pontuações (porcentagens) da detecção
				classestensor = modelo.get_tensor_by_name("detection_classes:0")  # Obtêm informações das classes
				numerodeteccoes = modelo.get_tensor_by_name("num_detections:0")  # Obtêm o número de detecções

				# Manipulações da imagem (frame) de entrada
				frameaux = np.expand_dims(rgb, axis=0)  # Expande as dimensões do vetor da imagem

				# Realiza a inferência/detecção, computando as caixas delimiitadoras, probabilidades e os rótulos das classes
				(caixas, pontuacoes, labels, N) = sessao.run([caixastensor, porcentagenstensor, classestensor, numerodeteccoes], feed_dict={imagemtensor: frameaux})

				# Reduz as listas em uma única dimensão
				caixas = np.squeeze(caixas)  # Reduz a dimensão da lista de caixas delimitadoras
				pontuacoes = np.squeeze(pontuacoes)  # Reduz a dimensão da lista das pontuações (porcentagens)
				labels = np.squeeze(labels)  # Reduz a dimensão da lista de rótulos (labels)

				# Faz um laço de repetição (loop) sobre as predições das caixas delimitadoras obtidas
				for (caixa, pontuacao, label) in zip(caixas, pontuacoes, labels):
					# Se a pontuação (probabilidade) for menor que a confiança mínima estipulada, ignorá-la
					if pontuacao < filtroconfianca:
						continue

					# Desenha a predição, porcentagem e a caixa delimitadora na imagem de saída
					label = categoriaid[label]  # Armazena a identificação e o nome da classe
					identificacao = int(label["id"])  # Armazena apenas o número de identificação da classe

					# Redimensiona a caixa delimitadora para a faixa entre 0 à 1 para o comprimento (width) e altura (height) e calcula os pontos centrais
					(comecoY, comecoX, fimY, fimX) = caixa  # Extrai as coordenadas da caixa delimitadora

					if (opcaoexecucao == 1):
						comecoX = int(comecoX * comprimento)  # Obtêm a coordenada inicial (em pixels) em X
						comecoY = int(comecoY * altura)  # Obtêm a coordenada inicial (em pixels) em Y
						fimX = int(fimX * comprimento)  # Obtêm a coordenada final (em pixels) em X
						fimY = int(fimY * altura)  # Obtêm a coordenada final (em pixels) em Y
						fimX = int(fimX - comecoX)  # Correção da coordenada fimX
						fimY = int(fimY - comecoY)  # Correção da coordenada fimY
						# Operações do rastreador
						rastreador = dlib.correlation_tracker()  # Inicializa o rastreador de correlação do dlib
						retangulo = dlib.rectangle(comecoX, comecoY, comecoX + fimX, comecoY + fimY)  # Constrói o objeto retângular do dlib a partir das coordenadas da caixa delimitadora
						rastreador.start_track(rgb, retangulo)  # Inicia o processo de rastreamento
						rastreadores.append(rastreador)  # Armazena os rastreadores atuais na lista de rastreadores, para que possa ser utilizado durante o pulo de frame
						porcentagens.append(pontuacao)  # Armazena as porcentagens atribuídas aos objetos detectados
						rotulosrastreados.append(label["name"])  # Armazena os rótulos atribuídos aos objetos detectados
					else:
						comecoX = int(comecoX * comprimento)  # Obtêm a coordenada inicial (em pixels) em X
						comecoY = int(comecoY * altura)  # Obtêm a coordenada inicial (em pixels) em Y
						fimX = int(fimX * comprimento)  # Obtêm a coordenada final (em pixels) em X
						fimY = int(fimY * altura)  # Obtêm a coordenada final (em pixels) em Y
						retangulos.append((comecoX, comecoY, fimX, fimY))  # Constrói o objeto retângular do dlib a partir das coordenadas da caixa delimitadora
						porcentagens.append(pontuacao)  # Armazena as porcentagens atribuídas aos objetos detectados
						rotulosrastreados.append(label["name"])  # Armazena os rótulos atribuídos aos objetos detectados
						rotuloimprime = "{}: {:.2f}".format(label["name"], pontuacao)  # Armazena a classe referente aos objetos identificados e suas probabilidades (confiança)
						cv2.rectangle(frame, (comecoX, comecoY), (fimX, fimY), (0, 0, 255), 2)  # Exibe os retângulos de determinada cor para a classe
						cv2.putText(frame, rotuloimprime, (comecoX, comecoY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)  # Exibe a classe e as porcentagens para o usuário

			# Utiliza-se o rastreador ao invés do detector, garantindo maior desempenho
			else:
				# Percorre a lista de rastreadores, rótulos e porcentagens
				for (rastreador, rotulo, porcentagem) in zip(rastreadores, rotulosrastreados, porcentagens):
					# Atualizações do rastreador
					rastreador.update(rgb)  # Atualiza o rastreador
					pos = rastreador.get_position()  # Obtêm a posição atualizada do objeto

					# Obtêm as coordenadas do objeto em processo de rastreamento
					comecoX = int(pos.left())  # Armazena a posição de início da caixa em X (Esquerda)
					comecoY = int(pos.top())  # Armazena a posição de início da caixa em Y (Topo)
					fimX = int(pos.right())  # Armazena a posição de fim da caixa em X (Direita)
					fimY = int(pos.bottom())  # Armazena a posição de fim da caixa em Y (Inferior)

					# Armazena as coordenadas obtidas anteriormente na lista de retângulos
					retangulos.append((comecoX, comecoY, fimX, fimY))

					# Exibição da caixa delimitadora, porcentagens e rótulo dos objetos rastreados
					rotuloimprime = "{}: {:.2f}".format(rotulo, porcentagem)  # Armazena a classe referente aos objetos identificados e suas probabilidades (confiança)
					cv2.rectangle(frame, (comecoX, comecoY), (fimX, fimY), (0, 0, 255), 2)  # Exibe os retângulos de determinada cor para a classe
					cv2.putText(frame, rotuloimprime, (comecoX, comecoY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)  # Exibe a classe e as porcentagens para o usuário

			# Tratamento de verificação da lista de retângulos de caixas delimitadoras
			# Verifica se a lista está vazia
			if len(retangulos) == 0:
				# Percorre sobre os objetos rastreados existentes, sinalizando-os como desaparecidos
				for IDobjeto in list(desaparecidos.keys()):
					desaparecidos[IDobjeto] += 1  # Realiza a contagem para o objeto perdido

					# Caso foi chegado à um número máximo de frames consecutivos onde um objeto foi sinalizado como perdido, apaga-se o objeto da lista
					if desaparecidos[IDobjeto] > valordesaparece:
						del objetos[IDobjeto]  # Apaga o objeto da lista
						del coordenadasobjetos[IDobjeto]  # Apaga o objeto da lista
						del desaparecidos[IDobjeto]  # Apaga a lista de desaparecidos

			# Para o caso de a lista não estar vazia
			else:
				# Inicializa uma lista de centroides para o frame atual
				centroideatual = np.zeros((len(retangulos), 2), dtype="int")
				coordenadaatual = np.zeros((len(retangulos), 4), dtype="int")

				# Percorre sobre os retângulos de caixas delimitadoras
				for (i, (comecoX, comecoY, fimX, fimY)) in enumerate(retangulos):
					# Obtêm os pontos da centroide a partir das coordenadas das caixas delimitadoras
					centroX = int((comecoX + fimX) / 2.0)  # Obtêm a coordenada central em X
					centroY = int((comecoY + fimY) / 2.0)  # Obtêm a coordenada central em Y
					centroideatual[i] = (centroX, centroY)  # Armazena as coordenadas das centroides em uma lista
					coordenadaatual[i] = (comecoX, comecoY, fimX, fimY)

				# Se não está sendo rastreado quaisquer objetos, utiliza-se a centroide de entrada e registra-se cada uma
				if len(objetos) == 0:
					for i in range(0, len(centroideatual)):
						centroide = centroideatual[i]  # Armazena a centroide calculada
						objetos[IDobjetoaux] = centroide  # Vincula a centroide com o objeto
						desaparecidos[IDobjetoaux] = 0  # Reseta o contador de desaparecimento
						IDobjetoaux += 1  # Acrescenta a variável auxiliar de vinculação de objetos

					for i in range(0, len(coordenadaatual)):
						coordenada = coordenadaatual[i]
						coordenadasobjetos[IDobjetocoordaux] = coordenada
						IDobjetocoordaux += 1

				# Caso contrário, estamos rastreando os objetos, assim, precisamos conferir as centroides de entrada com as centroides existentes
				else:
					# Obtêm a ID dos objetos e suas centroides
					IDsobjetos = list(objetos.keys())  # Obtêm a ID dos objetos
					centroidesobjetos = list(objetos.values())  # Obtêm as centroides dos objetos

					# Calcula a distância entre as centroides de entrada e as centroides atuais, tentando verificar se as mesmas estão próximas e se conferem aos devidos objetos
					distcentroide = dist.cdist(np.array(centroidesobjetos), centroideatual)

					# Encontra-se o menor valor da linha e organiza os índices baseados em seus valores mínimos, assim, o menor valor se encontrará de início no índice da lista
					linhas = distcentroide.min(axis=1).argsort()

					# Encontra-se o menor valor da coluna e organiza os índices baseados nos índices das linhas
					colunas = distcentroide.argmin(axis=1)[linhas]

					# Para verificar se um objeto precisa ser atualizado, registrado ou apagado, é necessário verificar quais colunas e linhas foram analisadas
					linhausada = set()  # Verifica as linhas
					colunausada = set()  # Verifica as colunas

					# Percorre sobre a combinação (linhas, colunas) de índices de tuplas
					for (linha, coluna) in zip(linhas, colunas):
						# Se as linhas ou colunas foram analisadas anteriormente, ignorá-las
						if linha in linhausada or coluna in colunausada:
							continue

						# Se a distância entre centroides é maior que a distância máxima, não associar as duas centroides com o mesmo objeto
						if distcentroide[linha, coluna] > valordistancia:
							continue

						# Caso contrário, obter a ID do objeto com sua linha associada, definindo sua nova centroide, e resetar o contador de desaparecimento
						IDobjeto = IDsobjetos[linha]  # Obtem a ID do objeto de acordo com sua linha
						objetos[IDobjeto] = centroideatual[coluna]  # Vincula a nova centroide
						coordenadasobjetos[IDobjeto] = coordenadaatual[coluna]
						desaparecidos[IDobjeto] = 0  # Reseta o contador de desaparecimento

						# Indica que cada índice de linha e coluna foi analisado
						linhausada.add(linha)  # Indica a linha
						colunausada.add(coluna)  # Indica a coluna

					# Calcula ambos os índices de linha e coluna que ainda não foram examidados
					linhasnulas = set(range(0, distcentroide.shape[0])).difference(linhausada)  # Calcula a linha
					colunasnulas = set(range(0, distcentroide.shape[1])).difference(colunausada)  # Calcula a coluna

					# Tratamento para o caso do número de centroides for maior ou igual ao número de centroides de entrada, verificando se os objetos podem ter desaparecido
					if distcentroide.shape[0] >= distcentroide.shape[1]:
						# Percorre sobre os índices de linhas que não foram utilizados
						for linha in linhasnulas:
							# Obtêm a ID do objeto e seu índice de linha correspondente e incrementa o contador de desaparecimento
							IDobjeto = IDsobjetos[linha]  # Obtêm a ID do objeto
							desaparecidos[IDobjeto] += 1  # Realiza a contagem de frames consecutivos relacionados ao desaparecimento

							# Verifica se o número de frames consecutivos para o objeto foi sinalizado como desaparecido e, por garantia, apaga o objeto
							if desaparecidos[IDobjeto] > valordesaparece:
								del objetos[IDobjeto]  # Deleta o objeto
								del coordenadasobjetos[IDobjeto]
								del desaparecidos[IDobjeto]  # Deleta da lista de desaparecido

					# Caso contrário, o número de centroides de entrada é maior que o número de centroides dos objetos já existentes, portanto, é preciso registrar as centroides de entrada como objeto rastrável
					else:
						# Percorre sobre os índices de colunas que não foram utilizados
						for coluna in colunasnulas:
							centroide = centroideatual[coluna]  # Armazena a centroide
							objetos[IDobjetoaux] = centroide  # Vincula a centroide com o objeto
							desaparecidos[IDobjetoaux] = 0  # Reseta o contador de desaparecimento
							IDobjetoaux += 1  # Acrescenta a variável auxiliar de vinculação de objetos

							coordenada = coordenadaatual[coluna]
							coordenadasobjetos[IDobjetocoordaux] = coordenada
							IDobjetocoordaux += 1

			# Percorre sobre os objetos rastreados e suas centroides
			for (IDobjeto, coordenada) in coordenadasobjetos.items():
				# Verifica se o objeto rastreável existe para a ID atual
				coord = coordenadasrastreadas.get(IDobjeto, None)

				# Caso não há um objeto rastreável, crie-o
				if coord is None:
					IDobjeto = IDobjeto  # Vincule com a ID do objeto
					coordenads = [coordenada]  # Passe as coordenadas da centroide
				# Caso exista, armazene sua centroide
				else:
					coordenads.append(coordenada)  # Armazena as coordenadas da centroide

				# Armazena o objeto rastreável no dicionário
				coordenadasrastreadas[IDobjeto] = coord

				# Armazena a ID do objeto e as coordenadas do ponto central
				listapontos.append((totalframes, (IDobjeto+1), coordenada[0], coordenada[1], (coordenada[2] - coordenada[0]), (coordenada[3] - coordenada[1]), -1, -1, -1, -1))

			# Percorre sobre os objetos rastreados e suas centroides
			for (IDobjeto, centroide) in objetos.items():
				# Verifica se o objeto rastreável existe para a ID atual
				to = objetosrastreados.get(IDobjeto, None)

				# Caso não há um objeto rastreável, crie-o
				if to is None:
					IDobjeto = IDobjeto  # Vincule com a ID do objeto
					centroids = [centroide]  # Passe as coordenadas da centroide
				# Caso exista, armazene sua centroide
				else:
					centroids.append(centroide)  # Armazena as coordenadas da centroide

				# Armazena o objeto rastreável no dicionário
				objetosrastreados[IDobjeto] = to

				# Verifica o tamanho do vetor e conta o total de objetos que foram detectados/rastreados
				totalrastreado = len(objetosrastreados.keys())

				# Verifica o tamanho do vetor e conta os objetos presentes na tela que estão sendo detectados/rastreados
				contagematual = len(objetos.items())

				# Manipulações de desenho da ID e o ponto central do objeto
				cv2.putText(frame, "Contador de Gado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2);
				cv2.putText(frame, "Total: " + format(totalrastreado) + ", Atual: " + format(contagematual), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2);
				txt = "ID {}".format(IDobjeto)  # Armazena a ID do objeto em formato de texto para imprimir na tela ao usuário
				cv2.putText(frame, txt, (centroide[0] - 10, centroide[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Escreve o número da ID do objeto
				cv2.circle(frame, (centroide[0], centroide[1]), 4, (0, 0, 255), -1)  # Desenha um pequeno círculo que indica o centro da caixa delimitadora

			# Tratamento para a geração do arquivo de vídeo contendo as informações de rastreamento a partir de um arquivo de vídeo previamente gravado
			if gravarvideorastreio is None and opcaofonte == 1:
				formatovideo = cv2.VideoWriter_fourcc(*'XVID')  # Define o codec para gravação/compactação do arquivo de vídeo em tempo real
				# Para testar outros codecs, consulte a página: http://www.fourcc.org/codecs.php
				gravarvideorastreio = cv2.VideoWriter(saidastream + "_rastreio.avi", formatovideo, fpscerto, (comprimento, altura), True)  # Habilita a gravação do arquivo de vídeo

			# Tratamento para a gravação do arquivo de vídeo em tempo real (vídeo RAW e informações de rastreamento)
			if gravarvideo is None and gravarvideorastreio is None and opcaofonte == 2 and aquecimento >= 11:  # Caso o arquivo não esteja presente, sem gravações, for escolhido a opção em tempo real e a aplicação estabilizada
				formatovideo = cv2.VideoWriter_fourcc(*'XVID')  # Define o codec para gravação/compactação do arquivo de vídeo em tempo real
				# Para testar outros codecs, consulte a página: http://www.fourcc.org/codecs.php
				gravarvideo = cv2.VideoWriter(saidastream + "_original.avi", formatovideo, fpscerto, (comprimento, altura), True)  # Habilita a gravação do arquivo de vídeo
				gravarvideorastreio = cv2.VideoWriter(saidastream + "_rastreio.avi", formatovideo, fpscerto, (comprimento, altura), True)  # Habilita a gravação do arquivo de vídeo

			# Verifica a quantidade de vezs que foi passado pelo loop para ser iniciado a gravação no arquivo com o FPS já estabelecido
			if opcaofonte == 1:  # Caso seja escolhido a opção em tempo real e a aplicação tenha estabilizado
				gravarvideorastreio.write(frame)  # Escreve o arquivo de vídeo com os dados presentes no frame atual (frame do vídeo em tempo real) sem os dados de rastreamento (caixas delimitadoras)

			# Verifica a quantidade de vezs que foi passado pelo loop para ser iniciado a gravação no arquivo com o FPS já estabelecido
			if opcaofonte == 2 and aquecimento >= 11:  # Caso seja escolhido a opção em tempo real e a aplicação tenha estabilizado
				gravarvideo.write(frameoriginal)  # Escreve o arquivo de vídeo com os dados presentes no frame atual (frame do vídeo em tempo real) com os dados de rastreamento (caixas delimitadoras)
				gravarvideorastreio.write(frame)  # Escreve o arquivo de vídeo com os dados presentes no frame atual (frame do vídeo em tempo real) sem os dados de rastreamento (caixas delimitadoras)

			# Exibe a fonte de vídeo selecionada e a janela de rastreamento ao usuário
			cv2.imshow("Contador de Gado", frame)  # Exibe a fonte de vídeo selecionada

			# Fica observando caso seja pressionado alguma tecla do teclado
			teclado = cv2.waitKey(30) & 0xFF

			# Se a tecla "q" for pressionada, o programa é encerrado
			if teclado == ord("q"):  # Caso a tecla "q" seja pressionada
				break  # Encerra o loop

			# Se a tecla "q" for pressionada, o programa é encerrado
			if teclado == ord("d"):  # Caso a tecla "q" seja pressionada
				manualdetecta = 1  # Encerra o loop

			# Manipulações de operações de frames
			contaframe.update()  # Atualiza a contagem de FPS
			totalframes += 1  # Realiza a contagem de frames processados
			contaframe.stop()  # Para a atualização de FPS
			fpsatual = format(contaframe.fps())  # Obtêm o valor de FPS para o frame processado
			totalFPS = totalFPS + float(fpsatual)  # Realiza a soma total de FPS

			if opcaofonte == 2:
				# Tratamento para sincronização e obtenção do frame correto de gravação do vídeo em tempo real
				if 0 <= aquecimento <= 10:
					aquecimento = aquecimento + 1;  # Contador auxiliar de estabilização da gravação do vídeo
				if 6 <= aquecimento <= 10:  # Passado 5 contagens, o processamento do sistema encontra-se estabilizado
					contagemfps = contagemfps + float(fpsatual)  # Armazena 5 valores considerados estáveis de FPS
				if aquecimento >= 11:  # Depois de 11 contagens, obtêm-se o valor correto de FPS
					fpscerto = contagemfps / 5  # Realiza a média de 5 frames estáveis para encontrar o FPS certo de gravação do arquivo de vídeo
					fpscerto = round(fpscerto, 2)  # Obtêm a taxa de frames correta para gravação do vídeo

# Exibe a média de frames de execução do sistema
totalFPS = totalFPS / totalframes  # Realiza a média aritmética de FPS de execução da detecção e rastramento
totalFPS = round(totalFPS, 2)  # Arredonda o valor de FPS, fixando num valor de apenas 2 casas após a vírgula
print('\nFPS médio: ', totalFPS)  # Mensagem informando ao usuário a média de FPS

# Grava os pontos de rastreamento para um arquivo
print('\nGravando o arquivo de pontos de rastreamento (.csv)...')  # Mensagem informando ao usuário que os pontos de rastreamento estão sendo gravados no arquivo .csv
pd.DataFrame(listapontos).to_csv(logarquivo + ".csv", index=False, header=['Frame', 'ID', 'X', 'Y', 'Comprimento', 'Altura', 'Dumb1', 'Dumb2', 'Dumb3', 'Dumb4'])  # Grava os pontos (coordenadas) de rastreamento no arquivo .csv definido pelo usuário
print('Pontos gravados!')  # Mensagem informando ao usuário que os pontos de rastreamento foram gravados no arquivo .csv
print('Encerrando a aplicação...')  # Mensagem informando ao usuário que a aplicação está em processo de encerramento

# Caso a fonte de vídeo seja em arquivo, para a leitura
if opcaofonte == 1:
	print('Fechando o arquivo de vídeo...')  # Mensagem informando ao usuário que o arquivo de vídeo está sendo fechado
	gravarvideorastreio.release()  # Fecha a conexão de gravação do arquivo de vídeo
	vs.release()  # Fecha a conexão com o arquivo de vídeo aberto

# Caso a fonte de vídeo seja em tempo real, encerra a transmissão e a gravação
else:
	print('Fechando a comunicação em tempo real...')  # Mensagem informando ao usuário que a fonte de vídeo em tempo real está sendo fechada
	gravarvideo.release()  # Fecha a conexão de gravação do arquivo de vídeo
	gravarvideorastreio.release()  # Fecha a conexão de gravação do arquivo de vídeo
	vs.release()  # Fecha a conexão com a fonte de vídeo em tempo real, liberando-a

# Fecha as janelas de vídeo e rastreamento
print('Programa finalizado!\n')  # Mensagem ao usuários informando que o programa foi encerrado
cv2.destroyAllWindows()  # Fecha todas as abertas janelas do OpenCV
