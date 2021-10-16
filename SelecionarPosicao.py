import cv2
import random

escala = 0.5
circulos = []
contador = 0
contador2 = 0
ponto1 = []
ponto2 = []
meusPontos = []
cor = []

def pontosDoMouse(btnPressionadoOuNao, x, y, flags, params):
    global contador, ponto1, ponto2, contador2, circulos, cor

    ## indica que o botão do mouse esquerdo está pressionado.
    if btnPressionadoOuNao == cv2.EVENT_LBUTTONDOWN:
        if contador==0:
            ponto1= int(x // escala), int(y // escala);
            contador += 1
            cor = (random.randint(0, 2) * 200, random.randint(0, 2) * 200, random.randint(0, 2) * 200)
        elif contador==1:
            ponto2= int(x // escala), int(y // escala)
            type = input('Informe o tipo de dado: ')
            name = input('Informa o nome: ')
            meusPontos.append([ponto1, ponto2, type, name])
            contador=0
        circulos.append([x, y, cor])
        contador2 += 1

imgOndeCirculoDesenhado = cv2.imread('cut_2_sem_qr_code.png')

# (0, 0) = tamanho desejado para a imagem de saída
# escala =	fator de escala ao longo do eixo horizontal e vertical
imgOndeCirculoDesenhado = cv2.resize(imgOndeCirculoDesenhado, (0, 0), None, escala, escala)

while True:
    # Exibir pontos
    for x, y, cores in circulos:
        # (x, y) = centro do circulo / 3 = raio do circulo
        # cv2.FILLED = Espessura do contorno do círculo, tipo de linha CHEIA
        cv2.circle(imgOndeCirculoDesenhado, (x, y), 3, cores, cv2.FILLED)
    cv2.imshow("Imagem original ", imgOndeCirculoDesenhado)
    cv2.setMouseCallback("Imagem original ", pontosDoMouse)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(meusPontos)
        break