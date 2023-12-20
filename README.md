# Processamento de Imagem
Código para processamento de imagens em python desenvolvido na disciplina de computação gráfica utilizando a linguagem de programação Python e a biblioteca Pillow (PIL).

## Tipo de Processamento disponíveis no Código

1. Imagem Negativa --> Inverte as cores da imagem.
2. Imagem Monocromática --> Converte a imagem para tons de cinza.
3. Binarização --> Converte a imagem para preto e branco com base em um limiar.
4. Limiarização --> Aplica limiarização à imagem com limiares definidos pelo usuário.
5. Fatiamento --> Mantém uma faixa específica de intensidades de pixel, criando um efeito fatiado.
6. Contraste Incrementado --> Melhora o contraste da imagem dentro de uma faixa de intensidade definida pelo usuário.
7. Histograma --> Exibe o histograma da imagem.
8. Equalização de Histograma --> Melhora o contraste da imagem equalizando o histograma.
9. Ruído Gaussiano --> Adiciona ruído gaussiano aleatório à imagem.
10. Ruído Sal e Pimenta --> Introduz ruído sal e pimenta à imagem.
11. Filtragem Passa-baixa (Média) --> Aplica um filtro passa-baixa à imagem usando a média.
12. Filtragem Passa-baixa (Mediana) --> Aplica um filtro passa-baixa à imagem usando a mediana.
13. Filtragem Passa-alta --> Aplica um filtro passa-alta à imagem.
14. Filtragem de Linha (Horizontal) --> Realça as bordas horizontais na imagem.
15. Filtragem de Linha (Vertical) --> Realça as bordas verticais na imagem.
16. Filtragem de Linha (+45 graus) --> Realça as bordas em um ângulo de +45 graus.
17. Filtragem de Linha (-45 graus) --> Realça as bordas em um ângulo de -45 graus.
18. Operador de Roberts --> Aplica o operador de Roberts para detecção de bordas.
19. Operador de Sobel --> Aplica o operador de Sobel para detecção de bordas.
20. Operador de Prewitt --> Aplica o operador de Prewitt para detecção de bordas.
21. Operador de Frei-Chen --> Aplica o operador de Frei-Chen para detecção de bordas.
    
## 🤔 Como Funciona o Código
Primeiramente, é carregada uma imagem de escolha do usuário na linha 423:
```
img = Image.open('image/gray.jpg')
```
Caso queira inserir outra imagem de sua escolha, é só simplesmente mudar o diretório da imagem

Ao executar o código, irão ser mostradas as opções no terminal de qual tipo de processamento o usuário deseja reaizar na imagem.

![image](https://github.com/rodineves/Processamento-de-Imagem/assets/105732866/cc9ad21c-5864-4f4e-b04b-bb02ddd13d4c)

Ao escolher a opção digitando o número rspectivo e clicando Enter, o processamento ocorre e a imagem pós-processada é salva na mesma pasta da imagem original. Porém, se for de sua escolha pode mudar os diretórios:
```
inverted_img.save('image/gatenho_invertido.jpg')
```
Acima é o exemplo de salvar a imagem negativa na pasta <i>image</i>, mas pode mudar em todas as opções de processamento.
