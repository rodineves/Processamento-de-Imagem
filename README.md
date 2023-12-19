# Processamento-de-Imagem
Código para processamento de imagens em python desenvolvido na disciplina de computação gráfica utilizando a linguagem de programação Python e a biblioteca Pillow (PIL).

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
