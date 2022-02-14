Modelo para a escrita em LaTeX de teses na Universidade da Beira Interior, seguindo o despacho Reitoral n� 49/R/2010
Vers�o 2.2 - 2016/06/01

Em rela��o � Vers�o 2.1 na Vers�o 2.2 existem duas op��es para as Listas, Lista de Figuras e Lista de Tabelas, podem aparecer as palavras "Figura" e "Tabela" nas respectivas listas. Como exemplo:
	2.1 Correspond�ncia entre as cores das riscas das resist�ncias e o seu valor �hmico. .3
	ou
	Tabela 2.1 Correspond�ncia entre as cores das riscas das resist�ncias e o seu valor�hmico. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3

Em rela��o � Vers�o 2.0 na Vers�o 2.1 passa-se a deixar de compilar em PDFLaTeX para se passar a compilar em XeLaTeX.
� necessa?io passar a compilar em XeLaTeX para utilizar o tipo de letra Trebuchet.

Para utilizar o XeLaTeX a codifica��o dos ficheiros tem que ser em UTF-8.

Utilizadores de Linux com gestor de pacotes DEB t�m que ter o pacote "ttf-mscorefonts-installer" instalado
para utilizar o tipo de letra Trebuchet. N�o foram testados outros gestores de pacotes.

O modelo foi compilado em XeLaTeX sem erros num sistema de Windows 10 64-bit, com 
basic-MikTeX 2.9.5840 64-bit e Texmaker 4.5. Inclu�dos no .zip:

O modelo foi compilado em XeLaTeX sem erros num sistema de Windows 8.1 Profissional 64-bit, com 
basic-MikTeX 2.9.5840 64-bit e Texmaker 4.5. Inclu�dos no .zip:

O modelo foi compilado em XeLaTeX sem erros num sistema de Windows 7 Profissional 64-bit, com 
basic-MikTeX 2.9.5840 64-bit e TeXnicCenter 2.02 Stable 64-bit. Inclu�dos no .zip:

O modelo foi compilado em XeLaTeX e sem erros num sistema LinuxMint16 Cinnamon 64-bit e Linux Mint Debian Edition 2
64-bit (N�o foram testadas outras distribui��es), com Texmaker 4.0.3 e com texlive-full. Inclu�dos no .zip:

- Tese.tex, o ficheiro principal do documento;
- PaginaRosto.tex, que gera a p�gina de rosto;
- Intro.tex e Exemplos.tex, exemplos de cap�tulos com tabela, figura e refer�ncias;
- formatacaoUBI.tex e estiloUBI.sty, definem a formata��o da tese, n�o � recomend�vel 
editar estes ficheiros;
- estilo-biblio.bst, define o estilo da bibliografia, pode ser trocado por qualquer 
outro ficheiro de acordo com a norma a utilizar (deixada em aberto pelo despacho);
- bibliografia.bib, onde se inserem as refer�ncias da tese em formato bibTeX;
- direct�rio imagens, onde por defeito dever�o ser colocadas as imagens a utilizar.

Contribui��es, d�vidas e sugest�es para: latex@e-projects.ubi.pt

http://www.e-projects.ubi.pt/latex

