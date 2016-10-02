from __future__ import print_function
import warnings, numpy as np, matplotlib.pyplot as plt, os
warnings.filterwarnings("ignore")
import lc_io, lc_fourier, lc_tpf, lc_fit, lc_realtime

while True:
    try:
        x = int(input('1- analise de arquivos FITS\n2- analise em tempo real\n'))
    except:
        print('Opcao invalida')
        continue
    else:
        if x == 1 or x == 2:
            break
        else:
            print('Opcao invalida')
            continue
if x == 2:
    lc_realtime.realtime()
else:
    arq=None
    print('*-----ANALISE DE CURVAS DE LUZ-----*')
    while True:
        opcao = str(input('n- usar um novo arquivo \
            \np- plotar curva de luz do arquivo atual\
            \nx- plotar os pixels do alvo atual\
            \nm- definir uma nova mascara para o alvo atual\
            \ne- extrair nova curva de luz aplicando mascara\
            \nt- remover CBVs da curva de luz\
            \nq- sair\
            \nArquivo atual: '+str(arq)+'\n')).lower()
        if opcao == 'n':
            arq = input('Insira o nome do arquivo: ')
            try:
                hdulist = lc_io.abrir(arq)
            except:
                arq=None
                continue
            lc_io.fechar(hdulist)
        elif opcao == 'p':
            coluna = input('Nome da coluna de dados: ')
            lc_io.plot(arq, coluna)
        elif opcao == 'x':
            lc_tpf.plotpixel(arq)
        elif opcao == 'm':
            mask = input('Nome da mask de pixels de saida .txt: ')
            lc_tpf.plotmask(arq, mask)
        elif opcao == 'e':
            novo_arq = input('Nome do arquivo .fits de saida: ')
            mask = input('Nome da mask de pixels .txt: ')
            lc_io.new_curve(arq, mask, novo_arq)
        elif opcao == 't':
            novo_arq = input('Nome do arquivo .fits de saida: ')
            cbv_list = input('Lista de vetores CBV: ')
            lc_fit.tendencia(arq, novo_arq, cbv_list)
        elif opcao == 'q':
            break;
        else:
            lc_io.error('Insira uma opcao valida')
            continue
