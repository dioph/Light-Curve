import numpy as np, scipy as sci

# raiz quadrada da soma dos quadrados
def sumerr(a):
    tot = 0.0
    for i in a:
        tot += i**2
    err = np.sqrt(tot)
    return err

# Transformada de Fourier
def fourier(x, y, f1, f2, df):
    '''
    Transformada de Fourier

    x:   valores de tempo
    y:   valores dependentes do tempo
    f1:  frequência do início do intervalo
    f2:  frequência do fim do intervalo
    df:  tamanho dos passos de frequência
    '''

    real = []
    imag = []
    xf = []
    yf = []
    passo = 0
    n = len(x)
    for f in np.arange(f1, f2, df):
        real.append(0.0)
        imag.append(0.0)
        for i in range(n):
            real[-1] += y[i] * np.cos(2.0 * np.pi * f * x[i])
            imag[-1] += y[i] * np.sin(2.0 * np.pi * f * x[i])
        xf.append(f)
        if n > 0:
            yf.append((real[-1]**2 + imag[-1]**2) / n**2)
        else:
            yf.append(np.nan)
        passo += 1
    xf = np.array(xf, dtype='float32')
    yf = np.array(np.sqrt(yf), dtype='float32')
    return xf, yf

# Interpolação Gaussiana 2D
def gaussiana(params, *args):
    '''
    Interpolação Gaussiana 2D

    params:
    --cen:   centro (x,y)
    --A:     amplitude
    --sigma: largura (x,y)
    --B:     amplitude do termo de rotação
    --D:     background
    args:
    --pos:   posição para onde extrapolar a curva (x,y)
    --flux:  fluxo (variavel dependente)
    '''

    cen = [params[0],params[1]]
    A = params[2]
    sigma = [params[3],params[4]]
    B = params[5]
    D = params[6]

    posx = args[0]
    posy = args[1]
    flux = args[2]

    dx = posx - cen[0]
    dy = posy - cen[1]
    z = (np.square(dx) / sigma[0]**2) + (np.square(dy) / sigma[1]**2)
    g = A * sci.exp(-z - B * dx * dy) + D

    res = np.square(flux-g)
    return res
