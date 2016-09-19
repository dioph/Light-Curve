from __future__ import print_function
import warnings, numpy as np, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import lc_io, lc_fourier
#import tkinter as tk    # GUI

arq=None
print('LIGHT CURVE ANALYSIS')
while True:
    option = str(input('n- use new file\np- plot current file\nx- plot pixels\nq- quit\nCurrent file: '+str(arq)+'\n')).lower()
    if option == 'n':
        arq = input('Please insert file name: ')
        try:
            hdulist = lc_io.open(arq)
        except:
            arq=None
            continue
        lc_io.close(hdulist)
    elif option == 'p':
        yname = input('Data column name: ')
        try:
            lc_io.plot(arq, yname)
        except:
            continue
        else:
            plt.show()
    elif option == 'x':
        lc_io.plotpixel(arq)
        plt.show()
    elif option == 'q':
        break;
    else:
        lc_io.error('Type a valid option')
        continue
