import cv2
import numpy as np
import time
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
from astropy.stats import LombScargle

def realtime():
	timedata = np.array([], dtype='float64')
	fluxdata = np.array([], dtype='float32')
	app = QtGui.QApplication([])
	win = pg.GraphicsWindow(title='real time')
	win.resize(1000, 1000)
	pg.setConfigOptions(antialias=True)
	pw = win.addPlot()
	cam = cv2.VideoCapture(0)
	while time.clock() < 60:
			ret, img = cam.read()
			timedata = np.append(timedata, time.clock())
			fluxdata = np.append(fluxdata, np.mean(img))
			pw.plot(timedata, fluxdata)
			time.sleep(0.001)
			cv2.imshow('eggs', img)
			if cv2.waitKey(1) == 27:
				break

	newflux = np.array([], dtype='float32')
	newtime = np.array([], dtype='float64')
	k = 0
	while k < len(fluxdata) and fluxdata[k] <= 50:
		k+=1
	while k < len(fluxdata):
		newflux = np.append(newflux, fluxdata[k])
		newtime = np.append(newtime, timedata[k])
		k += 1
	freq, power = LombScargle(newtime, newflux).autopower()
	best = freq[np.argmax(power)]
	xf = newtime
	yf = LombScargle(newtime, newflux).model(xf, best)
	win.removeItem(pw)
	p1 = win.addPlot()
	p1.plot(newtime, newflux)
	p2 = win.addPlot()
	p2.plot(freq, power)
	p3 = win.addPlot()
	p3.plot(xf, yf)
	cv2.destroyAllWindows()
	app.exec_()

if __name__ == '__main__':
	realtime()
