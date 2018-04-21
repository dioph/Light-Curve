import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtGui
from PyQt5.QtWidgets import *
from astroquery.vizier import Vizier
import kplr
import sys, re

class interface(QWidget):
    def __init__(self, parent=None):
        super(interface, self).__init__(parent=parent)
        self.lst_size = 10
        self.client = kplr.API()
        self.layout = QHBoxLayout(self)
        
        self.lay_search = QVBoxLayout()
        self.bt_search = QPushButton('SEARCH')
        self.bt_search.clicked.connect(self.search)
        self.txt_search = QLineEdit()
        self.txt_search.returnPressed.connect(self.search)
        self.msg_search = QLabel(self)
        self.lst_search = QListWidget()
        self.lst_search.itemActivated.connect(self.info)
        self.lay_search.addWidget(self.bt_search)
        self.lay_search.addWidget(self.txt_search)
        self.lay_search.addWidget(self.msg_search)
        self.lay_search.addWidget(self.lst_search)
        self.layout.addLayout(self.lay_search)
        
        self.lay_info = QVBoxLayout()
        self.lab_info = QLabel(self)
        self.lay_info.addWidget(self.lab_info)
        self.layout.addLayout(self.lay_info)
        
        self.win = pg.GraphicsWindow()
        self.layout.addWidget(self.win)
        
        self.lay_menu = QVBoxLayout()
        self.menu = QComboBox()
        self.menu.addItem('')
        self.menu.addItem('TPF')
        self.menu.activated.connect(self.update)
        self.lay_menu.addWidget(self.menu)
        self.layout.addLayout(self.lay_menu)
        
    def search(self):
        txt = self.txt_search.text()
        self.lst_search.clear()
        self.msg_search.setText('')
        try:
            v = Vizier(column_filters={'KIC':txt})
            stars = v.get_catalogs('J/ApJS/229/30/catalog')[0]
        except:
            try:
                dics = txt.split(',')
                keys = []
                vals = []
                for s in dics:
                    pair = re.split('<|=|>', s)
                    keys.append(pair[0])
                    if '>' in s:
                        pair[1] = '>'+pair[1]
                    elif '=' in s and '..' not in pair[1]:
                        pair[1] = '='+pair[1]
                    elif '<' in s:
                        pair[1] = '<'+pair[1]
                    vals.append(pair[1])
                kwargs = dict(zip(keys, vals))
                v = Vizier(column_filters=kwargs)
                stars = v.get_catalogs('J/ApJS/229/30/catalog')[0]
            except:
                self.msg_search.setText('<font color="red"> WRONG SEARCH FORMAT </font>')
                return
        for star in stars:
            QListWidgetItem(str(star['KIC']), self.lst_search)
        
    def info(self, item):
        kic = item.text()
        v = Vizier(column_filters={'KIC':'='+kic})
        stars = v.get_catalogs('J/ApJS/229/30/catalog')[0]
        star = stars[0]
        self.kic = star['KIC']
        self.lab_info.setText(u'KIC {0}\n'
                            u'Teff={1} K\n'
                            u'logg={2:.3f}\n'
                            u'Dist={3:.3f} kpc\n'
                            u'Mass={4:.3f} M\u2609\n'
                            u'RA={5:.3f}\n'
                            u'Dec={6:.3f}'.format(star['KIC'], star['Teff'], star['log_g_'], star['Dist'], star['Mass'], star['_RA'], star['_DE']))
                            
    def update(self, arg_1):
        star = self.client.star(self.kic)
        tpf = star.get_target_pixel_files()
        print(tpf)
           
def start():
    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    font = QtGui.QFont('Courier')
    app.setFont(font)
    new = interface()
    new.show()
    app.exit(app.exec_())
    return new
    
if __name__ == "__main__":
    new = start()
