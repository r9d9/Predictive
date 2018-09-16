import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    """taken from https://stackoverflow.com/questions/42622146/scrollbar-on-matplotlib-showing-page"""
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        #exit(self.qapp.exec_())
        self.qapp.exec_() # this instead of the previous line makes the program continue after closing the window


"""example of usage:

# create a figure and some subplots
fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(16,16))
for ax in axes.flatten():
    ax.plot([2,3,5,1])

# pass the figure to the custom window
a = ScrollableWindow(fig)
"""

def create_error_figures(error_df,error_df_tr,x_array,name,lim):
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    ax = axes.flatten()
    for col in error_df.iloc[:, 1:6].columns.values:
        axnr = error_df.columns.get_loc(col) - 1
        ax[axnr].set_title(col)
        ax[axnr].scatter(x_array, error_df[col], label="val")
        ax[axnr].scatter(x_array, error_df_tr[col], label="train")
        ax[axnr].legend()
        if lim == 1:
            ax[axnr].set_xlim([0, 50])
        ax[axnr].set_xlabel(name)
    fig.tight_layout()
    return fig
    #plt.show()

