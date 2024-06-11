###
# Source code written by Gurprit Singh
# (Max-Planck-Institute for Informatics, Saarbrucken)
###

# Added to make sure matplotlib don't use  Xwindows backend.
import matplotlib
matplotlib.use('Agg')
#######################
import numpy as np
import imageio
import array
#import Imath, OpenEXR
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['font.family'] = 'serif'
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=14)
rc('legend', fontsize=14)
rc('text.latex', preamble=r'\usepackage{cmbright}')

from matplotlib import pyplot as plt
import os
import sys, getopt
import argparse

### Helper functions
def readTXT(path):
    if not path.startswith('.DS_Store'):
        return np.loadtxt(open(path, 'rb'), delimiter=' ')

def append_it(filename, id):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + [id])

def loadTxtFilesFromFolder(folder, suffix='.txt'):
    data = []
    counter = 0
    for filename in os.listdir(folder):
        if not filename.startswith('.DS_Store') and filename.endswith(suffix):
            #yield filename
            filenameOrder = append_it(filename, counter)
            print (filenameOrder)
            counter+=1
            filedata = readTXT(os.path.join(folder, filename))
            if filedata is not None:
                data.append(filedata)
    return data

########

## plotting function
def variancePlot(folderpath, outfilename, legends, colors, xlimit, ylimit,dataorder, Markers,legendPos, suffix):
    varianceData = loadTxtFilesFromFolder(folderpath, suffix)

    nbFiles = len(varianceData)
    allData = np.array(varianceData[:][:])
    #print(allData)
    orderedData = allData[dataorder,:,:]
    #print(orderedData)
    allLogXdata = (orderedData[:,:,0]);
    allLogYdata = (orderedData[:,:,1]);
    #lineStyle = ['-','-.','-','-.','-','-.']
    #Markers = ['','','','']


    print('xlimit: ',np.min(allLogXdata),np.max(allLogXdata))
    print('ylimit: ',np.min(allLogYdata),np.max(allLogYdata))

    #print(np.max(varianceData[:][:,0]))
    #print(np.max(varianceData[:][:,0]))

    print('nbfiles: ',nbFiles)

    fig,ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim([ylimit[0], ylimit[1]])
    ax.set_xlim([xlimit[0], xlimit[1]])
    #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #plt.xlim( (0, xlimit) )
    #plt.ylim( (0, ylimit) )
    #start, end = ax.get_xlim()
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #ax.set_aspect(0.5)
    #plt.savefig(outfilename, bbox_inches=extent)
    #plt.show()

    for i in np.arange(nbFiles):
        nbBins = len(orderedData[i][:,0])
        x = orderedData[i][0:nbBins,0]
        y = orderedData[i][0:nbBins,1]
        #line1, = ax.plot(np.log2(x[:]),np.log2(y), color=colors[i], label=legends[i])
        line1, = ax.plot((x[:]),(y), color=colors[i], label=legends[i], marker=Markers[i])
        #line1, = ax.plot((x[:]),(y), color=colors[i], label=legends[i], linestyle=lineStyle[i], marker=Markers[i])
        #ax.fill_between(x[:], 0, y, color=colors[i], alpha=0.1)
        plt.legend(loc=legendPos)

    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(outfilename,bbox_inches='tight',pad_inches=0.05)
    plt.close()

def main(argv):
    # defined command line options
    # this also generates --help and error handling
    parser=argparse.ArgumentParser()
    parser.add_argument('-s','--suffix', default='.txt', help='<Required> Set flag', required=False)

    parser.add_argument('-x','--xlimit', type=float, nargs='*', help='2D x-coordinates', required=True)
    parser.add_argument('-y','--ylimit', type=float, nargs='*', help='2D y-coordinates', required=True)
    parser.add_argument('-i','--ifile', help='<Required> Set flag', required=True)
    parser.add_argument('-o','--ofile', help='<Required> Set flag', required=True)
    parser.add_argument('-m','--dataorder', type=int, nargs='*', help='array 0 1 2 3 ... nbFiles', required=True)
    parser.add_argument('-l','--legends', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-t','--linestyle',type=str,  nargs='*', help='array of line styles for each curve', required=False)
    parser.add_argument('-p','--legendpos',type=int, default=3, help='Any integer within 0-3', required=False)
    parser.add_argument('-z','--markers',type=str,  nargs='*', help='array of markers for each curve', required=True)
    parser.add_argument('-c','--colors', nargs='+', help='<Required> Set flag', required=True)
    #parser.add_argument('-e','--scale', type=int, help='<Required> Set flag', required=True)


    # parse the command line
    args = parser.parse_args()
    # access CLI options
    inputfile = args.ifile
    outputfile = args.ofile
    legends = args.legends
    colors = args.colors
    xlimit = args.xlimit
    ylimit = args.ylimit
    suffix = args.suffix
    lineStyle = args.linestyle
    Markers = args.markers
    dataorder = args.dataorder
    legendPosition = args.legendpos

    variancePlot(inputfile, outputfile, legends,colors, xlimit, ylimit, dataorder, Markers, legendPosition, suffix)


if __name__ == "__main__":
   main(sys.argv[1:])
