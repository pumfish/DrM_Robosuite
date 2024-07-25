import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

parser = argparse.ArgumentParser(description='Script to explore the data generated by an experiment.')
parser.add_argument('--data', '-d', type=str, required=True, help='location of the .pickle log data file')
parser.add_argument('--output', '-o', type=str, required=True, help='location to store results as a png')
parser.add_argument('--xkey', '-x', type=str, default=None, help='the key to use for x axis in plots')
parser.add_argument('--xscale', '-s', type=int, default=1, help='scaling for the x axis (optional)')
args = parser.parse_args()

# get inputs and setup output file
if '.png' in args.output:
    OUT_FILE = args.output
else:
    OUT_FILE = args.output + '/plot.png'
data = pickle.load(open(args.data, 'rb'))
xscale = 1 if args.xscale is None else args.xscale
if args.xkey == 'num_samples':
    xscale = xscale if 'act_repeat' not in data.keys() else data['act_repeat'][-1]

dict_keys = list(data.keys())
for k in dict_keys:
    if len(data[k]) == 1: del(data[k])

# plot layout
nplt = len(data.keys())
ncol = 4
nrow = int(np.ceil(nplt/ncol))

# plot data
xkey = args.xkey
start_idx = 2
end_idx = max([len(data[k]) for k in data.keys()])
xdata = np.arange(end_idx) if (xkey is None or xkey == 'None') else \
        [np.sum(data[xkey][:i+1]) * xscale for i in range(len(data[xkey]))]

# make the plot
plt.figure(figsize=(15,15), dpi=60)
for idx, key in enumerate(data.keys()):
    plt.subplot(nrow, ncol, idx+1)
    plt.tight_layout()
    try:
        last_idx = min(end_idx, len(data[key]))
        plt.plot(xdata[start_idx:last_idx], data[key][start_idx:last_idx], color=colors[idx%7], linewidth=3)
    except:
        pass
    plt.title(key)

plt.savefig(OUT_FILE, dpi=100, bbox_inches="tight")
