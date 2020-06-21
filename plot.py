import numpy as np
import joblib
import json
import sys
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap
import argparse
from matplotlib import pylab as plt
from multiprocessing import Pool

def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)

def process(args):
    el=args["label"]
    name=args["name"]
    feature=args["feature"]

    filename="data_npy/xc"+name+"."+feature+".npy"
    print("[LOAD]",filename)
    o=np.load(filename)
    print(o.shape)
    cmap = generate_cmap(["#0000FF", "#FFFFFF", "#FF0000"])
    plt.figure(figsize=(4*o.shape[1]/500, 4*o.shape[0]/500.0), dpi=250)
    plt.imshow(
        o, aspect=0.5, interpolation="none", cmap=cmap
    )
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().invert_yaxis()
    plt.title(el)

    out_filename="data_plot/"+name+"."+feature+".png"
    print(out_filename)
    plt.savefig(out_filename)



def main():
    data=[]
    filename="birdsong_metadata.csv"
    feature="spec"
    fp=open(filename)
    next(fp)
    for line in fp:
        arr=line.strip().split(",")
        name=arr[0]
        label=arr[2]
        data.append({"label":label,"name":name,"feature":feature})

    os.makedirs("data_plot",exist_ok=True)
    p = Pool(64)
    results=p.map(process, data)
    p.close()

main()

