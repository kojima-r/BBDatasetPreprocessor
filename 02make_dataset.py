import scipy
import glob
import numpy as np
import json
import os
from multiprocessing import Pool

def process(args):
    label=args["label"]
    name=args["name"]
    feature=args["feature"]
    filename="data_npy/xc"+name+"."+feature+".npy"
    print("[LOAD]",filename)
    f=np.load(filename)
    return name,label,f

def main():
    data=[]
    filename="birdsong_metadata.csv"
    feature="spec"
    test_ratio=0.2
    fp=open(filename)
    next(fp)
    for line in fp:
        arr=line.strip().split(",")
        name=arr[0]
        label=arr[2]
        data.append({"label":label,"name":name,"feature":feature})
    p = Pool(128)
    results=p.map(process, data)
    p.close()

    np.random.seed(1234)

    ml=max([r[2].shape[1]  for r in results])
    feature_num=results[0][2].shape[0]
    n=len(results)
    print("results:",n)

    data=np.zeros((n,feature_num,ml),dtype=np.float32)
    step_data=np.zeros((n,),dtype=np.int32)
    label_data=np.zeros((n,ml),dtype=np.int32)
    name_list=[]
    label_mapping={}
    for i,r in enumerate(results):
        name,label,f=r
        s=f.shape[1]
        data[i,:,:s]=f
        step_data[i]=s
        if label not in label_mapping:
            label_mapping[label]=len(label_mapping)
            label_data[i,:]=label_mapping[label]
        name_list.append(name)
    print(data.shape)
    data=np.transpose(data,[0,2,1])
    ##
    all_idx=list(range(n))
    np.random.shuffle(all_idx)
    m=int(n*test_ratio)
    train_idx=all_idx[:n-m]
    test_idx=all_idx[n-m:]
    info={}
    info["pid_list_train"]=[name_list[i] for i in train_idx]
    info["pid_list_test"]=[name_list[i] for i in test_idx]
    ##
    os.makedirs("dataset",exist_ok=True)
    train_data=data[train_idx,:,:]
    train_step_data=step_data[train_idx]
    train_label_data=label_data[train_idx,:]
    filename="dataset/train_data."+feature+".npy"
    np.save(filename,train_data)
    filename="dataset/train_step."+feature+".npy"
    np.save(filename,train_step_data)
    filename="dataset/train_label."+feature+".npy"
    np.save(filename,train_label_data)
    ##
    test_data=data[test_idx,:,:]
    test_step_data=step_data[test_idx]
    test_label_data=label_data[test_idx,:]
    filename="dataset/test_data."+feature+".npy"
    np.save(filename,test_data)
    filename="dataset/test_step."+feature+".npy"
    np.save(filename,test_step_data)
    filename="dataset/test_label."+feature+".npy"
    np.save(filename,test_label_data)

    #print(ml)
    fp = open("dataset/info."+feature+".json", "w")
    json.dump(info, fp)
    fp = open("dataset/label."+feature+".json", "w")
    json.dump({v:k for k,v in label_mapping.items()}, fp)

if __name__ == '__main__':
    main()
