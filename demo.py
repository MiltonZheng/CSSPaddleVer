from PIL import Image
import datetime
import os
import h5py
import numpy as np
import pickle

from neo4j_utils import retrieval

testset = "../datasets/NWPU-RESISC45/test/"
query_num = 100
file_num = 6500


def jpg2pdfByList(origin, sim_list):
    imglist = []
    files = [origin] + sim_list
    for f in files:
        img = Image.open(f)
        imglist.append(img)

    imgMerge = imglist.pop(0)

    imgMerge.save("./merge.pdf", "PDF", resolution=100.0,
                  save_all=True, append_images=imglist)
    print("origin and similar img have been merged into pdf!")


if __name__ == "__main__":
    access = np.random.randint(0, file_num, query_num)
    distance = np.random.randint(1, 3, query_num)
    hashdict = pickle.load(open(os.path.join(testset, "hashDict.pkl"), 'rb'))
    hashcodes = h5py.File(os.path.join(testset, "hashcodes.hy"), 'r')
    hashcodes = hashcodes['hashcodes'][()]
    test_path = open(os.path.join(testset, 'test_path.txt')).readlines()
    
    oldtime = datetime.datetime.now()
    for i in range(query_num):
        id = access[i]
        hashcode = hashcodes[id]
        dis = distance[i]
        agent_path = test_path[id].strip('\n')
        print("query: ", agent_path)
        res = retrieval(testset, hashcode, dis)  
        res_distance_list = list(res.keys())
        count = 0
        res_paths = []
        for key in res_distance_list:
            if count >= 10:
                break
            for item in res[key]:
                if count >= 10:
                    break
                count += 1
                res_paths.append(item)
        jpg2pdfByList(agent_path, res_paths)
        break
