import h5py
import os
from tqdm import tqdm
import csv
import pickle


__PATH__ = '../datasets/NWPU-RESISC45/test'

if __name__ == "__main__":
    # nodes
    csvfile_node = open(os.path.join(__PATH__, 'first_hash_nodes.csv'), 'w')
    node_writer = csv.writer(csvfile_node)
    node_writer.writerow(['hashCode:ID', ':LABEL'])

    # relationships
    csvfile_relation = open(os.path.join(__PATH__, 'first_hash_relationships.csv'), 'w')
    relation_writer = csv.writer(csvfile_relation)
    relation_writer.writerow([':START_ID', 'dist:int', ':END_ID', ':TYPE'])

    hyfile_node = h5py.File(os.path.join(__PATH__, 'hashcodes.hy'), 'r')
    hashcodes = hyfile_node['hashcodes'][()]

    node_type = 'IHashNode'
    relation_type = 'Simi'

    hashDict = {}
    txt_path_file = open(os.path.join(__PATH__, 'test_path.txt'))
    line = txt_path_file.readline()

    total = len(hashcodes)
    # 循环生成每个测试集图片的哈希码（每个哈希码作为一个结点）和 结点之间的关系
    for i in tqdm(range(total), desc="computing hamming distance"):
        if line:
            filepath = line.strip('\n')
            curr_hash = hashcodes[i]
            # if the key already exists, then append the value
            if curr_hash in hashDict:
                hashDict[curr_hash].append(filepath)
            # if the key does not exist
            else:
                # add this key to hashdict and record the node
                hashDict[curr_hash] = [filepath]
                node_writer.writerow([curr_hash, node_type])
                
                #! calcute the hamming distance between the current hashcode and all the hashcodes
                min_dis = 48
                min_hashcode1 = []
                # * this one is specifically designed for the case of threshold 2
                min_hashcode2 = []
                for j in range(total):
                    if j == i:
                        continue
                    curr_dis = bin(curr_hash ^ hashcodes[j]).count('1')
                    if curr_dis < min_dis:
                        if min_dis == 2:
                            # * that means curr_dis is 1
                            # * but we still need to keep to ones that are 2
                            min_hashcode2 = min_hashcode1
                        # * update
                        min_dis = curr_dis
                        min_hashcode1 = [hashcodes[j]]
                    elif curr_dis == min_dis:
                        if hashcodes[j] not in min_hashcode1:
                            min_hashcode1.append(hashcodes[j])
                    # * if curr_dis is greater than min_dist but equal to 2
                    # * that means the threshold is 2, we can keep it in min_hashcode2
                    elif curr_dis == 2:
                        if hashcodes[j] not in min_hashcode2:
                            min_hashcode2.append(hashcodes[j])

                # write relationship to csv
                # m indicts the nodes that are similar with i
                batch_rows = []
                for m_hashcode in min_hashcode1:
                    cur_rows = [curr_hash, min_dis, m_hashcode, relation_type]
                    batch_rows.append(cur_rows)
                for hashcode_2 in min_hashcode2:
                    cur_rows = [curr_hash, 2, hashcode_2, relation_type]
                    batch_rows.append(cur_rows)
                relation_writer.writerows(batch_rows)
        line = txt_path_file.readline()

    with open(os.path.join(__PATH__, 'hashDict.pkl'), 'wb') as f:
        pickle.dump(hashDict, f)

    print(hashDict)
    csvfile_node.close()
    csvfile_relation.close()
    hyfile_node.close()
    txt_path_file.close()
