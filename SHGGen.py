import h5py
import os
from tqdm import tqdm
import csv
import pickle

    
def hammingDist(hashstr1, hashstr2):
    """Calculate the Hamming distance between two bit strings"""
    return bin(int(hashstr1, 2) ^ int(hashstr2, 2)).count('1')


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
    hashcodestr = []

    node_type = 'IHashNode'
    relation_type = 'Simi'

    hashDict = {}
    txt_path_file = open(os.path.join(__PATH__, 'test_path.txt'))
    line = txt_path_file.readline()

    # 循环生成每个测试集图片的哈希码（每个哈希码作为一个结点）和 结点之间的关系
    for i in tqdm(range(len(hashcodes)), desc="computing hamming distance"):  
        if line:
            filepath = line.strip('\n')
            curr_hash = hashcodes[i]
            hashcodestr.append(curr_hash)
            # if the key already exists, then append the value
            if curr_hash in hashDict:
                hashDict[curr_hash].append(filepath)
            # if the key does not exist
            else:
                min_dist = 48
                min_hashcode = []
                backup = []
                # add this key to the dict
                hashDict[curr_hash] = [filepath]
                # write node to csv for neo4j
                node_writer.writerow([curr_hash, node_type])
                for j in range(i):
                    curr_dist = hammingDist(curr_hash, hashcodestr[j])
                    if curr_dist < min_dist:
                        # 这里对应DAC做出了改动，如果先2后1，那么把2单独存起来
                        if min_dist == 2:
                            # print("先1后2")
                            backup = min_hashcode
                        min_dist = curr_dist
                        # min_index=[j]
                        min_hashcode = [hashcodestr[j]]
                    elif curr_dist == min_dist:
                        if hashcodestr[j] not in min_hashcode:
                            min_hashcode.append(hashcodestr[j])
                            # min_index.append(j)
                    # 这里是先1后2，以及先2后1的更新阶段
                    # 先1后2：此时只需往backup里面加就行
                    # 先2后1：同样的做法
                    elif curr_dist == 2:
                        if hashcodestr[j] not in min_hashcode:
                            backup.append(hashcodestr[j])

                # write relationship to csv
                # m indicts the nodes that are similar with i
                batch_rows = []
                # for m in min_index:
                for m_hashcode in min_hashcode:
                    cur_rows = [curr_hash, min_dist, m_hashcode, relation_type]
                    batch_rows.append(cur_rows)
                if len(backup) > 0:
                    for hashcode_2 in backup:
                        cur_rows = [curr_hash, 2, hashcode_2, relation_type]
                        batch_rows.append(cur_rows)
                # add all similar relationships to CSV (similar with hashcode[i])
                relation_writer.writerows(batch_rows)
        line = txt_path_file.readline()

    with open(os.path.join(__PATH__, 'hashDict.pkl'), 'wb') as f:
        pickle.dump(hashDict, f)

    print(hashDict)
    csvfile_node.close()
    csvfile_relation.close()
    hyfile_node.close()
    txt_path_file.close()
