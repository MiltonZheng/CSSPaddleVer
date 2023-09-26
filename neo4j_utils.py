# coding: utf-8
import pickle
from neo4j import GraphDatabase, basic_auth
import datetime
import os

neo4j_url = "bolt://192.168.254.249:7687"
driver = GraphDatabase.driver(neo4j_url, auth=basic_auth("neo4j", "123456"))


# @cache.cached()
def retrieval(dataset, agent, radius):
    session = driver.session()
    tx = session.begin_transaction()

    hashdict = pickle.load(open(os.path.join(dataset, "hashDict.pkl"), 'rb'))
    curr_hash = agent
    # * this one retrieves nodes that are within {radius} steps from the agent
    cypher_findZeroOneSimiNodes = "MATCH p = (node:IHashNode{hashCode:'" + str(curr_hash) + \
                                  "' })-[:Simi*.."+str(radius)+"]-(SimiNode) " + \
                                  " RETURN  distinct reduce(sum=0, r in relationships(p) | sum + 1), SimiNode.hashCode"
    # * this one retrieves node that has the same hashcode as the agent
    # cypher_findZeroOneSimiNodes = "MATCH (node:IHashNode) where node.hashCode ='" + curr_hash + "' RETURN  node.hashCode"
    # * this one calculates the sum of the distances along the path
    # cypher_findZeroOneSimiNodes = "MATCH (node:IHashNode{hashCode:'" + curr_hash + \
    #                               "' })-[r:Simi]-(SimiNode)  WHERE r.dist<= " + str(radius) + \
    #                               " RETURN  distinct r.dist, SimiNode.hashCode"
    retrievalDict = {}
    SimiNodes_result = tx.run(cypher_findZeroOneSimiNodes)
    for k, v in SimiNodes_result:
        v = int(v)
        if k not in retrievalDict:
            retrievalDict.update({k: hashdict[v]})
        else:
            retrievalDict[k].extend(hashdict[v])
    tx.close()
    session.close()
    return retrievalDict
