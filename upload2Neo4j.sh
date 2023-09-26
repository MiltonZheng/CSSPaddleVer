#!/bin/bash

firstNodePath=../datasets/NWPU-RESISC45/test/first_hash_nodes.csv
firstRelationPath=../datasets/NWPU-RESISC45/test/first_hash_relationships.csv
neo4j stop

rm -rf /root/neo4j-community-4.4.18/data/databases/neo4j/*
rm -rf /root/neo4j-community-4.4.18/data/transactions/neo4j/*
starttime=$(date +%s)

/root/neo4j-community-4.4.18/bin/neo4j-admin import --nodes=IHashNode=${firstNodePath} --relationships=Simi=${firstRelationPath} --id-type=STRING
endtime=$(date +%s)
cost=$((endtime - starttime))

neo4j start

echo $cost
