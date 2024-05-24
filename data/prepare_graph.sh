#!/bin/bash


wget https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
wget https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gunzip com-amazon.ungraph.txt.gz
gunzip com-orkut.ungraph.txt.gz
gunzip com-friendster.ungraph.txt.gz