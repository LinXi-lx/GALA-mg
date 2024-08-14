#!/bin/bash


wget https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
wget https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gunzip com-lj.ungraph.txt.gz
gunzip com-orkut.ungraph.txt.gz
gunzip com-friendster.ungraph.txt.gz

tar -xvf depend.tar

download_and_convert_graph() {
    local graph_name=$1


    for ext in .properties .graph .md5sums; do
        wget -c "http://data.law.di.unimi.it/webdata/${graph_name}/${graph_name}${ext}"
    done

    java -cp "depend/*" it.unimi.dsi.webgraph.BVGraph -o -O -L "${graph_name}"
    java -cp "depend/*" it.unimi.dsi.webgraph.ArcListASCIIGraph -g BVGraph "${graph_name}" "${graph_name}.edgelist"
}

download_and_convert_graph twitter-2010
download_and_convert_graph uk-2002 
download_and_convert_graph enwiki-2022 
download_and_convert_graph hollywood-2011