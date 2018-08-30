#!/bin/sh

for file in `ls zip/train/*zip`
do
    echo $file
    unzip -q $file -d ./tmp/
    find tmp/train/TC -type f | xargs -i mv "{}" data/train/TC
    find tmp/train/nonTC -type f | xargs -i mv "{}" data/train/nonTC  
    #find tmp/test -type f | xargs -i mv "{}" data/test/img
    rm -r ./tmp/*
done
