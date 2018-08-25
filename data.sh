#!/bin/sh

for file in `ls zip/*zip`
do
    echo $file
    unzip -q $file -d ./tmp/
    find tmp/train/TC -type f | xargs -i mv "{}" train/TC
    find tmp/train/nonTC -type f | xargs -i mv "{}" train/nonTC
    rm -r ./tmp/*
done
