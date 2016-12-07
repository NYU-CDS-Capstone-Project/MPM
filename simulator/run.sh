#!/bin/bash

export TOP=https://raw.githubusercontent.com/lukasheinrich/weinberg-exp/master/example_yadage

function yadage-run {
   docker run --rm -e PACKTIVITY_WITHIN_DOCKER=true -v $PWD:$PWD -w $PWD -v /var/run/docker.sock:/var/run/docker.sock lukasheinrich/yadage yadage-run $*
}

echo n_events $1
echo sqrtshalf $2
echo polbeam1 $3
echo polbeam2 $4

yadage-run -t $TOP workdir rootflow.yml -p nevents=$1 -p seeds=[1,2,3,4] -a $TOP/input.zip \
           -p runcardtempl=run_card.templ -p proccardtempl=sm_proc_card.templ \
           -p sqrtshalf=$2 -p polbeam1=$3 -p polbeam2=$4