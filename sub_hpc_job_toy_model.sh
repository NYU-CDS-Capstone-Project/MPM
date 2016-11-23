#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l mem=32GB
#PBS -N toy_model
#PBS -M mc3784@nyu.edu
#PBS -j oe
#PBS -m e


module purge

SRCDIR=$HOME/CAPSTONE/MPM
RUNDIR=$SCRATCH/toy_model/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cp -R $SRCDIR/plots $RUNDIR
cp $SRCDIR/toy_model.py $RUNDIR 

cd $RUNDIR

module load scipy/intel/0.16.0

python toy_model.py
