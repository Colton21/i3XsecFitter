#!/bin/bash

# parameter space and variables
normmin=50
normmax=200
n_spacing=2

parallelcounter=0

jobtype="prop"
FLUX='atmo'
echo $FLUX

# fields
dagfile="/scratch/chill/nuSQuIDS/propagate_${FLUX}.dag"
jobname='nuSQuIDS_propagate'
jobcounter=0


# create empty dagman files
: > $dagfile

# write tablemaker dag
for ((n = $normmin; n <= $normmax; n++))
do
    if [ $(( $n % $n_spacing )) -eq 0 ]
    then
        jobid=$jobname\_$n\_$FLUX
        norm=$n
        flux=$FLUX
        echo $jobid
        echo JOB $jobid propagate.submit >> $dagfile
        echo VARS $jobid norm=\"${norm}\" flux=\"${FLUX}\" type=\"${jobtype}\">> $dagfile
        ((jobcounter++))
    fi
done

# be chatty
echo DAGMan will process $jobcounter jobs
echo the dagman output has been written to $dagfile
