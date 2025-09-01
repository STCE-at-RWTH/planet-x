#! /usr/bin/env bash

HOST=login23-1.hpc.itc.rwth-aachen.de
REMOTE_HOME=$(ssh $TIM_ID@$HOST pwd)

echo "Using CLAIX user $TIM_ID".
echo "Will try to copy from $TIM_ID 's home directory: $REMOTE_HOME"
echo "Searching from: $1."
echo "Expecting format <...>/data/JOB_ID/ARRAY_JOB_INDEX/<output files>."
echo "Searching for array job ID: $2"

ARRAY_JOB_DIRS=($(ssh $TIM_ID@$HOST "cd $REMOTE_HOME/$1/$2 && ls -d */"))
echo "Found ${#ARRAY_JOB_DIRS[@]} matching jobs."
echo "${ARRAY_JOB_DIRS[@]}"

ARGS=("$@")
NFILES_TO_FETCH=$(($# - 2))
FILES_TO_FETCH=(${ARGS[@]:2:$NFILES_TO_FETCH})
echo "${arr[@]}"
for j in ${ARRAY_JOB_DIRS[@]}; do
  mkdir -p $2/$j
  if [[ $NFILES_TO_FETCH -gt 0 ]]; then
    for f in ${FILES_TO_FETCH[@]/#/$REMOTE_HOME/$1/$2/$j}; do
      scp -r $TIM_ID@$HOST:$f $2/$j
    done
  else
    scp -r $TIM_ID@$HOST:$REMOTE_HOME/$1/$2/$j $2
  fi
done
