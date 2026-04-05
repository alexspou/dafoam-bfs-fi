#!/usr/bin/env bash
set -e

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit 1
fi

shopt -s extglob

# 1. Generate mesh
echo "Generating mesh..."
blockMesh
checkMesh

# 2. Run SST reference
echo "Running SST reference..."
cp constant/turbulenceProperties_SST constant/turbulenceProperties
cp 0_orig/* 0/

decomposePar
mpirun -np 4 python runPrimal.py
reconstructPar
rm -rf processor*

# 3. Extract reference data
LAST_TIME=$(foamListTimes -latestTime | tail -1)
echo "Extracting reference data from time: $LAST_TIME"

getFIData -refFieldName U -refFieldType vector -time $LAST_TIME
getFIData -refFieldName p -refFieldType scalar -time $LAST_TIME

# 4. Copy reference data to 0/ and clean up
cp -rf ${LAST_TIME}/*Data* 0/
rm -rf ${LAST_TIME}

# 5. Switch to k-omega baseline
cp constant/turbulenceProperties_KW constant/turbulenceProperties

echo "Done! Next: mpirun -np 4 python runScript_FI.py 2>&1 | tee logOpt.txt"
