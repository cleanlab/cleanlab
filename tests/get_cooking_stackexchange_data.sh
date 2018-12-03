#!/bin/bash

# Modified from https://github.com/facebookresearch/fastText/blob/master/tests/fetch_test_data.sh#L111

DATADIR=$1

echo "Downloading cooking dataset to ${DATADIR}"

data_result="${DATADIR}"/cooking/cooking.stackexchange.txt
if [ ! -f "$data_result" ]
then
  mkdir -p "${DATADIR}"/cooking/
  wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/cooking.stackexchange.tar.gz -O "${DATADIR}"/cooking/cooking.stackexchange.tar.gz
  tar xvzf "${DATADIR}"/cooking/cooking.stackexchange.tar.gz --directory "${DATADIR}"/cooking || exit 1
  cat "${DATADIR}"/cooking/cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > "${DATADIR}"/cooking/cooking.preprocessed.txt
fi
