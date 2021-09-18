echo 'Downloading and setting up Dog_Cat_classifier model'
DEST_DIR='saved_model'

ggID='19p56gKV5EvMJCedIybkKaVCcGEBpCeZw'  
ggURL='https://drive.google.com/uc?export=download'  
FILENAME="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${FILENAME}"  
mkdir $DEST_DIR
mv $FILENAME $DEST_DIR
echo 'Done'
