mkdir -p dataset/coco2014
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./dataset/coco2014/
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P ./dataset/coco2014/
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P ./dataset/coco2014/

unzip ./dataset/coco2014/captions_train-val2014.zip -d ./dataset/coco2014/
rm ./dataset/coco2014/captions_train-val2014.zip
unzip ./dataset/coco2014/train2014.zip -d ./dataset/coco2014/
rm ./dataset/coco2014/train2014.zip 
unzip ./dataset/coco2014/val2014.zip -d ./dataset/coco2014/ 
rm ./dataset/coco2014/val2014.zip 
