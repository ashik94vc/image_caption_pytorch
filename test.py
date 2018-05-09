from torchvision import datasets as ds

cap = ds.CocoCaptions(root="dataset/coco2014/train2014", annFile="dataset/coco2014/annotations/captions_train2014.json")

print(cap[0])
