wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar -xf 'images.tar' -C '.'
rm -rf images.tar
mkdir images
python train_test_split.py
rm -rf Images