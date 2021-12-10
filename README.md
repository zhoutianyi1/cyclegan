# CycleGAN implementation

## Dataset
The data we use is from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/), choose the ```maps.zip``` to download.

## Run the code yourself
Place the ```maps``` folder at the top directory(or other kinds of dataset of your choice), then in the ```main.py```, set the ```MODE='train'``` if you want to train your own models. You can tweak batch size of epochs according to your will as well. Type```python main.py``` to run the code. After running, you will have ```ba.pth```, ```ab.pth```. \
For testing, tweak ```MODE='test'```, and input A domain images(where you need to manually change the image path in the code), type ```python main.py``` in the terminal to run the code. Then the Generator will give you B domain images.