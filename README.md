## Training
~~~
python train.py config/train_wikitext.py
~~~

Note: you can modify training configurations based on your device.
Modify Line 10 in train.py to switch between different model architectures.
By default we use cpu to train the model. If you want to use a gpu, please modify Line 51 of train.py.

## Sampling
~~~
python sample.py --out_dir=YOUR_MODEL_DIR_PATH
~~~
Modify Line 9 in sample.py to switch between different model architectures.