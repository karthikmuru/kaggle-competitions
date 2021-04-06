#### Training

```
python3 ./object_recognition/train.py --batch_size 16 --epochs 15 --save_path ./weights/cifar10_weights.pth
```

#### Evaluating the trained model with the test set

```
python3 ./object_recognition/evaluate.py --load_path ./weights/cifar10_weights.pth
```
