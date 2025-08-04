# Knowledge-Driven Baseline for Few-shot Fine-Grained Visual Recognition

PyTorch implementation of Knowledge-Driven Baseline for Few-shot Fine-Grained Visual Recognition.

## Datasets

You must first specify the value of `data_path` in `config.yml`. This should be the absolute path of the folder where you plan to store all the data.

We follow [FRN](https://github.com/Tsingularity/FRN) setting to use the same data settings for training.

## Training scripts

CUB cropped/CUB

```
python train.py \
    --opt sgd \
    --lr 1e-3 \
    --gamma 1e-1 \
    --epoch 150 \
    --decay_epoch 70 120 \
    --val_epoch 20 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 15 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 1 5 \
    --resnet \
    --gpu 0 
```

FGVC-Aircraft

```
ython train.py \
    --opt sgd \
    --lr 1e-2 \
    --gamma 1e-1 \
    --epoch 150 \
    --decay_epoch 70 120 \
    --val_epoch 5 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 15 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 1 5 \
    --resnet \
    --gpu 0 

```

Or Running the shell script `train.sh` will train and evaluate the model with hyperparameters matching our paper. Explanations for these hyperparameters can be found in `trainers/trainer.py`.

For the pretrained knowledges, you can find from **pretrained_knowledges** directly, and modified the ***pretrained_knowledge_path***  in train.py

After training concludes, test accuracy and 95% confidence interval are logged in the std output and `*.log` file. To re-evaluate a trained model, run `test.py`, setting the internal `model_path` variable to the saved model `*.pth` you want to evaluate.

## Contact

We have tried our best to verify the correctness of our released data, code. However, there are a large number of experiment settings, all of which have been extracted and reorganized from our original codebase. There may be some undetected bugs or errors in the current release. If you encounter any issues or have questions about using this code, please feel free to contact us via sunjieqi1017@163.com.

## Acknowledgment

Our project references the codes in the following repos.

- [FRN](https://github.com/Tsingularity/FRN)

