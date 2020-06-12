## Triggered Tremor Deep Clustering
Applying the methology from [Triggered Earthquakes](../../triggered_earthquake/README.md) to The Triggered Tremor dataset.

## Run
update `config.gin` with desired parameters
```shell script
python train.py
```
View results with Tensorboard
```shell script
tensorboard --logdir runs
```
Models will be stored in `./models` (ignored from source control)

## Results
While the model runs, the model is not converging. This method needs to be explored further for this dataset.