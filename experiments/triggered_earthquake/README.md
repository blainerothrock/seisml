## Automating the Detection of Dynamically Triggered Earthquakes

Reproduction of the paper [Automating the Detection of Dynamically Triggered Earthquakes via a 
Deep Metric Learning Algorithm](https://watermark.silverchair.com/srl-2019165.1.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAtAwggLMBgkqhkiG9w0BBwagggK9MIICuQIBADCCArIGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMb-r10hLYmbWjD4I7AgEQgIICg1StPbWNfWdt0xCDI3gVk9dDH9B8lDfMftNEo_kmcXDsN03eknrmyudlR-YBoGEC4o1N-ICf8p8kMFYrzLdefLASjImKWiTT82VHOnUBon2yUVO_EXDn2yAAKuo-8L57xH2lzVr5r52f8E9U5PZJ7U5olc6PFZJcmZCTDRFHQi4-hYU-T6wLvXFiPJed6Bg3pfZPbms1nWctdioMjoNwRTSEmxtCQlzzfrGzGJVt5EkXTao1i6MkLrDyIRPC_mg8ieur9eCwsOzPCNP5ddf2uiTmFgMQgs82PFVia1isRlIWToJfEdlOKN3RuAR6ddGcCfkvfJJbspIAkbN4zku2dr89Rtk3Axezlee1IvbZuuGc-HZskUmVAEMsaRF5dWB3hquYWPiVX_TmiDsjsMlHKPZaxbR4D0eD5tEF7VLLYIobLWYMnxCT9czyYWsJTjeduxPVNjeej0hvI8EDXauzQRksS2t9Q-zGZ61BKR2LzJUBradYgWmEUqyRIry29bR74Fy-6ITxxyPzfmHFboayEZaMbu4K05O9rqcd309cjeYH2p5C6efvwLFyXV5pMNkZlrUwKtYuifj-Ki1hOyROBAKvBCB8xl2iFtdqKswFdOllcz1SMCK3je7zwvMJ4auYQ7_ws1lOmMEd5NNVC_Xn4StRBA6a2VLMWgfBd41fo6Xe9DSoGHkXb3WUu0VvXthUPx4RzdPJEtkXQC7itA8wdWRYF4OGXjoHZE-iE7fcXgcl-jcTtXTDq1JmYWDmF8IR9GzlebyK_zRB-A5ppt4ZD8ggB2HSOc9t_h0_45dxkcLmFsP0gJeFscPdZnxLrTg7iinevcVIgCEteHvUQS77w3Eq-kI)

## Files
* `config.gin`: all parameters of the experiment, set to match the paper.
    * `train.batch_size`
    * `train.num_workers`
    * `train.learning_rate`
    * `train.weight_decay`
    * `train.epochs`
    * `triggered_earthquake_transform.sampling_rate`
    * `triggered_earthquake_transform.max_freq`
    * `triggered_earthquake_transform.min_freq`
    * `triggered_earthquake_transform.corner`
    * `triggered_earthquake_transform.aug_prob`
    * `triggered_earthquake_transform.target_length`
    * `triggered_earthquake_transform.random_trim_offset`
    * `triggered_earthquake_dataset.data_dir`
        * default dir `~/.seisml/data/triggered_earthquake`
        * specifiying a new directory will download the data to that location
    * `triggered_earthquake_dataset.force_download`
    * `triggered_earthquake_dataset.label`
        * classes determined by the fild structure (see below)
    * `triggered_earthquake_dataset.testing_quakes`
        * (*Array*): determines the earthquake(s) used for testing.
        * The paper runs trains a model and reports results alternating through all 8 earthquakes as a test set
* `train.py`
    * main file to run, will train based on specification in `gin.config` and report training results.
    * Training results are reported to Tensorboard (see below)
    * **TODO**: save model
* **TODO**: inference
* `triggered_earthquake_engine.py`
    * helper methods for Pytorch Ignite supporing `train.py`
        
## Dataset
The dataset for this experiement is built into seisml and contains a specific directory format.
All seismic samples are single channel obspy Streams and all transform are performed on a Trace.
* Folder structure for the experiement (dataset data is already configured)
```text
data_folder/
    earthquake1/
        positive/
            file1.sac
            file2.sac
            ...
        negative/
            file1.sac
            file2.sac
            ...
        more_optional_labels_like_chaos/
            more_files.sac
            ...
    earthquake2/
        positive/
            file1.sac
            file2.sac
            ...
        negative/
            file1.sac
            file2.sac
            ...
        more_optional_labels_like_chaos/
            more_files.sac
            ...
    ...
```

## To Run
* clone and create Anaconda invironment (see root [readme](../../README.md))
* configure `config.gin` -- current version matches paper parameters
* train the model: 
```shell script
python train.py
```
* view training results in Tensorboard, in a separate terminal instance while training, run:
```shell script
tensorboard --logdir runs
```
* **TODO**: results will be posted ...
* **TODO**: saved model will be posted ...
* **TODO**: inference on new data ...

        
        
   
   