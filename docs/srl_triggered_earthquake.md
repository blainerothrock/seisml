## Reproducing *Automating the Detection of Dynamically Triggered Earthquakes via a Deep Metric Learning Algorithm*

[paper](https://watermark.silverchair.com/srl-2019165.1.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAArcwggKzBgkqhkiG9w0BBwagggKkMIICoAIBADCCApkGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMFPamseNcqsTfNoawAgEQgIICahBp-9KiGZaVc4QlHJc3biExGeWYLGunxlFrGENwQw8VLKF_hWV4hU6aOjLhw_E_BvP5WtqXwJukAwjuIkqskINLCTLX4l-1bVfZErGhrAuE6VGnf2BHGbVpGNj3D34-bW8raakm2tYErSloyzkiG9Y62O0mxhuE_EbN-wKSX0TB-yLmeBAWq0wU0enb8MjBk_zIrlBDicTj7lGovWon1YF_2rKCy2Ad0UelHyuQpajOPe9CF6BZK6Bt6JjbxRxJSMyN0lzgBNOH-YZ0G98tXj_wvoqfcm7378oOS4exndyGiJVD0KZ_AhTYVHmT7v2FGtPm8fstotSYaLNYCtHXxwjItqSGiefqKqlL6HIznwnFOzJNPx_Py0jhRLxtHnPaxjepBsMveRyQPvcdbYz2n1yjpaeXKL71f5oeb6KtCw0ME2AWLPzwwUwqAyrwm2VNWMZbo3JOLHqs6BKbXn36NylcwW04s8UcFmLuoGeghah9FG_d6z9mLWYRZYPB0iYB5srlWj_eCJx9QLScnSfOJ0mcdlBh-BulzBxU20HCEOrvtuSnnvyWEWzRZLlIlbHCRnU7pn0Og_iXtu7lmnO-_xQrsA1qBJd3By6PnZnoHScS2J0ZftYk9LrqjT396gdtdJwgL9gUZkOo4GaxbjHMqCoLuoeofCY7f0oBj1JWBWRz-_wjKlhuJL4LbZBpqa0DGeZRAujEZYlxLiQt-HI8cDM_YDpwGerGaBzO6j_3hrO1BS6qZO-z-3ftr5k_45I8R-fflfEk_AmwbacWuPlKDxfI2SadF5Wtju3l9KoFthN29_W5i_Upzb0HLQ)


**NOTE**: this codebase was originally structured with the code used in this paper. These steps will change throughout
the development of this project.  

### Download Triggered Earthquake data
* Download the raw triggered earthquake data [here](https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_earthquakes.tar.gz) (3.87GB).
* Highly recommend to verify the download with [triggered_earthquakes.md5](../seisml/utility/checksums/triggered_earthquakes.md5)
```bash
md5sum -c triggered_earthquakes.md5

> triggered_earthquakes.tar.gz: OK
```

### Preparing the data
The data should be kept in a directory structure as follows:

```
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


* using the script `scripts/prepare_data.py` run the follow to preprocess the data for training:
```bash
prepare_data.py [-h] --output_directory OUTPUT_DIRECTORY
                       --dataset_directory DATASET_DIRECTORY
                       [--accepted_labels ACCEPTED_LABELS]
```

* `OUTPUT_DIRECTORY`: where prepared data will be stored (`triggered_earthquakes/prepared/`)
* `DATASET_DIRECTORY`: location of the raw data (`triggered_earthquakes/raw/`)
* `ACCEPTED_LABELS`: labels to select give the folder structure (`positive:negative`).

### Train the Model
* alter `pipeline/convolutional-siamese/train.sh` with a new runId and correct directory for the prepared data.
* run `tain.sh`:
```bash
cd pipeline/convolutional-siamese/
./train.sh
```

### Check Results
* A new folder in `convolutional-siamese/runs` should be created for the run id.
* Verify the confusion matrix in `convolutional-siamese/runs/{runId}/results.txt` is similar to the following with 
accuracy in the low 90%.
```
Predict  0        1        
Actual
0        35       0        

1        8        49 
```