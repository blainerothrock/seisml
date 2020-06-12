# Triggered Tremor
 Experiments using the Triggered Tremor dataset.
  * [ConvNet](convnet/README.md)
  * [Deep Clustering](deep_clustering/README.md)
   
 ## Dataset `seisml/datasets/TriggeredTremor`

 ### Origin Details
 The seismic data for this project was provided by a number of different sources. [The Broadband Array in Taiwan for Seismology](https://doi.org/10.7914/SN/TW) and the [Central Weather Bureau of Taiwan](https://gdms.cwb.gov.tw) provided seismograms containing tremor that was dynamically triggered by seismic surface waves from one of six large earthquakes. The earthquakes are:
 
| Manitude       | Location                             | Datetime     |
| :-----: | :----------------------------------------- | :----------------------------: |
|  M7.8   | Southern Qinghai, China                     | `2001-11-14 09:26:10 (UTC)`   |
|  M8.2   | Hokkaido, Japan region                      | `2003-09-25 19:50:06 (UTC)`   |
|  M9.1   | 2004 Sumatra - Andaman Islands Earthquake   | `2004-12-26 00:58:53 (UTC)`   |
|  M8.6   | Northern Sumatra, Indonesia                 | `2005-03-28 16:09:36 (UTC)`   |
|  M8.4   | Northern Sumatra, Indonesia                 | `2007-09-12 11:10:26 (UTC)`   |
|  M8.2   | Hokkaido, Japan region                      | `2003-09-25 19:50:06 (UTC)`   |
|  M9.1   | 2011 Great Tohoku Earthquake, Japan         | `2011-03-11 05:46:24 (UTC)`   |


[The High Sensitivity Seismograph Network Japan](https://www.hinet.bosai.go.jp) provided seismograms containing tremor that was dynamically triggered by seismic surface waves from the following earthquake.

| Manitude       | Location                             | Datetime     |
| :-----: | :----------------------------------------- | :----------------------------: |
|  M8.6   | Off the west coast of northern Sumatra      | `2012-04-11 08:38:36 (UTC)`   |


The positive identifications of dynamically triggered tremor were made by Kevin Chao and Vivian Tang. These seismograms are called "positive examples". All positive examples begin `D/4`s after the earthquake's origin time, where `D` is the distance in km from the epicenter to the recording station, and they end `1000`s later.
Negative examples consist of seismograms of the same duration (`1000`s) recorded on `2011-03-04`, a day without large earthquakes by stations from the following networks:

* [Broadband Array in Taiwan for Seismology](https://doi.org/10.7914/SN/TW)
* [Australian National Seismograph Network](http://www.ga.gov.au)
* Global Seismograph Network: [UI](https://doi.org/10.7914/SN/IU) & [II](https://doi.org/10.7914/SN/II)
* [Malaysian National Seismic Network](http://www.met.gov.my)

Each positive or negative example is one component of a three-component recording. Each example was band-pass filtered with corner frequencies of `2` and `5` Hz.

## Aggregated Data
Data is made public:
* [Raw Data](https://blainerothrock-public.s3.us-east-2.amazonaws.com/seisml/triggered_tremor/triggered_tremor_sample.tar.gz)
    * above pre-processing is already applied
    * Model-ready data can be reproduced by simply combining all files into a single directory and noting the label
        * no additonal pre-processing unless noted in the experiment readme.
* Model-ready datasets are detailed [internally](../../seisml/utility/download_data.py) in seisml
 
 