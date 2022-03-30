# Frequency Dynamic Convolution-Recurrent Neural Network (FDY-CRNN) for Sound Event Detection


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/frequency-dynamic-convolution-frequency/sound-event-detection-on-desed)](https://paperswithcode.com/sota/sound-event-detection-on-desed?p=frequency-dynamic-convolution-frequency)

Official implementation of <br>
 - **Frequency Dynamic Convolution: Frequency-Adaptive Pattern Recognition for Sound Event Detection** (Submitted to INTERSPEECH 2022) <br>
by Hyeonuk Nam, Seong-Hu Kim, Byeong-Yun Ko, Yong-Hwa Park <br>[![arXiv](https://img.shields.io/badge/arXiv-2203.15296-brightgreen)](https://arxiv.org/abs/2203.15296)<br>



Currently, only model achitecture for FDY-CRNN is available in this repo. Whole code implementation for training SED with FDY-CRNN will be uploaded soon (before mid-April).

## Frequency Dynamic Convolution

<img src=./utils/fig2.png align="left" height="332" width="741"> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br>

Frequency Dynamic Convolution applied kernel that adapts to each freqeuncy bin of input, in order to remove tranlation-invariance of 2D convolution along the frequency axis.
- Traditional 2D convolution enforces translation-invaraince on time-frequency domain audio data in both time and frequency axis.
- However, sound events exhibit time-frequency patterns that are translation-invariant in time axis but not in frequency axis.
- Thus, frequency dynamic convolution is proposed to remove physical inconsistency of traditional 2D convolution on sound event detection.

<img src=./utils/fig3.jpg align="left" height="270" width="395"> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br>

Above bar chart compares the class-wise event-based F1 scores for the CRNN model with and without freqeuncy dynamic convolution. It can be observed that
- Quasi-stationary sound events such as blender, frying and vacuum cleaner exhibits simple time-frequency patterns (example shown below (a), a log mel spectrogram of vacuum cleaner sound). Thus, frequency dynamic convolution is less effective on these sound events.
- Non-stationary sound events such as alarm/bell ringing, cat, dishes, dog, electric shaver/toothbrush, running water and speech exhibits intricate time-frequency patterns (example shown below (b), a log mel spectrogram of speech sound). Thus, frequency dynamic convolution is especially effective on these sound events.

<img src=./utils/fig4.jpg align="left" height="330" width="437"> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br> <br>

For more detailed explanations, refer to the paper mentioned above, or contact the author of this repo below.

## Requirements
Python version of 3.7.10 is used with following libraries
- pytorch==1.8.0
- pytorch-lightning==1.2.4
- pytorchaudio==0.8.0
- scipy==1.4.1
- pandas==1.1.3
- numpy==1.19.2


other requrements in [requirements.txt](./requirements.txt)


## Datasets
You can download datasets by reffering to [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) or [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task). Then, set the dataset directories in [config yaml files](./configs/) accordingly. You need DESED real datasets (weak/unlabeled in domain/validation/public eval) and DESED synthetic datasets (train/validation).

## Training (Currently unavailable)
You can train and save model in `exps` folder by running:
```shell
python main.py
```

#### Results of FDY-CRNN on DESED Real Validation dataset:

Model                   | PSDS1          | PSDS2          | Collar-based F1  | Intersection-based F1
------------------------|----------------|----------------|------------------|-----------------
w/o Dynamic Convolution | 0.416          | 0.640          | 51.8%            | 74.4%
DY-CRNN                 | 0.441          | 0.663          | 52.6%            | 75.0%
TDY-CRNN                | 0.415          | 0.652          | 51.2%            | 75.1%
FDY-CRNN                | **0.452**      | **0.670**      | **53.3%**        | **75.3%**

   - These results are based on max values of each metric for 16 separate runs on each setting (refer to paper for details).

## Reference
[DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task) <br>
[SED with FilterAugment](https://github.com/frednam93/FilterAugSED)

## Citation & Contact
If this repository helped your works, please cite papers below!
```bib
@article{nam2022frequency,
      title={Frequency Dynamic Convolution: Frequency-Adaptive Pattern Recognition for Sound Event Detection}, 
      author={Hyeonuk Nam and Seong-Hu Kim and Byeong-Yun Ko and Yong-Hwa Park},
      journal={arXiv preprint arXiv:2203.15296},
      year={2022},
}

```
Please contact Hyeonuk Nam at frednam@kaist.ac.kr for any query.
