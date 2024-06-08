# Pharyngitis detection using deep learning CNN

Pharyngitis, commonly known as a sore throat, is a prevalent inflammatory condition affecting the throat. Detecting and diagnosing such common infections and diseases efficiently is crucial for timely treatment. In this repository, we explore the application of artificial intelligence techniques, specifically deep learning, to identify pharyngitis cases.

The dataset used in training this model is a public dataset obtained from *[Toward automated severe pharyngitis detection with smartphone camera using deep learning networks](https://data.mendeley.com/datasets/8ynyhnj2kz/2)*

## set-up
```bash
pip install -r requirements.txt
```
### download the model
you can download the model manually from the [release assets](https://github.com/guy-977/sore-throat-detection/releases/tag/model)  

or use the bash script
```bash
bash download_model.bash
```

## Training
```bash
python train.py
```

## Inference
```bash
python infer.py -i <path_to_image>
```

## Acknowledgement
### The Dataset
*[Toward automated severe pharyngitis detection with smartphone camera using deep learning networks](https://data.mendeley.com/datasets/8ynyhnj2kz/2)*
```text
Yoo, TaeKeun (2020), “Toward automated severe pharyngitis detection with smartphone camera using deep learning networks”, Mendeley Data, V2, doi: 10.17632/8ynyhnj2kz.2
```