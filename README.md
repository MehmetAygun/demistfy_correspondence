# Demistfying Unsupervised Semantic Correspondence Estimation

<img width="720" alt="main" src="https://user-images.githubusercontent.com/5329637/176780938-488143a5-7d80-4010-b8b1-a341bb156607.svg?token=GHSAT0AAAAAABNA2V6NR2G27FQDESH3IRFUYUOHABQ">

### Installation

```bash
git clone https://github.com/MehmetAygun/demistfy_correspondence
cd demistfy_correspondence
conda env create -f environment.yml
conda activate demistfy
```

### Data

You can download the datasets that we consider in the paper from respective links: <a href="http://cvlab.postech.ac.kr/research/SPair-71k/">Spair</a>, <a href="http://www.vision.caltech.edu/datasets/cub_200_2011/">CUB</a>, <a href="https://github.com/benjiebob/StanfordExtra">Stanforddogs</a>, <a href="https://github.com/prinik/AwA-Pose">Awa-Pose</a>, <a href="https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/">AFLW</a>. 

For evaluation, you need copy put random_pair.txt file from pairs directory to respective dataset folder. Final folder structure should be like below;

```bash
data_folder/dataset
└── .....
└── random_pair.txt
```

### Evaluation

To evaluate a method, you need to run evaluate.py from respective dataset folder in the evaluation directory. 

For instance to evaluate None projection (original embeddings) 
```bash
python evaluate.py --layer 3 --logpath None
```

This will create a log file under logs directory, and in the end of the file you can find the metrics that we described in the paper. 

To evaluate a finetuned unsupervised method you need to give use --projpath argument, and if you want to test another type of architecture like transofmer you need to give --model_type argument.

model_types are: resnet50, dino, vit

For instance to evalute asym using unsupervised transformer as a backbone for the CUB dataset, you should go to evalution/cub directory and run :

```bash
python evaluate.py --model_type dino --layer 3 --projpath path_to_cub_asym_projection --logpath dino_asym_cub
```

### Training

To train a projection layer using unsupervised losses (eq,dve,lead,cl,asym) go to projection directory and run train.py script.
You need to give datapath with --datapath argument, dataset with --dataset which can be : spair, stanforddogs, cub, awa, aflw
To use unsupervised losses give as single argument without any parameter: for example --eq for training projection with EQ loss(other options are cl, asym, dve, lead).

If you do not give any parameter for unsupervised loses, the scripts trains projection with keypoint supervision.

You can also set batchsize, layers, weightdecay and model_type or initial weights for the backbone with model_weights argument. 

For instance, to train a projection layer using ASYM loss on top of supervised CNN for the Stanforddogs dataset:

```bash
python train.py  --layer 3 --batchsize 8 --asym --dataset stanforddogs --datapath path_to_stanforddogs_dataset --logpath path_for_log
```

### Trained Models

You can download the pretrained models from <a href="https://drive.google.com/file/d/1bC54tNCe6gdW2x-OsSupFCzSwSG9sRE6/view?usp=sharing">here</a> 

### New PCK Metric and Detailed Analysis

<img width="720" alt="errors" src="https://user-images.githubusercontent.com/5329637/176784273-bb8edb0f-a84d-4cf1-8254-fef130fe8f09.png">



If you want to use our proposed version of PCK and detailed analysis the function is implemented <a href="https://github.com/MehmetAygun/demistfy_correspondence/blob/main/evaluation/model/evaluation.py#L162 ">here</a>.


### Citing
If you find the code useful in your research, please consider citing:

    @inproceedings{aygun2022eccv,
        author    = {Aygun, Mehmet and Mac Aodha, Oisin},
        title     = {Demystifying Unsupervised Semantic Correspondence Estimation},
        booktitle = {European Conference on Computer Vision (ECCV)},
        year      = {2022}
    }
  
### Acknowledgements

We used partial code snippets from <a href="https://github.com/juhongm999/hpf">hpf</a>, <a href="https://github.com/isl-org/MiDaS">MiDaS</a> and <a href="https://github.com/ShirAmir/dino-vit-features">dino-vit-features</a>.
