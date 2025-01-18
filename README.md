# LIGImpute

scMASKGAN: A Generative Adversarial Network Framework for Imputation and Analysis of scRNA-seq Data. This code is used for imputed scRNA-seq data.

The specific installation steps are as follows：

```shell
conda conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install matplotlib
conda install pandas
conda install numpy
conda install scikit-learn
```

Runing：

```python
python scMASKGAN.py --impute --n_epochs 500 --batch_size 64 --file_d ./Data/GSM5768752_NB5_UMI_COUNTS_RAW.csv --file_c louvain_labels_GSM5768752.csv --job_name GSM5768752 --outdir ./outputscMASKGAN
```

The sample data files are saved in Data.zip. You can run them by unzipping them in this directory. The other two .py files can also be run according to the above code. The "xxx_labels.csv" file represents the clustering label file of the original data.
