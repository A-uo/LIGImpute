# LTGAN

LTGAN: A Generative Adversarial Network Framework for Imputation and Analysis of scRNA-seq Data. This code is used for imputed scRNA-seq data.

The specific installation steps are as follows：

```shell
conda env create -f LTGAN_env.yaml
conda activate LTGAN_env
```

Runing：

```python
python LTGAN.py --impute --n_epochs 500 --batch_size 64 --file_d ./Data/GSM5768747_NB5_UMI_COUNTS_RAW.csv --file_c louvain_labels_GSM5768752.csv --job_name GSM5768752 --outdir ./outputLTGAN
```

The sample data files are saved in Data.zip. You can run them by unzipping them in this directory. The other two .py files can also be run according to the above code. The "xxx_labels.csv" file represents the clustering label file of the original data.
