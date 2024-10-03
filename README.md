<h1>LTGAN</h1>
<p>LTGAN: A Generative Adversarial Network Framework for Imputation and Analysis of scRNA-seq Data. This code is used for imputed scRNA-seq data.</p>
<p>The specific installation steps are as follows：</p>
<pre><code class='language-shell' lang='shell'>conda env create -f LTGAN_env.yaml
conda activate LTGAN_env
</code></pre>
<p>Runing：</p>
<pre><code class='language-python' lang='python'>python LTGAN.py --impute --n_epochs 500 --batch_size 64 --file_d ./Data/GSM5768747_NB5_UMI_COUNTS_RAW.csv --file_c louvain_labels_GSM5768752.csv --job_name GSM5768752 --outdir ./outputLTGAN
</code></pre>
<p>Among them, the sample data files stored in the Data folder, the &quot;xxx_labels.csv&quot; file represents the clustering label file of the original data.</p>
