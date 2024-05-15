from utils import LangCellTranscriptomeTokenizer
import scanpy as sc

data = sc.read_h5ad('/path/to/adata.h5ad')
data.obs['n_counts'] = data.X.sum(axis=1)
data.var['ensembl_id'] = data.var['feature_id']

tk = LangCellTranscriptomeTokenizer(dict([(k, k) for k in data.obs.keys()]), nproc=4)
tokenized_cells, cell_metadata = tk.tokenize_anndata(data)
tokenized_dataset = tk.create_dataset(tokenized_cells, cell_metadata)

tokenized_dataset.save_to_disk('/path/to/tokenized_dataset')