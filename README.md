**MFMnet: Synergistic Drug Combination Prediction with Multidimensional Feature Fusion Methods and Attention Mechanism**

- `enhanced_model.py` contains the code for the MFMnet model, integrating multiple GNN types and adaptive fusion;

- `enhanced_main.py` is the main function for MFMnet, with ensemble learning capability;

- `model_analysis.py` is a script for analyzing and comparing model performance;

- `attention.py` contains the code for the attention mechanism;

- `muti_head_attention.py` contains the code for the multi-head attention mechanism;

- `transformer.py` contains the code for the transformer model;

- `loader.py` contains the code for data processing;


## Environmental Requirements

The MFMnet model is built on PyTorch and PyTorch Geometric. You can create a conda environment with relevant dependencies using the following commands:

```linux
conda create -n MDSyn python=3.11
conda activate MDSyn
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
conda install pyg -c pyg
conda install pandas
conda install rdkit
conda install scipy==1.11.4
conda install tqdm
conda install matplotlib seaborn scikit-learn
pip install scikit-optimize
```


### Compatibility Notes

The code has been modified to run on different versions of PyTorch and SciPy. Key compatibility fixes include:

1. Removed dependency on `torch.sym_int` to enable the code to run on older PyTorch versions
2. Added custom `_canonical_mask` and `_none_or_dtype` functions to replace functionalities in newer PyTorch versions
3. Replaced `scipy.interp` with `numpy.interp` to adapt to newer SciPy versions
4. Updated deprecated API calls, such as changing `torch_geometric.data.DataLoader` to `torch_geometric.loader.DataLoader`
5. Added device detection logic to allow the code to run in environments without CUDA support


## MFMnet Model Improvements

The MFMnet model has several significant improvements over the original model:

1. **Multiple Graph Neural Networks**: Integrates GCN, GIN, and GAT to capture different aspects of molecular interactions.

2. **Adaptive Feature Fusion**: Dynamically adjusts the weights of different features based on their importance.

3. **Enhanced Attention Mechanism**: Uses multi-head attention with positional encoding to better capture relationships between features.

4. **Specialized Feature Processing**: Modules specifically designed to handle drug embeddings and cell line gene expression data.

5. **Ensemble Learning**: Combines predictions from multiple model variants to improve overall performance.

6. **Optimized Classifier**: A deeper classification network with batch normalization and dropout to enhance generalization ability.


## Acknowledgments
The skeleton of the code is inherited from PyTorch and PyTorch Geometric.
