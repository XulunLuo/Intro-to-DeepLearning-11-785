- **Model**: 7-layer Diamond MLP architecture with layers [2268→2176→2176→2176→1536→1024→512→42]. 
  Uses GELU activations, BatchNorm1d after each hidden layer (except first), and Dropout(0.1). 
  Context window of 40 frames (81 total frames including center). ~19.89M trainable parameters.

- **Training Strategy**: AdamW optimizer (lr=0.001, weight_decay=0.001) with OneCycleLR scheduler 
  over 45 epochs. CrossEntropyLoss criterion. Mixed precision training (AMP) with GradScaler. 
  Gradient clipping (max_norm=1.0). Batch size 1024. Xavier normal weight initialization.

- **Augmentations**: SpecAugment-style augmentation using torchaudio transforms:
  FrequencyMasking (freq_mask_param=6) and TimeMasking (time_mask_param=20), both applied during training.

- **Notebook Execution**: Run cells sequentially from top to bottom. Requires PyTorch with CUDA, 
  torchaudio, wandb, and standard data science libraries (numpy, pandas, tqdm). 
  Data should be placed in the ROOT directory specified in the notebook.
  Test-Time Augmentation (TTA) with 5 passes is used for final predictions.