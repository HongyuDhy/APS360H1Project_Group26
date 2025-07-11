Instruction:
1. In LoadData.py, change ImageFolder Path (in def get loader) to your own path of the dataset.
2. All custom networks (include baseline) are in CustomNetwork.py, all pretrained networks are in Network.py (for further comparison).
3. All the parameters that needs to be changed in COVID19Training.py (main file) are the statement start with parser.add argument.
 - the primary parameters are '--epochs', '--batch_size', '--lr', '-model', make sure the model name are in the dictionary.
4. Please check if your gpu/cuda is available for the network (type nvidia-smi in bash/terminal and check memory used after the training starts). It might be out of memory.
5. The parameters with highest validation accuracy will be saved, as well as the figure of training/validation loss/accuracy. 