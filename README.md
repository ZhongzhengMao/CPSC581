# CPSC581
This README provides step-by-step instructions to reproduce the results obtained from running the code.

## Description
Before running the code, ensure that you have the following dependencies installed:

- Python (version 3.6 or above)
- PyTorch (version 1.9.0)
- Torchvision (version 0.10.0)
- Matplotlib
- Scikit-learn

It is strongly advised to install or update the PyTorch library and the torchvision package as specified below and in the accompanying ipynb to avoid potential errors when loading the MNIST dataset.
```
!pip install --upgrade torch==1.9.0
!pip install --upgrade torchvision==0.10.0
```

The code will perform the following steps:
   - Load and preprocess the MNIST dataset.
   - Define the CNN architecture.
   - Train the CNN model using different optimizers (Adam, SGD, RMSprop) for a specified number of epochs.
   - Evaluate the model on the test set and record the accuracies for each optimizer.
   - Plot the training and validation losses for each optimizer.
   - Plot the accuracy curves for each optimizer over the epochs.

During the execution, you will see the training progress printed in the console, including the epoch number, batch number, training loss, and validation loss.

After the training is complete, the code will display the accuracies achieved by each optimizer on the test set.

Two plots will be generated and displayed:
   - "PyTorch_CNN_Loss" plot: Shows the training and validation losses for each optimizer over the batches.
   - "PyTorch_CNN_Accuracy" plot: Shows the accuracy curves for each optimizer over the epochs.

Additionally, the code will run the Random Forest and SVM classifiers from scikit-learn on the MNIST dataset and display their accuracies.

The entire execution process may take some time, depending on your machine's specifications and the number of epochs specified.

Once the execution is complete, you can review the printed accuracies and the generated plots to analyze the performance of different optimizers and compare them with the scikit-learn classifiers.

That's it! By following these steps, you should be able to reproduce the results obtained from running the code.

Please refer to the code provided in the appendix or the ipynb to replicate the results.

## Appendix
```

```
