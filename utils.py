import numpy as np

def to_categorical(labels, num_classes):
    """
    Converts a 1D array of integer labels into a 2D one-hot encoded array.

    Parameters:
    - labels: A 1D NumPy array of integer labels.
    - num_classes: The total number of classes.

    Returns:
    - A 2D NumPy array where each row is a one-hot encoded vector.
    """
    one_hot = np.zeros((labels.shape[0], num_classes))
    
    one_hot[np.arange(labels.shape[0]), labels] = 1
    
    return one_hot