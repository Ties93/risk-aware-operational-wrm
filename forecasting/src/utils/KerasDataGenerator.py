import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """
    Standard data generator for Keras models. 
    Makes no changes to the sample rate of individual samples.
    """
    def __init__(
        self,
        X,
        y,
        batch_size,
        shuffle=True,
    ):
        self.X=X
        self.y=y
        self.batch_size=batch_size
        self.shuffle=shuffle,
        self.indices=np.arange(self.X.shape[0])

    def __len__(self):
        """Denotes the number of batchs per epoch"""
        return int(np.floor(self.X.shape[0])/self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data based on an index"""
        id_0 = self.batch_size*index
        id_1 = min(self.batch_size*(index+1), self.X.shape[0])
        inds = self.indices[id_0:id_1]
        return self.X[inds].astype('float32'), self.y[inds].astype('float32')
        
    def on_epoch_end(self):
        """Updates indices"""
        self.indices=np.arange(self.X.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

class UniformDataGenerator(tf.keras.utils.Sequence):
    """
    Inverse Histogram Sampling (IHS) for Keras models. Samples uniformly from the training set respecting labels and weights.
    Takes a dataset (X, y), a timeseries of labels describing which labels each sample of the dataset belongs to.
    """
    def __init__(
        self,
        X,
        y,
        labels,
        batch_size,
        shuffle=True,
        samples_per_epoch=1000
    ):
        self.X=X
        self.y=y
        self.labels=labels
        self.label_names=np.unique(self.labels)

        self.batch_size=batch_size
        self.shuffle=shuffle,
        self.samples_per_epoch=samples_per_epoch

        self.indices=np.arange(self.X.shape[0])
        self.on_epoch_end()
        
        # The probability a certain label is sampled, start with uniform distribution.
        # Can be changed after intilializing the class
        self.probabilities=[1/len(self.label_names) for _ in range(len(self.label_names))]
        
        self.n_samples_per_label={l: len(self.grouped_indices[l]) for l in self.label_names}
        self.update_remove_batch_from_epoch()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.samples_per_epoch/self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data. Index argument is included for compatibility, but not necessary since samples are generated randomly from the available data."""
        # Generate the labels to sample for this batch.
        label_index = np.random.choice(self.label_names, p=self.probabilities, size=(self.batch_size), replace=True)

        # Calculate amount of samples per label in this batch
        N_samples = {l: len(label_index[label_index==l]) for l in self.label_names}

        # Generate the actual indices of the samples, concatenate the indices per label and shuffle!
        batch_indices = np.concatenate(
            [self.generate_batch_per_label(
                label=l,
                N_samples=N_samples[l]
            ) for l in self.label_names],
            axis=0
        ).tolist()
        np.random.shuffle(batch_indices)
        # print(self.X)
        # print(self.y)
        # Return the data on the generated indices
        return self.X[batch_indices].astype('float32'), self.y[batch_indices].astype('float32')

    def generate_batch_per_label(self, label, N_samples):
        """Generates sample indices for one batch per label"""
        if self.remove_batch_from_epoch[label]:
            # If N samples per epoch >= total amount of samples in the label * a safety factor for crashing, you can remove the batch from the dataset.
            # Sampling without replacement.
            batch_indices = self.grouped_indices[label][:N_samples]
            self.grouped_indices[label] = self.grouped_indices[label][N_samples:]
            return batch_indices

        # Else we randomly pick samples from the dataset per batch with replacement
        return np.random.choice(self.grouped_indices[label], size=N_samples, replace=True)

    def update_remove_batch_from_epoch(self, safety_factor=1.05):
        n_labels = len(self.label_names)
        generated_samples_per_label = self.samples_per_epoch / n_labels * safety_factor
        self.remove_batch_from_epoch = {l: True if self.n_samples_per_label[l] >= generated_samples_per_label else False for l in self.label_names}

    def on_epoch_end(self):
        'Updates indices. Shuffles the indices per label category.'
        self.grouped_indices={l: np.array([self.indices[i] for i in self.indices if self.labels[i]==l]) for l in self.label_names}
        if self.shuffle:
            for l in np.unique(self.labels):
                np.random.shuffle(self.grouped_indices[l])
