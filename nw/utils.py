# Adapted from https://github.com/alanqrwang/nwhead
# No license specified in original repository

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# import hnswlib
from sklearn.cluster import KMeans
from contextlib import contextmanager

@contextmanager
def manage_meta(dataset):
    if hasattr(dataset, 'get_meta'):
        dataset.set_get_meta(True)
        try:
            yield
        finally:
            dataset.set_get_meta(False)
    else:
        yield

class DatasetMetadata(Dataset):
    def __init__(self, dataset, metadata):
        super().__init__()
        self.dataset = dataset
        self.targets = self.dataset.targets
        self.metadata = metadata

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        if isinstance(datum, dict):
            return datum['image'], datum['class_idx'], self.metadata[idx]
        else:
            return datum[0], datum[1], self.metadata[idx]

class FeatureDataset(Dataset):
    def __init__(self, features, targets, metadata):
        super().__init__()
        self.features = features
        self.targets = targets
        self.metadata = metadata

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.metadata[idx]

class FullDataset(Dataset):
    """Dataset used by full evaluation."""
    def __init__(self, underlying_dataset, n_shot_full):
        super().__init__()
        self.underlying_dataset = underlying_dataset
        y_array = underlying_dataset.targets
        self.indices = get_separated_indices(y_array)

        # Ensures that classes are balanced
        min_length = min([len(l) for l in self.indices])
        n_shot_full = min(n_shot_full, min_length)

        self.keys = []
        for l in self.indices:
            self.keys += l[:n_shot_full]

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)

class RandomLoader(DataLoader):
    '''Use for regression tasks.'''
    def __init__(self, dataset, total_samples):
        self.dataset = dataset
        self.total_samples = total_samples
        super(RandomLoader, self).__init__(dataset)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.total_samples:
            raise StopIteration
        support_idxs = [self.i]
        return self.collate_fn([self.dataset[i] for i in support_idxs])

    def next(self):
        return self.__next__()

class InfiniteRandomLoader(DataLoader):
    def __init__(self,
                 dataset,
                 num_per_batch,
                 ):
        self.dataset = dataset
        self.num_per_batch = num_per_batch
        super(InfiniteRandomLoader, self).__init__(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        support_idxs = np.random.choice(len(self.dataset), size=self.num_per_batch, replace=False)
        return self.collate_fn([self.dataset[i] for i in support_idxs])

    def next(self):
        return self.__next__()

class InfiniteUniformClassLoader(DataLoader):
    def __init__(self,
                 dataset,
                 n_shot,
                 n_way=None,
                 ):
        self.dataset = dataset
        y_array = dataset.targets
        self.indices = get_separated_indices(y_array)
        self.n_classes = len(self.indices)
        self.n_shot = n_shot
        self.n_way = n_way
        if n_way:
            assert n_way <= len(self.indices)

        super(InfiniteUniformClassLoader, self).__init__(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def next(self, qy=None):
        if self.n_way:
            assert len(qy) <= self.n_way, "qy must be smaller than n_way"
            qy = qy.cpu().detach().numpy()
            probs = np.ones(len(self.indices))
            probs[qy] = 0
            probs /= probs.sum()
            subclasses = np.random.choice(self.n_classes, size=(self.n_way-len(qy)), replace=False, p=probs)

            subclasses = np.concatenate([subclasses, qy])
            indices = [self.indices[i] for i in subclasses] 
        else:
            indices = self.indices

        support_idxs = np.array([np.random.choice(
            row, size=self.n_shot, replace=False) for row in indices]).flatten()

        # Get support data from dataset and collate into mini-batch
        return self.collate_fn([self.dataset[i] for i in support_idxs])

def get_separated_indices(vals):
    '''
    Separates a list of values into a list of lists,
    where each list is the indices of a fixed label/attribute.

    Maps labels/attributes to natural numbers, if needed.
    
    E.g. [0, 1, 1, 2, 3] -> [[0], [1, 2], [3], [4]]
    '''
    if torch.is_tensor(vals):
        vals = vals.cpu().detach().numpy()
    num_unique_vals = len(np.unique(vals))
    # Map (potentially non-consecutive) labels to (consecutive) natural numbers
    d = dict([(y,x) for x,y in enumerate(sorted(set(vals)))])
    indices = [[] for _ in range(num_unique_vals)]
    for i, c in enumerate(vals):
        indices[d[c]].append(i)
    return indices

def get_multilabel_class_indices(targets):
    """
    Build one image-index pool for every class in a multi-label dataset.

    Args:
        targets: Tensor or array with shape [num_images, num_classes].
                 Positive labels must have values greater than zero.

    Returns:
        A list containing one list of image indices per class.
    """
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    if targets.ndim != 2:
        raise ValueError(
            "Multi-label targets must have shape "
            f"[num_images, num_classes], got {tuple(targets.shape)}"
        )

    class_indices = []

    for class_idx in range(targets.shape[1]):
        indices = torch.where(targets[:, class_idx] > 0)[0].tolist()

        if not indices:
            raise ValueError(
                f"Class {class_idx} has no positive samples"
            )

        class_indices.append(indices)

    return class_indices

def linear_normalization(arr, new_range=(0, 1)):
    """Linearly normalizes a batch of images into new_range
    arr: (batch_size, n_ch, l, w)
    """
    bs, nch, _, _ = arr.shape
    flat_arr = torch.flatten(arr, start_dim=2, end_dim=3)
    max_per_batch, _ = torch.max(flat_arr, dim=2, keepdim=True) 
    min_per_batch, _ = torch.min(flat_arr, dim=2, keepdim=True) 

    # Handling the edge case of image of all 0's
    max_per_batch[max_per_batch==0] = 1 

    max_per_batch = max_per_batch.view(bs, nch, 1, 1)
    min_per_batch = min_per_batch.view(bs, nch, 1, 1)

    return (arr - min_per_batch) * (new_range[1]-new_range[0]) / (max_per_batch - min_per_batch) + new_range[0]

class InfiniteUniformMultiLabelClassLoader(DataLoader):
    """
    Samples n_shot positive images for each selected class.

    An image may be sampled for multiple classes, but every sampled support
    is assigned one integer class identity for the NW head.
    """

    def __init__(self, dataset, n_shot, n_way=None):
        self.dataset = dataset
        self.n_shot = n_shot
        self.n_way = n_way

        targets = torch.as_tensor(dataset.targets)
        self.class_indices = get_multilabel_class_indices(targets)
        self.n_classes = len(self.class_indices)

        if n_shot < 1:
            raise ValueError("n_shot must be at least 1")

        if n_way is not None:
            if n_way < 1 or n_way > self.n_classes:
                raise ValueError(
                    f"n_way must be between 1 and {self.n_classes}"
                )

        for class_idx, indices in enumerate(self.class_indices):
            if len(indices) < n_shot:
                raise ValueError(
                    f"Class {class_idx} has only {len(indices)} "
                    f"samples, but n_shot={n_shot}"
                )

        super().__init__(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _select_classes(self, query_targets=None):
        if self.n_way is None:
            return np.arange(self.n_classes)

        if query_targets is None:
            return np.random.choice(
                self.n_classes,
                size=self.n_way,
                replace=False,
            )

        query_targets = torch.as_tensor(query_targets)

        if query_targets.ndim == 1:
            query_targets = query_targets.unsqueeze(0)

        if query_targets.ndim != 2:
            raise ValueError(
                "Query targets must have shape [batch, classes]"
            )

        required_classes = torch.where(
            query_targets > 0
        )[1].unique().cpu().numpy()

        if len(required_classes) > self.n_way:
            raise ValueError(
                f"Query batch contains {len(required_classes)} positive "
                f"classes, but n_way={self.n_way}. Increase n_way or "
                "reduce the query batch size."
            )

        remaining_classes = np.setdiff1d(
            np.arange(self.n_classes),
            required_classes,
        )

        additional_count = self.n_way - len(required_classes)

        if additional_count > 0:
            additional_classes = np.random.choice(
                remaining_classes,
                size=additional_count,
                replace=False,
            )
            selected_classes = np.concatenate(
                [required_classes, additional_classes]
            )
        else:
            selected_classes = required_classes

        return np.sort(selected_classes)

    def next(self, query_targets=None):
        selected_classes = self._select_classes(query_targets)

        support_images = []
        support_class_ids = []

        for class_idx in selected_classes:
            selected_indices = np.random.choice(
                self.class_indices[int(class_idx)],
                size=self.n_shot,
                replace=False,
            )

            for image_idx in selected_indices:
                sample = self.dataset[int(image_idx)]
                support_images.append(sample[0])
                support_class_ids.append(int(class_idx))

        support_images = self.collate_fn(support_images)
        support_class_ids = torch.tensor(
            support_class_ids,
            dtype=torch.long,
        )

        # SupportSetTrain currently expects a third metadata output.
        metadata = torch.zeros(
            len(support_class_ids),
            dtype=torch.long,
        )

        return support_images, support_class_ids, metadata

class KNN:
    '''KNN.'''
    def __init__(self, data, labels, n_neighbors=20) -> None:
        self.data = data
        self.labels = labels
        self.n_neighbors = n_neighbors
    
    def __call__(self, x):
        '''Query for nearest neighbors'''
        distances = -torch.cdist(x, self.data.to(x.device))
        indices = torch.argsort(distances, dim=-1, descending=True).cpu().detach().numpy()
        indices = indices[:, :self.n_neighbors]

        data = torch.cat([self.data[ind] for ind in indices], dim=0)
        labels = torch.cat([self.labels[ind] for ind in indices], dim=0)
        return data, labels

# class HNSW:
#     '''HNSW index for fast approximate nearest neighbor search.'''
#     def __init__(self, data, labels, n_neighbors=20) -> None:
#         self.data = data
#         self.labels = labels
#         self.n_neighbors = n_neighbors
#         # Create an HNSW index
#         num_elements, self.dim = data.shape
#         self.index = hnswlib.Index(space='l2', dim=self.dim)

#         # Initialize the index and add data points
#         self.index.init_index(max_elements=num_elements, ef_construction=100, M=16)
#         self.index.add_items(data)
    
#     def __call__(self, x):
#         '''Query for nearest neighbors'''
#         x = x.cpu().detach().numpy()
#         indices, _ = self.index.knn_query(x, k=self.n_neighbors)
#         indices = indices.astype(np.int64)
#         data = torch.cat([self.data[ind] for ind in indices], dim=0)
#         labels = torch.cat([self.labels[ind] for ind in indices], dim=0)
#         return data, labels

def compute_clusters(embeddings, labels, n_clusters, closest=True):
    '''Performs k-means clustering to find support set.
    
    :param closest: If True, uses support features closest to cluster centroids. Otherwise,
                uses true cluster centroids.
    '''
    img_ids = np.arange(len(embeddings))
    sfeat = []
    slabel = []
    sindices = []
    for c in np.unique(labels):
        embeddings_class = embeddings[labels==c]
        img_ids_class = img_ids[labels==c]
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(embeddings_class)
        centroids = torch.tensor(kmeans.cluster_centers_).float()
        slabel += [c] * n_clusters 
        if closest:
            dist_matrix = torch.cdist(centroids, embeddings_class)
            min_indices = dist_matrix.argmin(dim=-1)
            dataset_indices = img_ids_class[min_indices]
            if n_clusters == 1:
                dataset_indices = [dataset_indices]
            closest_embedding = embeddings[dataset_indices]
            sfeat.append(closest_embedding)
            sindices = sindices + dataset_indices.tolist()
        else:
            sfeat.append(centroids)
    sfeat = torch.cat(sfeat, dim=0)
    slabel = torch.tensor(slabel)
    sindices = torch.tensor(sindices)
    return sfeat, slabel, sindices

def compute_multilabel_clusters(
    embeddings,
    targets,
    n_clusters,
    closest=True,
):
    """
    Compute class-specific prototypes for multi-label targets.

    Args:
        embeddings: Feature tensor shaped [num_images, feature_dim].
        targets: Multi-hot tensor shaped [num_images, num_classes].
        n_clusters: Number of prototypes per class.
        closest: Select real samples nearest to centroids when True.

    Returns:
        support_features: [num_classes * n_clusters, feature_dim]
        support_labels: [num_classes * n_clusters]
        support_indices: Original dataset indices of selected images
    """
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must have shape [N, D], got "
            f"{tuple(embeddings.shape)}"
        )

    if targets.ndim != 2:
        raise ValueError(
            f"Targets must have shape [N, C], got "
            f"{tuple(targets.shape)}"
        )

    if len(embeddings) != len(targets):
        raise ValueError(
            "Embeddings and targets must contain the same "
            "number of samples"
        )

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")

    if not closest:
        raise NotImplementedError(
            "SIC explanations require real support images; "
            "use closest=True"
        )

    embeddings = embeddings.detach().cpu()
    targets = targets.detach().cpu()

    support_features = []
    support_labels = []
    support_indices = []

    for class_idx in range(targets.shape[1]):
        class_mask = targets[:, class_idx] > 0
        class_indices = torch.where(class_mask)[0]
        class_embeddings = embeddings[class_mask]

        if len(class_embeddings) < n_clusters:
            raise ValueError(
                f"Class {class_idx} has only "
                f"{len(class_embeddings)} positive samples, "
                f"but n_clusters={n_clusters}"
            )

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=0,
        ).fit(class_embeddings.numpy())

        centroids = torch.as_tensor(
            kmeans.cluster_centers_,
            dtype=embeddings.dtype,
        )

        distances = torch.cdist(
            centroids,
            class_embeddings,
        )

        nearest_local_indices = distances.argmin(dim=1)
        selected_dataset_indices = class_indices[
            nearest_local_indices
        ]

        support_features.append(
            embeddings[selected_dataset_indices]
        )
        support_labels.extend(
            [class_idx] * n_clusters
        )
        support_indices.extend(
            selected_dataset_indices.tolist()
        )

    return (
        torch.cat(support_features, dim=0),
        torch.tensor(support_labels, dtype=torch.long),
        torch.tensor(support_indices, dtype=torch.long),
    )
