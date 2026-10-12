from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
import torch


def data_loader(
    train_inputs,
    val_inputs,
    train_labels,
    val_labels,
    test_inputs=None,
    test_labels=None,
    batch_size=50,
):
    train_inputs, val_inputs = (
        torch.tensor(train_inputs),
        torch.tensor(val_inputs),
    )
    train_labels, val_labels = (
        torch.tensor(train_labels),
        torch.tensor(val_labels),
    )

    train_data = TensorDataset(train_inputs, train_labels)
    val_data = TensorDataset(val_inputs, val_labels)

    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=batch_size,
    )

    val_dataloader = DataLoader(
        val_data,
        sampler=SequentialSampler(val_data),
        batch_size=batch_size,
    )

    test_dataloader = None

    if test_inputs is not None and test_labels is not None:
        test_inputs = torch.tensor(test_inputs)
        test_labels = torch.tensor(test_labels)

        test_data = TensorDataset(test_inputs, test_labels)
        test_dataloader = DataLoader(
            test_data,
            sampler=SequentialSampler(test_data),
            batch_size=batch_size,
        )

    return train_dataloader, val_dataloader, test_dataloader