import torch

def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)

def unpack_data(dataB, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
        else:
            raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))
    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))