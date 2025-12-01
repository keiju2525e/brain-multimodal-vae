from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

class ImageBrainDataset(Dataset):
      def __init__(self, data: List[Tuple[int, Dict[str, Optional[int]]]]):
          self.data = data

      def __len__(self) -> int:
          return len(self.data)

      def __getitem__(self, idx: int) -> Tuple[int, Dict[str, Optional[int]]]:
          return self.data[idx]

def get_dataset(train_data, test_data):
    train_dataset = ImageBrainDataset(train_data)
    test_dataset = ImageBrainDataset(test_data)

    return train_dataset, test_dataset