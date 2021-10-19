import argparse
from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMNist
from torchvision import transforms

from datamodule import BaseDataModule , load_and_orint_info

DOWNLOAD_DATA_DIRNAME = BaseDataModule.data_dirname() / "download"


class MNIST(BaseDataModule):
    def __init__(self,args:argparse.Namespace)->None:
        super().__init__(args)
        self.data_dirname = DOWNLOAD_DATA_DIRNAME
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize( (0.13,),(0.3,) ) ]  )
        self.dims = (1,28,28)
        self.output_dims = (1,)
        self.mapping=list(range(10))

    def prepare_data(self, *args, **kwargs) -> None:
        """
        Download train and test MNIST data from pythorch canonical source"""
        TorchMNist(self.data_dirname,train=True,download=True)
        TorchMNist(self.data_dirname,download = True , train=False)
    

    def setup(self , stage=None) -> None:
        """
        split into train , test , val and set dims"""

        mnist_full = TorchMNist(self.data_dirname , train = True , transform=self.transform)
        self.data_train , self.data_val = random_split(mnist_full , [55000,5000])
        self.dta_test = TorchMNist(self.data_dirname , train =False, transform = self.transform)



if __name__ == "__main__":
    load_and_orint_info(MNIST)