
from typing import Union, Callable, Optional, Iterator

from ..utils.logger import logger
from ..utils.detection_types import  JsonDict
from torch.utils.data import IterableDataset

from ..datasets.base import DatasetBase
from ..datapoint.image import Image
from ..dataflow import CacheData, CustomDataFromList, MapData
from .registry import get_dataset


class DatasetAdapter(IterableDataset):
    """
    A helper class derived from `torch.utils.data.IterableDataset` to process datasets within
    pytorch frameworks (e.g. Detectron2). It wraps the dataset and defines the compulsory
    `:meth:__iter__` using  meth:`dataflow.build` .
    """
    def __init__(self, name_or_dataset: Union[str,DatasetBase], cache_dataset: bool,
                 image_to_framework_func: Optional[Callable[[Image],JsonDict]]=None,**build_kwargs ):
        """

        :param name_or_dataset: Registered name of the dataset or an instance.
        :param cache_dataset: If set to true, it will cache the dataset (without loading images).
        :param image_to_framework_func: A mapping function that converts image datapoints into the framework format
        :param build_kwargs: optional parameters for defining the dataflow.
        """
        if isinstance(name_or_dataset,str):
            self.dataset = get_dataset(name_or_dataset)
        else:
            self.dataset = name_or_dataset
        df = self.dataset.dataflow.build(**build_kwargs)
        self.df = MapData(df, image_to_framework_func)
        if cache_dataset:
            logger.info("Loading dataset into memory")
            df_list = CacheData(self.df, shuffle = True).get_cache()
            self.df = CustomDataFromList(df_list)
        else:
            self.df.reset_state()

    def __iter__(self) -> Iterator[Image]:
        return iter(self.df)

    def __len__(self) -> int:
        return len(self.dataset)
