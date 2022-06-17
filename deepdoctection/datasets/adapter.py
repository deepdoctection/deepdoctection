
from typing import Union, Callable, Optional, Iterator

from torch.utils.data import IterableDataset

from ..utils.logger import logger
from ..utils.detection_types import  JsonDict
from ..utils.tqdm import get_tqdm
from ..utils.logger import log_once
from ..datasets.base import DatasetBase
from ..datapoint.image import Image
from ..dataflow import CustomDataFromList, MapData, RepeatedData
from ..mapper.maputils import LabelSummarizer

from .registry import get_dataset


class DatasetAdapter(IterableDataset):
    """
    A helper class derived from `torch.utils.data.IterableDataset` to process datasets within
    pytorch frameworks (e.g. Detectron2). It wraps the dataset and defines the compulsory
    `:meth:__iter__` using  meth:`dataflow.build` .

    DatasetAdapter is meant for training and will therefore produce an infinite number of datapoints
    by shuffling and restart iteration once the previous dataflow is exhausted.
    """
    def __init__(self, name_or_dataset: Union[str,DatasetBase], cache_dataset: bool,
                 image_to_framework_func: Optional[Callable[[Image],JsonDict]]=None, **build_kwargs ):
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

        if cache_dataset:
            logger.info("Yielding dataflow into memory and create torch dataset")
            categories = self.dataset.dataflow.categories.get_categories(as_dict=True,filtered=True)
            summarizer = LabelSummarizer(categories)
            df.reset_state()
            max_datapoints = build_kwargs.get("max_datapoints")

            if max_datapoints is not None:
                max_datapoints = int(max_datapoints)

            datapoints = []
            with get_tqdm(total=max_datapoints) as status_bar:
                for dp in df:
                    if dp.image is not None:
                        log_once(
                            "Datapoint have images as np arrays stored and they will be loaded into memory. "
                            "To avoid OOM set 'load_image'=False in dataflow build config. This will load "
                            "images when needed and reduce memory costs!!!",
                            "warn",
                        )
                    anns = dp.get_annotation()
                    cat_ids = [int(ann.category_id) for ann in anns]
                    summarizer.dump(cat_ids)
                    datapoints.append(dp)
                    status_bar.update()
            summarizer.print_summary_histogram()
            df = CustomDataFromList(datapoints,shuffle=True)
        df = MapData(df, image_to_framework_func)
        self.df = RepeatedData(df,-1)

        self.df.reset_state()

    def __iter__(self) -> Iterator[Image]:
        return iter(self.df)

    def __len__(self) -> int:
        return len(self.dataset)
