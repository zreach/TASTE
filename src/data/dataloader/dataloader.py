import math
import copy
from logging import getLogger

import torch

from src.data.interaction import Interaction, cat_interactions
from src.data.transform import construct_transform
from src.utils import InputType, FeatureType, FeatureSource, ModelType


class AbstractDataLoader(torch.utils.data.DataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        self.shuffle = shuffle
        self.config = config
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = self.step = self.model = None
        self._init_batch_size_and_step()
        index_sampler = None
        self.generator = torch.Generator()
        self.generator.manual_seed(config["seed"])
        self.transform = construct_transform(config)

        if not config["single_spec"]:
            index_sampler = torch.utils.data.distributed.DistributedSampler(
                list(range(self.sample_size)), shuffle=shuffle, drop_last=False
            )
            self.step = max(1, self.step // config["world_size"])
            shuffle = False
        
        super().__init__(
            dataset=list(range(self.sample_size)),
            batch_size=self.step,
            collate_fn=self.collate_fn,
            num_workers=config["worker"],
            shuffle=shuffle,
            sampler=index_sampler,
            generator=self.generator,
        )

    def _init_batch_size_and_step(self):
        """Initializing :attr:`step` and :attr:`batch_size`."""
        raise NotImplementedError(
            "Method [init_batch_size_and_step] should be implemented"
        )

    def update_config(self, config):
        """Update configure of dataloader, such as :attr:`batch_size`, :attr:`step` etc.

        Args:
            config (Config): The new config of dataloader.
        """
        self.config = config
        self._init_batch_size_and_step()

    def set_batch_size(self, batch_size):
        """Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        self._batch_size = batch_size

    def collate_fn(self):
        """Collect the sampled index, and apply neg_sampling or other methods to get the final data."""
        raise NotImplementedError("Method [collate_fn] must be implemented.")

    def __iter__(self):
        global start_iter
        start_iter = True
        res = super().__iter__()
        start_iter = False
        return res

    def __getattribute__(self, __name: str):
        global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)

class NegSampleDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=True):
        self.logger = getLogger()
        # TODO 在这里对数据集负采样
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_neg_sample_args(self, config, dataset, dl_format, neg_sample_args):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.dl_format = dl_format
        self.neg_sample_args = neg_sample_args
        self.times = 1
        if (
            self.neg_sample_args["distribution"] in ["uniform", "popularity"]
            and self.neg_sample_args["sample_num"] != "none"
        ):
            self.neg_sample_num = self.neg_sample_args["sample_num"]

            if self.dl_format == InputType.POINTWISE:
                self.times = 1 + self.neg_sample_num
                if config['neg_my']:
                    self.sampling_func = self._neg_sample_my
                else:
                    self.sampling_func = self._neg_sample_by_point_wise_sampling

                self.label_field = config["LABEL_FIELD"]
                dataset.set_field_property(
                    self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1
                )
            elif self.dl_format == InputType.PAIRWISE:
                self.times = self.neg_sample_num
                self.sampling_func = self._neg_sample_by_pair_wise_sampling

                self.neg_prefix = config["NEG_PREFIX"]
                self.neg_item_id = self.neg_prefix + self.iid_field

                columns = (
                    [self.iid_field]
                    if dataset.item_feat is None
                    else dataset.item_feat.columns
                )
                for item_feat_col in columns:
                    neg_item_feat_col = self.neg_prefix + item_feat_col
                    dataset.copy_field_property(neg_item_feat_col, item_feat_col)
            else:
                raise ValueError(
                    f"`neg sampling by` with dl_format [{self.dl_format}] not been implemented."
                )

        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            raise ValueError(
                f'`neg_sample_args` [{self.neg_sample_args["distribution"]}] is not supported!'
            )

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_candidate_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num * candidate_num
            )
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
            self.model.train()
            return self.sampling_func(inter_feat, neg_item_ids)
        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_item_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            ) # 除非repeatable_sampler item_ids没用
            return self.sampling_func(inter_feat, neg_item_ids)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        return inter_feat

    def _neg_sample_my(self, inter_feat, neg_item_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data) # 加入其它特征
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = inter_feat.label
        new_data.update(Interaction({self.label_field: labels}))
        return new_data
    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids):
        # TODO: 这里输入的inter_feat是有正有负的，但是又强制全归为了正
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data) # 加入其它特征
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

    def get_model(self, model):
        self.model = model

class TrainDataLoader(NegSampleDataLoader):
    """:class:`TrainDataLoader` is a dataloader for training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        self._set_neg_sample_args(
            config, dataset, config["MODEL_INPUT_TYPE"], config["train_neg_sample_args"]
        )
        self.sample_size = len(dataset)
        # TODO 在这里对数据集负采样
        # dataset可以当做
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        if self.neg_sample_args["distribution"] != "none":
            batch_num = max(batch_size // self.times, 1)
            new_batch_size = batch_num * self.times
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)
    def _get_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        pass
    def update_config(self, config):
        self._set_neg_sample_args(
            config,
            self._dataset,
            config["MODEL_INPUT_TYPE"],
            config["train_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed_data = self.transform(self._dataset, data) # equal
        return self._neg_sampling(transformed_data)


class NegSampleEvalDataLoader(NegSampleDataLoader):
    """:class:`NegSampleEvalDataLoader` is a dataloader for neg-sampling evaluation.
    It is similar to :class:`TrainDataLoader` which can generate negative items,
    and this dataloader also permits that all the interactions corresponding to each user are in the same batch
    and positive interactions are before negative interactions.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.logger = getLogger()
        phase = sampler.phase if sampler is not None else "test"
        self._set_neg_sample_args(
            config, dataset, InputType.POINTWISE, config[f"{phase}_neg_sample_args"]
        )
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_num = dataset.user_num
            dataset.sort(by=dataset.uid_field, ascending=True)
            self.uid_list = []
            start, end = dict(), dict()
            for i, uid in enumerate(dataset.inter_feat[dataset.uid_field].numpy()):
                if uid not in start:
                    self.uid_list.append(uid)
                    start[uid] = i
                end[uid] = i
            self.uid2index = np.array([None] * user_num)
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)
            for uid in self.uid_list:
                self.uid2index[uid] = slice(start[uid], end[uid] + 1)
                self.uid2items_num[uid] = end[uid] - start[uid] + 1
            self.uid_list = np.array(self.uid_list)
            self.sample_size = len(self.uid_list)
        else:
            self.sample_size = len(dataset)
        if shuffle:
            self.logger.warning("NegSampleEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            inters_num = sorted(self.uid2items_num * self.times, reverse=True)
            batch_num = 1
            new_batch_size = inters_num[0]
            for i in range(1, len(inters_num)):
                if new_batch_size + inters_num[i] > batch_size:
                    break
                batch_num = i + 1
                new_batch_size += inters_num[i]
            self.step = batch_num
            self.set_batch_size(new_batch_size)
        else:
            self.step = batch_size
            self.set_batch_size(batch_size)

    def update_config(self, config):
        phase = self._sampler.phase if self._sampler.phase is not None else "test"
        self._set_neg_sample_args(
            config,
            self._dataset,
            InputType.POINTWISE,
            config[f"{phase}_neg_sample_args"],
        )
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)
        if (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
            
        ):
            uid_list = self.uid_list[index]
            data_list = []
            idx_list = []
            positive_u = []
            positive_i = torch.tensor([], dtype=torch.int64)

            for idx, uid in enumerate(uid_list):
                index = self.uid2index[uid]
                transformed_data = self.transform(self._dataset, self._dataset[index])
                data_list.append(self._neg_sampling(transformed_data))
                idx_list += [idx for i in range(self.uid2items_num[uid] * self.times)]
                positive_u += [idx for i in range(self.uid2items_num[uid])]
                positive_i = torch.cat(
                    (positive_i, self._dataset[index][self.iid_field]), 0
                )

            cur_data = cat_interactions(data_list)
            idx_list = torch.from_numpy(np.array(idx_list)).long()
            positive_u = torch.from_numpy(np.array(positive_u)).long()

            return cur_data, idx_list, positive_u, positive_i
        else:
            data = self._dataset[index]
            transformed_data = self.transform(self._dataset, data)
            cur_data = self._neg_sampling(transformed_data)
            return cur_data, None, None, None