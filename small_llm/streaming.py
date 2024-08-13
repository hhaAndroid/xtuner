import json

import numpy as np

from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import copy
import os
from xtuner._lite.datasets.text import TextCollator


class PretrainTokenizeFunction:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, item):
        try:
            text = item['text'] + self.tokenizer.eos_token
        except:
            text = item['content'] + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = [len(input_ids)]
        return {"input_ids": input_ids, "labels": copy.deepcopy(input_ids), "num_tokens": num_tokens}


class Streaming:

    def __init__(self, file, max_epoch=1):

        self.file = file
        self.offset = 0
        self.epoch = 1
        self.max_epoch = max_epoch

    def __iter__(self):
        return self

    def __next__(self):

        with open(self.file, "r") as f:
            f.seek(self.offset)
            line = f.readline()

            if line:
                self.offset = f.tell()
                return line

            elif not line and self.epoch < self.max_epoch:
                # Completed one round , starting the next round.
                self.offset = 0
                self.epoch += 1
                return next(self)

            else:
                raise StopIteration

        return line

    def state_dict(self):
        return {
            "file": self.file,
            "offset": self._offset,
            "epoch": self.epoch,
            "max_epoch": self.max_epoch,
        }

    def load_state_dict(self, state_dict):
        assert self.file == state_dict["file"]

        self.offset = state_dict["offset"]
        self.epoch = state_dict["epoch"]
        self.max_epoch = state_dict["max_epoch"]

    @classmethod
    def from_state_dict(cls, state_dict):
        streaming = cls(state_dict["file"], state_dict["max_epoch"])
        streaming.offset = state_dict["offset"]
        streaming.epoch = state_dict["epoch"]
        return streaming

    def reset(self):
        self.offset = 0


class MultiStreamingDataset(IterableDataset):

    def __init__(
        self,
        streamings,
        weights,
        max_length,
        tokenize_fn,
        seed,
        dp_rank,
        dp_world_size,
        pack="hard",
        crossover=False,
    ):

        assert len(streamings) == len(weights)
        self.streamings = streamings
        self.activated = np.array([True for _ in self.streamings], dtype=bool)
        for sid, stream in enumerate(self.streamings):
            stream.reset()
            try:
                # Skip the data that does not belong to this rank.
                for _ in range(dp_rank):
                    next(stream)

            except StopIteration:
                # The current streaming has ended.
                self.activated[sid] = False

        if sum(self.activated) == 0:
            raise RuntimeError(

                f"[DP_RANK {dp_rank}] All streaming contain "
                f"less than {dp_rank} samples, please ensure that "
                "the number of samples in each streaming is greater "
                f"than dp_world_size({dp_world_size})"
            )

        self.random_state = np.random.RandomState(seed + dp_rank)
        self.weights = np.array(weights)

        self.pack = pack
        self.max_length = max_length
        self.tokenize_fn = tokenize_fn
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.crossover = crossover

    def reactivate(self):
        self.activated = [True for _ in self.streamings]
        for stream in self.streamings:
            stream.offset = 0
            for _ in range(self.dp_rank):
                next(stream)

    @property
    def probabilities(self):
        if sum(self.activated) == 0:
            # All streamings have ended, beginning the next round
            self.reactivate()

        probs = (self.weights * self.activated) / sum(self.weights[self.activated])
        return probs

    @property
    def num_streamings(self):
        return len(self.streamings)

    def per_rank_next(self, streaming_id):

        sid = streaming_id
        streaming = self.streamings[sid]

        try:
            data = next(streaming)
        except StopIteration:
            # The current streaming has ended, switch to another streaming to continue reading data.
            self.activated[sid] = False
            sid = self.random_state.choice(self.num_streamings, p=self.probabilities)
            return self.per_rank_next(sid)

        # Skip the data that does not belong to this rank.
        try:
            for _ in range(self.dp_world_size - 1):
                next(streaming)
        except StopIteration:
            # The current streaming has ended.
            self.activated[sid] = False

        return data, sid

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info and worker_info.num_workers > 1:
            raise NotImplementedError(
                "`MultiStreamingDataset` only supports dataloader num_workers <= 1"
            )

        input_ids = []
        labels = []
        num_tokens = []
        while True:
            sid = self.random_state.choice(self.num_streamings, p=self.probabilities)

            if self.pack:

                while len(input_ids) < self.max_length:
                    if self.crossover:
                        # Packed data is composed of data from different streaming.
                        sid = self.random_state.choice(
                            self.num_streamings, p=self.probabilities
                        )

                    # The `sid` may have already been changed
                    line, sid = self.per_rank_next(sid)

                    # TODO support for decoding other formats
                    raw_data = json.loads(line)
                    tokenized = self.tokenize_fn(raw_data)

                    if 0 in tokenized["num_tokens"]:
                        breakpoint()
                    if self.pack == "hard":
                        input_ids.extend(tokenized["input_ids"])
                        labels.extend(tokenized["labels"])
                        num_tokens.extend(tokenized["num_tokens"])
                    elif self.pack == "soft":
                        input_ids.extend(tokenized["input_ids"][: self.max_length])
                        labels.extend(tokenized["labels"][: self.max_length])
                        num_tokens.extend(
                            [min(tokenized["num_tokens"][0], self.max_length)]
                        )
                    else:
                        raise NotImplementedError()

                packed_tokens = copy.deepcopy(num_tokens)
                if len(input_ids) == self.max_length:
                    consumed_tokens = self.max_length

                elif len(input_ids) > self.max_length and self.pack == "hard":
                    consumed_tokens = min(sum(num_tokens), self.max_length)
                    packed_tokens[-1] = consumed_tokens - sum(packed_tokens[:-1])

                elif len(input_ids) > self.max_length and self.pack == "soft":
                    consumed_tokens = sum(num_tokens[:-1])
                    packed_tokens = packed_tokens[:-1]

                else:
                    raise RuntimeError()

                packed_ids = input_ids[:consumed_tokens]
                packed_labels = labels[:consumed_tokens]

                remain_tokens = len(input_ids[consumed_tokens:])
                if remain_tokens:
                    input_ids = input_ids[consumed_tokens:]
                    labels = labels[consumed_tokens:]
                    num_tokens = [remain_tokens]
                else:
                    input_ids = []
                    labels = []
                    num_tokens = []

                yield {
                    "input_ids": packed_ids,
                    "labels": packed_labels,
                    "num_tokens": packed_tokens,
                }

            else:
                line, _ = self.per_rank_next(sid)
                # TODO support for decoding other formats
                raw_data = json.loads(line)
                tokenized = self.tokenize_fn(raw_data)

                yield {
                    "input_ids": tokenized["input_ids"][: self.max_length],
                    "labels": tokenized["labels"][: self.max_length],
                    "num_tokens": [min(tokenized["num_tokens"][0], self.max_length)],
                }

    def state_dict(self):
        cur_random_state = self.random_state.get_state()
        cur_streamings = [stream.state_dict() for stream in self.streamings]

        return {
            "random_state": cur_random_state,
            "streamings": cur_streamings,
            "weights": self.weights,
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
        }

    def load_state_dict(self):

        assert self.dp_rank == state_dict["dp_rank"]
        assert self.dp_world_size == state_dict["dp_world_size"]
        assert self.num_streamings == len(state_dict["streamings"])

        self.random_state.set_state(state_dict["random_state"])
        self.weights = state_dict["weights"]

        for i in range(self.num_streamings):
            self.streamings[i].load_state_dict(state_dict["streamings"][i])


class StreamingDataset(MultiStreamingDataset):
    def __init__(
        self,
        folder,
        weight_file,
        max_length,
        tokenize_fn,
        seed,
        dp_rank,
        dp_world_size,
        pack="hard",
        crossover=False,
    ):
        def find_weight(jsonl_path, weightfile):
            for key in weightfile:
                if key in jsonl_path:
                    return weightfile[key]

        streamings = []
        weights = []
        weight_file = json.load(open(weight_file))
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    path = os.path.join(dirpath, filename)
                    w = find_weight(path, weight_file)
                    if w is not None:
                        weights.append(w)
                        streamings.append(Streaming(path, max_epoch=1))
        super().__init__(
            streamings,
            weights,
            max_length,
            tokenize_fn,
            seed,
            dp_rank,
            dp_world_size,
            pack,
            crossover,
        )


if __name__ == "__main__":
    streamings = []
    weights = []
    for _ in range(1):
        weights.append(1)
        streamings.append(Streaming("test_data.jsonl", max_epoch=2))

    # from xtuner._lite.datasets import TextTokenizeFunction
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "checkpoints/intern_1_8_b", trust_remote_code=True
    )
    tokenize_fn = PretrainTokenizeFunction(tokenizer)
    dataset = StreamingDataset("data/wanjuan", 4, tokenize_fn, 1, 0, 1)

    loader = DataLoader(dataset, batch_size=1, collate_fn=TextCollator())
    for i, _ in enumerate(loader):
        print(i)
