import collections
import json
import random

import polars as pl
import torch


class DatasetBase(torch.utils.data.Dataset):

    '''
    map-style dataset
    '''

    def __init__(self, example, tokenizer, config):
        super().__init__()
        self.example = example
        self.tokenizer = tokenizer
        self.config = config
        self.data = collections.defaultdict(list)
        self.read_fn()
        self.data = pl.DataFrame(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :]

    def text_fn(self, text):
        return text if isinstance(text, str) else str(text)

    def label_fn(self, label):
        return label if isinstance(label, int) else int(eval(label))

    def line_fn(self, line):
        raise NotImplementedError

    def read_fn(self):
        with open(self.example, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    for key, value in self.line_fn(line).items():
                        self.data[key].append(value)
                except Exception as e:
                    raise e
                if self.config.do_debug and idx >= 499:
                    break

    def batch_fn(self, data):
        data = pl.concat(data, how='vertical')
        return {column.name: column.to_list() for column in data.get_columns()}

    def tensor_fn(self, feed_dict):
        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict

    def collate_fn(self, data):
        raise NotImplementedError


class Dataset(DatasetBase):

    def __init__(self, example, tokenizer, config):
        super().__init__(
            example=example,
            tokenizer=tokenizer,
            config=config
        )

    def line_fn(self, line):
        if self.config.input_qd:
            if self.config.input_format == 'text':
                sp = line.rstrip().split('\t')[:3]
                text = [self.text_fn(sp[0]), self.text_fn(sp[1])]
                label = self.label_fn(sp[2])
            elif self.config.input_format == 'json':
                sp = json.loads(line.rstrip())
                text = [self.text_fn(sp['query']), self.text_fn(sp['title'])]
                label = self.label_fn(sp['label'])
            else:
                raise NotImplementedError
        else:
            if self.config.input_format == 'text':
                sp = line.rstrip().split('\t')[:2]
                text = self.text_fn(sp[0])
                label = self.label_fn(sp[1])
            elif self.config.input_format == 'json':
                sp = json.loads(line.rstrip())
                text = self.text_fn(sp['query'])
                label = self.label_fn(sp['label'])
            else:
                raise NotImplementedError
        return {"text": text, "label": label}

    def collate_fn(self, data):
        data_dict = self.batch_fn(data)
        texts, labels = data_dict["text"], data_dict["label"]
        feed_dict = self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True
        )
        feed_dict['labels'] = labels
        return self.tensor_fn(feed_dict=feed_dict)


class TokenDataset(DatasetBase):

    def __init__(self, example, tokenizer, config):
        self.label2id = {}
        for label in config.schema:
            self.label2id[label] = len(self.label2id)
        self.id2label = {value: key for key, value in self.label2id.items()}
        super().__init__(
            example=example,
            tokenizer=tokenizer,
            config=config
        )

    def line_fn(self, line):
        if self.config.input_format == 'text':
            sp = json.loads(line.rstrip())
            token = sp[0]
            label = [self.label2id[label] for label in sp[1]]
        elif self.config.input_format == 'json':
            sp = json.loads(line.rstrip())
            token = sp['token']
            label = [self.label2id[label] for label in sp['label']]
            assert len(token) == len(label)
        else:
            raise NotImplementedError
        return {"token": token, "label": label}

    def collate_fn(self, data):
        data_dict = self.batch_fn(data)
        tokens, labels = data_dict["token"], data_dict["label"]
        feed_dict = self.tokenizer(
            tokens,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True
        )
        ner_labels = []
        for batch_index, label in enumerate(labels):
            word_ids = feed_dict.word_ids(batch_index=batch_index)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            ner_labels.append(label_ids)
        feed_dict['labels'] = ner_labels
        return self.tensor_fn(feed_dict=feed_dict)


class InferDataset(DatasetBase):

    def __init__(self, example, tokenizer, config):
        super().__init__(
            example=example,
            tokenizer=tokenizer,
            config=config
        )

    # HACK infer阶段传入的example是数组而非文件
    def read_fn(self):
        for idx, line in enumerate(self.example):
            try:
                if self.config.input_qd:
                    if self.config.input_format == 'text':
                        text = [self.text_fn(line[0]), self.text_fn(line[1])]
                    elif self.config.input_format == 'json':
                        text = [self.text_fn(line['query']), self.text_fn(line['title'])]
                    else:
                        raise NotImplementedError
                else:
                    if self.config.input_format == 'text':
                        text = self.text_fn(line[0])
                    elif self.config.input_format == 'json':
                        text = self.text_fn(line['query'])
                    else:
                        raise NotImplementedError
                self.data["text"].append(text)
            except Exception as e:
                raise e
            if self.config.do_debug and idx >= 499:
                break

    def collate_fn(self, data):
        data_dict = self.batch_fn(data)
        texts = data_dict["text"]
        feed_dict = self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True
        )
        return self.tensor_fn(feed_dict=feed_dict)


class TokenInferDataset(DatasetBase):

    def __init__(self, example, tokenizer, config):
        self.label2id = {}
        for label in config.schema:
            self.label2id[label] = len(self.label2id)
        self.id2label = {value: key for key, value in self.label2id.items()}
        super().__init__(
            example=example,
            tokenizer=tokenizer,
            config=config
        )

    # HACK infer阶段传入的example是数组而非文件
    def read_fn(self):
        for idx, line in enumerate(self.example):
            try:
                if self.config.input_format == 'text':
                    token = line[0]
                    label = [0] * len(token)
                elif self.config.input_format == 'json':
                    token = line['token']
                    label = [0] * len(token)
                else:
                    raise NotImplementedError
                self.data["token"].append(token)
                self.data["label"].append(label)
            except Exception as e:
                raise e
            if self.config.do_debug and idx >= 499:
                break

    def collate_fn(self, data):
        data_dict = self.batch_fn(data)
        tokens, labels = data_dict["token"], data_dict["label"]
        feed_dict = self.tokenizer(
            tokens,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True
        )
        ner_labels = []
        for batch_index, label in enumerate(labels):
            word_ids = feed_dict.word_ids(batch_index=batch_index)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            ner_labels.append(label_ids)
        feed_dict['labels'] = ner_labels
        return self.tensor_fn(feed_dict=feed_dict)


class StreamDatasetBase(torch.utils.data.IterableDataset):

    '''
    iterable-style dataset
    '''

    def __init__(self, example, tokenizer, config, shuffle, buffer_size=65536):
        super().__init__()
        self.example = example
        self.tokenizer = tokenizer
        self.config = config
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.start, self.end = 0, 0
        self.buffer = []
        with open(example, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, _ in enumerate(f):
                self.end += 1
                if self.config.do_debug and idx >= 499:
                    break
        self.worker_mod = 0
        self.worker_num = 1

    def __len__(self):
        assert self.end > self.start
        return self.end - self.start

    def __iter__(self):
        with open(self.example, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                if idx % self.worker_num != self.worker_mod:
                    continue
                try:
                    items = self.line_fn(line)
                except Exception:
                    raise
                if len(self.buffer) == self.buffer_size:
                    if self.shuffle:
                        pop_idx = random.randint(0, self.buffer_size - 1)
                        yield self.buffer[pop_idx]
                        self.buffer[pop_idx] = items
                    else:
                        yield self.buffer.pop(0)
                        self.buffer.append(items)
                else:
                    self.buffer.append(items)
        if self.shuffle:
            random.shuffle(self.buffer)
        while self.buffer:
            yield self.buffer.pop(0)

    def text_fn(self, text):
        return text if isinstance(text, str) else str(text)

    def label_fn(self, label):
        return label if isinstance(label, int) else int(eval(label))

    def line_fn(self, line):
        raise NotImplementedError

    def batch_fn(self, data):
        data_dict = collections.defaultdict(list)
        {data_dict[key].append(line[key]) for line in data for key in line}
        return data_dict

    def tensor_fn(self, feed_dict):
        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict

    def collate_fn(self, data):
        raise NotImplementedError


class StreamDataset(StreamDatasetBase):

    def __init__(self, example, tokenizer, config, shuffle):
        super().__init__(
            example=example,
            tokenizer=tokenizer,
            config=config,
            shuffle=shuffle
        )

    def line_fn(self, line):
        if self.config.input_qd:
            if self.config.input_format == 'text':
                sp = line.rstrip().split('\t')[:3]
                text = [self.text_fn(sp[0]), self.text_fn(sp[1])]
                label = self.label_fn(sp[2])
            elif self.config.input_format == 'json':
                sp = json.loads(line.rstrip())
                text = [self.text_fn(sp['query']), self.text_fn(sp['title'])]
                label = self.label_fn(sp['label'])
            else:
                raise NotImplementedError
        else:
            if self.config.input_format == 'text':
                sp = line.rstrip().split('\t')[:2]
                text = self.text_fn(sp[0])
                label = self.label_fn(sp[1])
            elif self.config.input_format == 'json':
                sp = json.loads(line.rstrip())
                text = self.text_fn(sp['query'])
                label = self.label_fn(sp['label'])
            else:
                raise NotImplementedError
        return {"text": text, "label": label}

    def collate_fn(self, data):
        data_dict = self.batch_fn(data)
        texts, labels = data_dict["text"], data_dict["label"]
        feed_dict = self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True
        )
        feed_dict['labels'] = labels
        return self.tensor_fn(feed_dict=feed_dict)


def get_dataloader(
    dataset,
    example,
    tokenizer,
    config,
    batch_size=32,
    shuffle=True,
    num_workers=16,
    drop_last=False
):
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset = getattr(dataset, 'dataset', dataset)
        dataset.worker_mod = worker_id
        dataset.worker_num = num_workers if num_workers > 0 else 1

    if issubclass(dataset, DatasetBase):
        d = dataset(
            example=example,
            tokenizer=tokenizer,
            config=config
        )
        return torch.utils.data.DataLoader(
            dataset=d,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=d.collate_fn,
            drop_last=drop_last
        )
    elif issubclass(dataset, StreamDatasetBase):
        d = dataset(
            example=example,
            tokenizer=tokenizer,
            config=config,
            shuffle=shuffle
        )
        return torch.utils.data.DataLoader(
            dataset=d,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=d.collate_fn,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn
        )
    else:
        raise NotImplementedError
