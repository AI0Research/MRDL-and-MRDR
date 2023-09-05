import json

import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, example, tokenizer, *args, **kwargs):
        super().__init__()
        def label2id_fn(x): return int(eval(x.replace('.0', ''))) if isinstance(x, str) else x
        def text_fn(x): return x if x.replace('"', '').replace("'", "") else ""
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.texts = []
        self.labels = []
        with open(example, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if kwargs.get('input_qd'):
                        # query\tdoc\tlabel
                        if kwargs.get('input_format') == 'text':
                            tmp_sp = line.rstrip().split('\t')[:3]
                            text = [text_fn(tmp_sp[0]), text_fn(tmp_sp[1])]
                            label = label2id_fn(tmp_sp[2])
                        elif kwargs.get('input_format') == 'json':
                            tmp_sp = json.loads(line.rstrip())
                            text = [text_fn(tmp_sp['query']), text_fn(tmp_sp['title'])]
                            label = label2id_fn(tmp_sp['label'])
                        else:
                            raise NotImplementedError('input format not exists ...')
                    else:
                        # query\tlabel
                        if kwargs.get('input_format') == 'text':
                            tmp_sp = line.rstrip().split('\t')[:2]
                            text = text_fn(tmp_sp[0])
                            label = label2id_fn(tmp_sp[1])
                        elif kwargs.get('input_format') == 'json':
                            tmp_sp = json.loads(line.rstrip())
                            text = text_fn(tmp_sp['query'])
                            label = label2id_fn(tmp_sp['label'])
                        else:
                            raise NotImplementedError('input format not exists ...')
                    self.texts.append(text)
                    self.labels.append(label)
                except Exception:
                    print("line {} has invalid format ...".format(idx))
                    continue
                if kwargs.get('do_debug') and idx >= 499:
                    break
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]
        label = self.labels[index]
        return [texts, label]

    def collate_fn(self, data, *args, **kwargs):
        data = list(zip(*data))
        texts, label = data[:2]
        texts = list(texts)
        label = list(label)
        feed_dict = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        feed_dict['labels'] = label
        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict


class InferDataset(torch.utils.data.Dataset):

    def __init__(self, example, tokenizer, *args, **kwargs):
        super().__init__()
        def label2id_fn(x): return int(eval(x.replace('.0', ''))) if isinstance(x, str) else x
        def text_fn(x): return x if x.replace('"', '').replace("'", "") else ""
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.texts = []
        for idx, line in enumerate(example):
            try:
                if kwargs.get('input_qd'):
                    if kwargs.get('input_format') == 'text':
                        text = [text_fn(line[0]), text_fn(line[1])]
                    elif kwargs.get('input_format') == 'json':
                        text = [text_fn(line['query']), text_fn(line['title'])]
                    else:
                        raise NotImplementedError('input format not exists ...')
                else:
                    if kwargs.get('input_format') == 'text':
                        text = text_fn(line[0])
                    elif kwargs.get('input_format') == 'json':
                        text = text_fn(line['query'])
                    else:
                        raise NotImplementedError('input format not exists ...')
                self.texts.append(text)
            except Exception:
                print("line {} has invalid format ...".format(idx))
                continue
            if kwargs.get('do_debug') and idx >= 499:
                break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]
        return texts

    def collate_fn(self, data, *args, **kwargs):
        texts = data
        texts = list(texts)
        feed_dict = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict


def DataLoader(
    dataset,
    example,
    tokenizer,
    batch_size=32,
    shuffle=True,
    num_workers=16,
    drop_last=False,
    *args,
    **kwargs
):
    d = dataset(
        example=example,
        tokenizer=tokenizer,
        *args,
        **kwargs
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=d,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=d.collate_fn,
        drop_last=drop_last
    )
    return data_loader
