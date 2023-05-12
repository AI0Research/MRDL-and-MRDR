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


class TwinDataset(torch.utils.data.Dataset):

    def __init__(self, example, tokenizer, *args, **kwargs):
        super().__init__()
        def label2id_fn(x): return int(eval(x.replace('.0', ''))) if isinstance(x, str) else x
        def text_fn(x): return x if x.replace('"', '').replace("'", "") else ""
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.querys = []
        self.titles = []
        self.labels = []
        with open(example, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if kwargs.get('input_qd'):
                        # query\tdoc\tlabel
                        if kwargs.get('input_format') == 'text':
                            tmp_sp = line.rstrip().split('\t')[:3]
                            query = text_fn(tmp_sp[0])
                            title = text_fn(tmp_sp[1])
                            label = label2id_fn(tmp_sp[2])
                        elif kwargs.get('input_format') == 'json':
                            tmp_sp = json.loads(line.rstrip())
                            query = text_fn(tmp_sp['query'])
                            title = text_fn(tmp_sp['title'])
                            label = label2id_fn(tmp_sp['label'])
                        else:
                            raise NotImplementedError('input format not exists ...')
                    else:
                        # TODO
                        pass
                    self.querys.append(query)
                    self.titles.append(title)
                    self.labels.append(label)
                except Exception:
                    print("line {} has invalid format ...".format(idx))
                    continue
                if kwargs.get('do_debug') and idx >= 499:
                    break
        assert len(self.querys) == len(self.titles) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return [self.querys[index], self.titles[index], self.labels[index]]

    def collate_fn(self, data, *args, **kwargs):
        data = list(zip(*data))
        query, title, label = data[:3]
        query = list(query)
        title = list(title)
        label = list(label)
        feed_dict = self.tokenizer(
            query,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        title_feed_dict = self.tokenizer(
            title,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        for key, value in title_feed_dict.items():
            feed_dict[key+'_pair'] = value
        feed_dict['labels'] = label

        # HACK duplicate
        for key, value in feed_dict.items():
            new_value = []
            for row in value:
                new_value.append(row)
                new_value.append(row)
            feed_dict[key] = new_value

        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict


class PairWiseDataset(torch.utils.data.Dataset):

    """
    训练数据: query \t doc \t doc_pair \t label \t label_pair \t label_rank
    label_rank: -1表示doc比doc_pair更不相关，0相同，1表示doc比doc_pair更相关
    测试数据: query \t doc \t label
    """

    def __init__(self, example, tokenizer, *args, **kwargs):
        super().__init__()
        def label2id_fn(x): return int(eval(x.replace('.0', ''))) if isinstance(x, str) else x
        def text_fn(x): return x if x.replace('"', '').replace("'", "") else ""
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.texts, self.texts_pair = [], []
        self.labels, self.labels_pair = [], []
        self.labels_rank = []
        with open(example, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if kwargs.get('input_qd'):
                        if kwargs.get('input_format') == 'text':
                            tmp_sp = line.rstrip().split('\t')[:6]
                            query = tmp_sp[0]
                            if len(tmp_sp) > 3:
                                doc, doc_pair = text_fn(tmp_sp[1]), text_fn(tmp_sp[2])
                                text = [query, doc]
                                text_pair = [query, doc_pair]
                                label = label2id_fn(tmp_sp[3])
                                label_pair = label2id_fn(tmp_sp[4])
                                label_rank = label2id_fn(tmp_sp[5])
                            else:
                                # HACK
                                doc, doc_pair = text_fn(tmp_sp[1]), text_fn(tmp_sp[1])
                                text = [query, doc]
                                text_pair = [query, doc_pair]
                                label = label2id_fn(tmp_sp[2])
                                label_pair = label2id_fn(tmp_sp[2])
                                label_rank = 0
                        elif kwargs.get('input_format') == 'json':
                            tmp_sp = json.loads(line.rstrip())
                            query = tmp_sp['query']
                            if 'title_pair' in tmp_sp and 'label_pair' in tmp_sp and 'label_rank' in tmp_sp:
                                doc, doc_pair = text_fn(tmp_sp['title']), text_fn(tmp_sp['title_pair'])
                                text = [query, doc]
                                text_pair = [query, doc_pair]
                                label = label2id_fn(tmp_sp['label'])
                                label_pair = label2id_fn(tmp_sp['label_pair'])
                                label_rank = label2id_fn(tmp_sp['label_rank'])
                            else:
                                doc, doc_pair = text_fn(tmp_sp['title']), text_fn(tmp_sp['title'])
                                text = [query, doc]
                                text_pair = [query, doc_pair]
                                label = label2id_fn(tmp_sp['label'])
                                label_pair = label2id_fn(tmp_sp['label'])
                                label_rank = 0
                        else:
                            raise NotImplementedError('input format not exists ...')
                    else:
                        # TODO
                        pass
                    self.texts.append(text)
                    self.texts_pair.append(text_pair)
                    self.labels.append(label)
                    self.labels_pair.append(label_pair)
                    self.labels_rank.append(label_rank)
                except Exception:
                    print("line {} has invalid format ...".format(idx))
                    continue
                if kwargs.get('do_debug') and idx >= 499:
                    break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return [self.texts[index], self.labels[index], self.texts_pair[index], self.labels_pair[index], self.labels_rank[index]]

    def collate_fn(self, data, *args, **kwargs):
        data = list(zip(*data))
        texts, label, texts_pair, label_pair, label_rank = [list(i) for i in data[:5]]
        feed_dict = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        feed_dict['labels'] = label
        feed_dict['labels_rank'] = label_rank
        feed_dict_pair = self.tokenizer(
            texts_pair,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        feed_dict_pair['labels'] = label_pair
        for key, value in feed_dict_pair.items():
            feed_dict[key+'_pair'] = value
        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict


class SoftLabelDataset(torch.utils.data.Dataset):

    def __init__(self, example, tokenizer, *args, **kwargs):
        from scipy.special import softmax
        super().__init__()
        def label2id_fn(x): return int(eval(x))
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.texts = []
        self.labels = []
        self.soft_labels = []
        with open(example, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if kwargs.get('input_qd'):
                        # query\tdoc\tlabel
                        tmp_sp = line.rstrip().split('\t')[:5]
                        text_a = tmp_sp[0]
                        text_b = tmp_sp[1]
                        text_a = text_a if text_a.replace('"', '').replace("'", "") else ""  # spark的csv保存，空句可能带引号
                        text_b = text_b if text_b.replace('"', '').replace("'", "") else ""
                        text = [text_a, text_b]
                        label = tmp_sp[2].replace('.0', '')

                        if len(tmp_sp) > 3:
                            soft_label = [0] * 5
                            for label_idx in eval(tmp_sp[3].replace('nan', label)):
                                if int(label_idx) not in [0, 1, 2, 3, 4]:
                                    continue
                                soft_label[int(label_idx)] += 1
                            soft_label = softmax(soft_label)
                            self.soft_labels.append(soft_label)
                        else:
                            soft_label = [0] * 5
                            soft_label[label2id_fn(label)] = 5
                            self.soft_labels.append(soft_label)
                    else:
                        # TODO
                        pass
                    self.texts.append(text)
                    self.labels.append(label2id_fn(label))
                except Exception:
                    print("line {} has invalid format ...".format(idx))
                    continue
                if kwargs.get('do_debug') and idx >= 499:
                    break
        assert len(self.texts) == len(self.labels) == len(self.soft_labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]
        label = self.labels[index]
        soft_label = self.soft_labels[index]
        return [texts, label, soft_label]

    def collate_fn(self, data, *args, **kwargs):
        data = list(zip(*data))
        texts, label, soft_label = data[:3]
        texts = list(texts)
        label = list(label)
        soft_label = list(soft_label)
        feed_dict = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        feed_dict['labels'] = label
        feed_dict['soft_labels'] = soft_label
        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict


class MarkerDataset(torch.utils.data.Dataset):

    def __init__(self, example, tokenizer, *args, **kwargs):
        super().__init__()
        def label2id_fn(x): return int(eval(x))
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.texts = []
        self.labels = []
        with open(example, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if kwargs.get('input_qd'):
                        # query \t doc \t label \t query_pos \t doc_pos
                        tmp_sp = line.rstrip().split('\t')[:5]
                        text_a = tmp_sp[0]
                        text_b = tmp_sp[1]
                        text_a = text_a if text_a.replace('"', '').replace("'", "") else ""  # spark的csv保存，空句可能带引号
                        text_b = text_b if text_b.replace('"', '').replace("'", "") else ""
                        text = [text_a, text_b]
                        label = tmp_sp[2].replace('.0', '')
                        text_a_pos = eval(tmp_sp[3])
                        text_b_pos = eval(tmp_sp[4])
                        pos2special_tokens = {}
                        for en in text_a_pos:
                            left, right = en
                            if right-left == 1 and text_a[left] == ' ':
                                continue
                            pos2special_tokens[right-1] = ['R', '[unused1]']
                        new_text = []
                        for char_idx, char in enumerate(text_a):
                            if char_idx in pos2special_tokens:
                                idx_type, type_marker = pos2special_tokens[char_idx]
                                if idx_type == 'L':
                                    new_text.append(type_marker)
                                    new_text.append(char)
                                else:
                                    new_text.append(char)
                                    new_text.append(type_marker)
                            else:
                                new_text.append(char)
                        text_a = ''.join(new_text)
                        pos2special_tokens = {}
                        for en in text_b_pos:
                            left, right = en
                            if right-left == 1 and text_b[left] == ' ':
                                continue
                            pos2special_tokens[right-1] = ['R', '[unused1]']
                        new_text = []
                        for char_idx, char in enumerate(text_b):
                            if char_idx in pos2special_tokens:
                                idx_type, type_marker = pos2special_tokens[char_idx]
                                if idx_type == 'L':
                                    new_text.append(type_marker)
                                    new_text.append(char)
                                else:
                                    new_text.append(char)
                                    new_text.append(type_marker)
                            else:
                                new_text.append(char)
                        text_b = ''.join(new_text)
                        text = [text_a, text_b]
                    else:
                        pass
                        # TODO
                    self.texts.append(text)
                    self.labels.append(label2id_fn(label))
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


class LabelCRDataset(torch.utils.data.Dataset):

    def __init__(self, example, tokenizer, *args, **kwargs):
        super().__init__()
        def label2id_fn(x): return int(eval(x.replace('.0', ''))) if isinstance(x, str) else x
        def text_fn(x): return x if x.replace('"', '').replace("'", "") else ""
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.texts = []
        self.labels = []
        self.labels_fix = []
        with open(example, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if kwargs.get('input_qd'):
                        # query\tdoc\tlabel
                        if kwargs.get('input_format') == 'text':
                            tmp_sp = line.rstrip().split('\t')[:4]
                            text = [text_fn(tmp_sp[0]), text_fn(tmp_sp[1])]
                            label = label2id_fn(tmp_sp[2])
                            label_fix = label2id_fn(tmp_sp[3])
                        elif kwargs.get('input_format') == 'json':
                            tmp_sp = json.loads(line.rstrip())
                            text = [text_fn(tmp_sp['query']), text_fn(tmp_sp['title'])]
                            label = label2id_fn(tmp_sp['label'])
                            label_fix = label2id_fn(tmp_sp['label_fix'])
                        else:
                            raise NotImplementedError('input format not exists ...')
                    else:
                        # TODO
                        pass
                    self.texts.append(text)
                    self.labels.append(label)
                    self.labels_fix.append(label_fix)
                except Exception:
                    print("line {} has invalid format ...".format(idx))
                    continue
                if kwargs.get('do_debug') and idx >= 499:
                    break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return [self.texts[index], self.labels[index], self.labels_fix[index]]

    def collate_fn(self, data, *args, **kwargs):
        data = list(zip(*data))
        texts, label, label_fix = data[:3]
        texts = list(texts)
        label = list(label)
        label_fix = list(label_fix)
        feed_dict = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True
        )
        feed_dict['labels'] = label
        feed_dict['labels_fix'] = label_fix
        for key in feed_dict.keys():
            if isinstance(feed_dict[key], (str, int, float, dict)):
                continue
            feed_dict[key] = torch.tensor(feed_dict[key])
        return feed_dict


class LabelCRInferDataset(torch.utils.data.Dataset):

    def __init__(self, example, tokenizer, *args, **kwargs):
        super().__init__()
        def label2id_fn(x): return int(eval(x.replace('.0', ''))) if isinstance(x, str) else x
        def text_fn(x): return x if x.replace('"', '').replace("'", "") else ""
        self.example = example
        self.tokenizer = tokenizer
        self.max_seq_length = kwargs.get('max_seq_length')
        self.texts = []
        self.labels = []
        for idx, line in enumerate(example):
            try:
                if kwargs.get('input_qd'):
                    if kwargs.get('input_format') == 'text':
                        text = [text_fn(line[0]), text_fn(line[1])]
                        label = line[2]
                    elif kwargs.get('input_format') == 'json':
                        text = [text_fn(line['query']), text_fn(line['title'])]
                        label = line['label']
                    else:
                        raise NotImplementedError('input format not exists ...')
                else:
                    # TODO
                    pass
                self.texts.append(text)
                self.labels.append(label)
            except Exception:
                print("line {} has invalid format ...".format(idx))
                continue
            if kwargs.get('do_debug') and idx >= 499:
                break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return [self.texts[index], self.labels[index]]

    def collate_fn(self, data, *args, **kwargs):
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
