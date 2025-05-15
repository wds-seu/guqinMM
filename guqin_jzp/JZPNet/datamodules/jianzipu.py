import os
import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from PIL import Image
# import pytorch_lightning as pl
import pytorch_lightning as pl
from collections import OrderedDict
from typing import List
from .utils import Vocabulary, Collate, AdaptiveBatchSampler
# from .read_jzp import get_jzp_string, get_jzp_character, get_jzp_structure
from sklearn.model_selection import KFold

class JZPDataset(Dataset):
    def __init__(self,
                 dir_name,
                 node_vocab: Vocabulary,
                 edge_vocab: Vocabulary,
                 indices=None,
                 transform=transforms.ToTensor()):
        super().__init__()
        self.dir_name = dir_name
        self.root_dir = os.path.join("./data/jzp", dir_name)
        df = pd.read_csv(
            os.path.join(self.root_dir, "metadata_3.txt"),
            delimiter="\t",
            header=0,
        )
        self.metadata = df[df["text"].apply(lambda x: False if "\\sqrt [" in x
                                             else True)].reset_index(drop=True)
        if indices is not None:
            self.metadata = self.metadata.loc[indices].reset_index(drop=True)
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        sample_id = self.metadata.iloc[index, 0]
        sample_id = sample_id.split("/")[-1]
        img_path = os.path.join(self.root_dir, "images_gray" , f"{sample_id}")
        img = Image.open(img_path)
        img = self.transform(img)
        
        lg_path = os.path.join(self.root_dir, "tree", f"{sample_id}.lg")

        if not os.path.exists(lg_path):
            print(f"LG file not found: {lg_path}")
            return None
        
        nodes, edges = self._read_lg_file(lg_path)
        return img, nodes, edges, sample_id

    def _read_lg_file(self, lg_path):
        objs = []
        rels_dict = OrderedDict()

        with open(lg_path, "r") as fin:
            for line in fin.readlines():
                line = line.strip()
                tokens = line.split(",")

                if line.startswith("O"):
                    objs.append({
                        "id": tokens[1],
                        "type": tokens[2],
                        "path": tokens[3]
                    })
                elif line.startswith("R"):
                    rels_dict[tokens[2]] = {
                        "src": tokens[1],
                        "dst": tokens[2],
                        "type": tokens[3],
                    }
        rels = rels_dict.values()

        objs = pd.DataFrame(objs, columns=["id", "type", "path"])
        sos_obj_id = f"{Vocabulary.sos_tok}^0"
        eos_obj_id = f"{Vocabulary.eos_tok}^1024"
        
        new_obj = pd.DataFrame([{
            "id": sos_obj_id,
            "type": Vocabulary.sos_tok,
            "path": ""
            }])
        objs = pd.concat([new_obj, objs], ignore_index=True)
        new_obj = pd.DataFrame([{
            "id": eos_obj_id,
            "type": Vocabulary.eos_tok,
            "path": ""
        }])
        objs = pd.concat([objs, new_obj], ignore_index=True)
        objs = (objs.sort_values(
            by="id",
            key=lambda col: col.apply(lambda x: int(x.split("^")[-1])),
            ignore_index=True,
        ).reset_index().set_index("id"))
        rels = pd.DataFrame(rels, columns=["src", "dst", "type"])
        new_rel = pd.DataFrame([{
            "src": sos_obj_id,
            "dst": objs.index[1],
            "type": Vocabulary.pad_tok
        }])
        rels = pd.concat([new_rel, rels], ignore_index=True)
        new_rel = pd.DataFrame([{
            "src": objs.index[-2],
            "dst": eos_obj_id,
            "type": Vocabulary.pad_tok
        }])
        rels = pd.concat([rels, new_rel], ignore_index=True)

        rels = rels.sort_values(
            by="dst",
            key=lambda col: col.apply(lambda x: int(x.split("^")[-1])),
            ignore_index=True,
        )
        
        objs.type = objs.type.apply(lambda x: self.node_vocab.stoi[x])
        rels.type = rels.type.apply(lambda x: self.edge_vocab.stoi[x])
        
        rels["src_type"] = rels.src.apply(lambda x: objs.loc[x, "type"])
        rels["dst_type"] = rels.dst.apply(lambda x: objs.loc[x, "type"])
        rels.src = rels.src.apply(lambda x: objs.loc[x, "index"])
        rels.dst = rels.dst.apply(lambda x: objs.loc[x, "index"])

        objs = objs.reset_index(drop=True).drop(columns="index")

        return objs, rels

    def random_split(self, splits: List[float]):
        assert sum(splits) == 1
        indices = torch.randperm(self.__len__()).tolist()
        lengths = [int(self.__len__() * x) for x in splits]
        lengths[-1] = self.__len__() - sum(lengths[:-1])
        subsets = []
        for i, length in enumerate(lengths):
            start = sum(lengths[:i])
            stop = start + lengths[i]
            subsets.append(
                JZPDataset(
                    self.dir_name,
                    self.node_vocab,
                    self.edge_vocab,
                    indices=indices[start:stop],
                ))
        return subsets
    
    def k_fold_split(self, n):
        kf = KFold(n_splits=n, shuffle=True, random_state=42)
        splits = list(kf.split(range(self.__len__())))
        subsets = []
        for fold_index, (train_idx, val_idx) in enumerate(splits):
            train_subset = JZPDataset(self.dir_name,self.node_vocab,self.edge_vocab,
                                      indices=train_idx)
            val_subset = JZPDataset(self.dir_name,self.node_vocab,self.edge_vocab,
                                      indices=val_idx)
            subsets.append((train_subset, val_subset))
        print(subsets)
        return subsets



class JZPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        node_vocab: Vocabulary,
        edge_vocab: Vocabulary,
        batch_size=16,
        rand_size=128,
        mem_size=5e5,
        train_dir="wushen_aug",
        val_dir=None,
        test_dir="zyt",
        n_splits = 5

    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.collate_fn = Collate(node_vocab.get_pad_idx(), edge_vocab.get_pad_idx())
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.batch_size = batch_size
        self.rand_size = rand_size
        self.mem_size = mem_size
        self.n_splits = n_splits
        self.current_fold = 0

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        if stage == "test":
            self.test_ds = JZPDataset(self.test_dir, self.node_vocab, self.edge_vocab)
        else:
            ds = JZPDataset(self.train_dir, self.node_vocab, self.edge_vocab)
            print(f"Total dataset length: {len(ds)}")
            if self.n_splits > 1:
                k_fold_subsets = ds.k_fold_split(self.n_splits)
                if self.current_fold >= self.n_splits:
                    raise ValueError(f"current_fold {self.current_fold} exceeds n_splits {self.n_splits}")
                self.train_ds, self.val_ds = k_fold_subsets[self.current_fold]  
                self.test_ds = JZPDataset(self.test_dir, self.node_vocab, self.edge_vocab)
            else:
                self.train_ds, self.val_ds, self.test_ds = ds.random_split([0.8, 0.1, 0.1])

                
    def train_dataloader(self):
        batch_sampler = AdaptiveBatchSampler(
            self.train_ds.metadata,
            batch_size=self.batch_size,
            rand_size=self.rand_size,
            mem_size=self.mem_size,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_ds'):
            self.setup(stage="test")
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=True,
        )
    def next_fold(self):
        self.current_fold += 1


