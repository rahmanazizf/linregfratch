import numpy as np
import pandas as pd
from typing import Union

# variasikan indeks untuk data validasi

# cacah data validasi 

class KFold:

    def __init__(self, n_split, random_state: int = 42, is_shufled: bool = False) -> None:
        self.n_split = n_split
        self.random_state = random_state
        self.is_shufled = is_shufled

    def _split_indices(self, X: np.ndarray | pd.DataFrame):
        """Mencacah indeks X menjadi n-bagian
        X: iterables
        return
            generator objek, indeks data dari setiap bagian yang telah dipecah
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        np.random.seed()
        shufle = np.random.choice(indices, n_samples, replace=False) if self.is_shufled else indices

        n_split = 3
        fold_sizes = np.ones(n_split, int) * int(n_samples / n_split)
        fold_sizes[: n_samples%n_split] += 1
        # print(fold_sizes)

        current = 0
        for fold_size in fold_sizes:
            stop = fold_size + current
            yield shufle[current:stop]

            current = stop

    def split(self, X: np.ndarray | pd.DataFrame):
        """Membagi indeks data training dan validasi
        X: iterables, data masukan yang ingin dipecah
        kembalian
            indeks data training, indeks data validasi
        """
        # panggil fungsi split indices
        # indeks validation set diperoleh dari fungsi _split indices
        # indeks sisanya digunakan untuk training
        # kembalian berupa generator objek indeks training dan indeks validasi
        for val_ids in self._split_indices(X):
            train_ids = np.array([ids for ids in np.arange(len(X)) if not ids in val_ids])
            yield (train_ids, val_ids)
        # TODO: tambah stratify
        # konsep stratify
        # membuat rasio label pada set data yang telah dipecah mirip seperti set data sebelum dipecah
        # caranya dengan memberikan masukan data output
        # hitung rasio label dalam output
        # np.random.choice?
        # hitung probability setiap label
        # tambahkan 