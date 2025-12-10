import random
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset

Document = Union[str, Image.Image]


class Corpus:
    """
    Corpus class for handling retrieving with simple mapping.
    This class is meant to be overridden by the user to handle their own corpus.

    Args:
        corpus_data (List[Dict[str, Any]]): List of dictionaries containing doc data.
        docid_to_idx_mapping (Optional[Dict[str, int]]): Optional mapping from doc IDs to indices.
    """

    def __init__(
        self,
        corpus_data: List[Dict[str, Any]],
        docid_to_idx_mapping: Optional[Dict[str, int]] = None,
        doc_column_name: str = "doc",
    ):
        """
        Initialize the corpus with the provided data.
        """
        self.corpus_data = corpus_data
        self.docid_to_idx_mapping = docid_to_idx_mapping
        self.doc_column_name = doc_column_name

        assert isinstance(
            self.corpus_data,
            (list, Dataset, HFDataset),
        ), "Corpus data must be a map-style dataset"

        assert self.doc_column_name in self.corpus_data[0], f"Corpus data must contain a column {self.doc_column_name}."

    def __len__(self) -> int:
        """
        Return the number of docs in the corpus.

        Returns:
            int: The number of docs in the corpus.
        """
        return len(self.corpus_data)

    def retrieve(self, docid: Any) -> Document:
        """
        Get the corpus row from the given Doc ID.

        Args:
            docid (str): The id of the document.

        Returns:
            Document: The document retrieved from the corpus.
        """
        if self.docid_to_idx_mapping is not None:
            doc_idx = self.docid_to_idx_mapping[docid]
        else:
            doc_idx = docid
        return self.corpus_data[doc_idx][self.doc_column_name]


class ColPaliEngineDataset(Dataset):
    # Output keys
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        corpus: Optional[Corpus] = None,
        query_column_name: str = "query",
        pos_target_column_name: str = "pos_target",
        neg_target_column_name: str = None,
    ):
        """
        Initialize the dataset with the provided data and external document corpus.

        Args:
            data (Dict[str, List[Any]]): A dictionary containing the dataset samples.
            corpus (Optional[Corpus]): An optional external document corpus to retrieve
            documents (images) from.
        """
        self.data = data
        self.corpus = corpus

        # Column args
        self.query_column_name = query_column_name
        self.pos_target_column_name = pos_target_column_name
        self.neg_target_column_name = neg_target_column_name
        
        # Track skipped corrupt images
        self._skipped_count = 0
        self._max_skip_attempts = 100  # Max attempts to find valid sample

        # Accept any object with __len__ and __getitem__ (duck typing!)
        if not (hasattr(self.data, '__len__') and hasattr(self.data, '__getitem__')):
            raise ValueError("Data must be a map-style dataset (needs __len__ and __getitem__)")

        assert self.query_column_name in self.data[0], f"Data must contain the {self.query_column_name} column"
        assert self.pos_target_column_name in self.data[0], f"Data must contain a {self.pos_target_column_name} column"
        if self.neg_target_column_name is not None:
            assert self.neg_target_column_name in self.data[0], (
                f"Data must contain a {self.neg_target_column_name} column"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Versuche Sample zu laden - bei korrupten Images automatisch zum nächsten springen
        max_attempts = self._max_skip_attempts
        attempt = 0
        
        while attempt < max_attempts:
            current_idx = (idx + attempt) % len(self.data)
            
            try:
                sample = self.data[current_idx]
                
                # Versuche Query zu extrahieren
                query = sample[self.query_column_name]
                
                # VALIDATION: Ensure query is a string (not an image)
                if not isinstance(query, str):
                    if isinstance(query, Image.Image):
                        query = ""
                    elif isinstance(query, (dict, list)):
                        query = ""
                    else:
                        query = str(query) if query is not None else ""

                # Versuche Images zu laden
                pos_targets_raw = sample[self.pos_target_column_name]
                if not isinstance(pos_targets_raw, list):
                    pos_targets_raw = [pos_targets_raw]
                
                pos_targets = []
                for img in pos_targets_raw:
                    if img is None:
                        raise ValueError(f"Missing target at index {current_idx}")
                    elif isinstance(img, Image.Image):
                        try:
                            img.load()  # Validiere Image
                            pos_targets.append(img)
                        except Exception:
                            raise ValueError(f"Corrupt image at index {current_idx}")
                    elif isinstance(img, bytes):
                        # Image als bytes (HuggingFace Dataset format) → convert to PIL
                        try:
                            from io import BytesIO
                            pil_img = Image.open(BytesIO(img))
                            pil_img.load()  # Validiere
                            pos_targets.append(pil_img)
                        except Exception:
                            raise ValueError(f"Corrupt image (bytes) at index {current_idx}")
                    elif isinstance(img, str):
                        # Text passage (für text-only training!)
                        pos_targets.append(img)
                    else:
                        raise ValueError(f"Invalid target type at index {current_idx}: {type(img)}")
                
                # Erfolgreich geladen - weiter mit neg_targets falls vorhanden
                if self.neg_target_column_name is not None:
                    neg_targets_raw = sample[self.neg_target_column_name]
                    if not isinstance(neg_targets_raw, list):
                        neg_targets_raw = [neg_targets_raw]
                    
                    neg_targets = []
                    for img in neg_targets_raw:
                        if img is None:
                            raise ValueError(f"Missing negative target at index {current_idx}")
                        elif isinstance(img, Image.Image):
                            try:
                                img.load()
                                neg_targets.append(img)
                            except Exception:
                                raise ValueError(f"Corrupt negative image at index {current_idx}")
                        elif isinstance(img, bytes):
                            # Image als bytes → convert to PIL
                            try:
                                from io import BytesIO
                                pil_img = Image.open(BytesIO(img))
                                pil_img.load()
                                neg_targets.append(pil_img)
                            except Exception:
                                raise ValueError(f"Corrupt negative image (bytes) at index {current_idx}")
                        elif isinstance(img, str):
                            # Text passage (für text-only training!)
                            neg_targets.append(img)
                        else:
                            raise ValueError(f"Invalid negative target type at index {current_idx}: {type(img)}")
                else:
                    neg_targets = None
                
                # Wenn wir hier ankommen, ist das Sample gültig
                if attempt > 0:
                    self._skipped_count += attempt
                    # Log warning bei korrupten Samples
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"⚠️ Skipped {attempt} corrupt sample(s) before index {current_idx}. "
                        f"Total skipped so far: {self._skipped_count}"
                    )
                
                # Process corpus if needed
                if self.corpus is not None:
                    pos_targets = [self.corpus.retrieve(doc_id) for doc_id in pos_targets]
                    if neg_targets is not None:
                        if len(neg_targets) > 5:
                            neg_targets = random.sample(neg_targets, 5)
                        neg_targets = [self.corpus.retrieve(doc_id) for doc_id in neg_targets]
                
                return {
                    self.QUERY_KEY: query,
                    self.POS_TARGET_KEY: pos_targets,
                    self.NEG_TARGET_KEY: neg_targets,
                }
                
            except Exception as e:
                # Fehler beim Laden (korruptes Image, etc.) - versuche nächsten Index
                error_msg = str(e)
                if ("UnidentifiedImageError" in error_msg or 
                    "cannot identify image" in error_msg.lower() or
                    "Corrupt image" in error_msg or
                    "Missing image" in error_msg or
                    "Invalid image" in error_msg):
                    # Log corrupt sample
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(
                        f"🔍 Corrupt sample at index {current_idx} (attempt {attempt+1}/{max_attempts}): {error_msg[:100]}"
                    )
                    attempt += 1
                    continue  # Versuche nächsten Index
                else:
                    # Anderer Fehler - weiterwerfen
                    raise
        
        # Alle Versuche fehlgeschlagen - wirf Fehler
        raise RuntimeError(f"Could not find valid sample after {max_attempts} attempts starting from index {idx}")

    def get_skip_statistics(self) -> dict:
        """
        Gibt Statistiken über geskippte Samples zurück.
        
        Returns:
            dict mit 'total_skipped', 'dataset_size', 'skip_rate'
        """
        return {
            'total_skipped': self._skipped_count,
            'dataset_size': len(self.data),
            'skip_rate': self._skipped_count / len(self.data) if len(self.data) > 0 else 0
        }
    
    def take(self, n: int) -> "ColPaliEngineDataset":
        """
        Take the first n samples from the dataset.

        Args:
            n (int): The number of samples to take.

        Returns:
            ColPaliEngineDataset: A new dataset containing the first n samples.
        """
        return self.__class__(
            self.data.take(n),
            self.corpus,
            self.query_column_name,
            self.pos_target_column_name,
            self.neg_target_column_name,
        )
