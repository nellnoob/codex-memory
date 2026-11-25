import hashlib
from pathlib import Path
import tempfile
from threading import Lock
from urllib.request import urlopen

REPO = "ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION = "c61e626"
MODEL = "model_quint8_avx2.onnx"
TOKENIZER = "tokenizer.json"
FILES = {
    MODEL: ("onnx/" + MODEL, "a6022dd8220ea6f6595562a1328ee216f4a94faa55362f2f4747c80f1e78772e"),
    TOKENIZER: (TOKENIZER, "4f2842d568e2724370aec203652a42ac783c7937f8347a1a2cc7506d71f1582f"),
}
DIMENSIONS = 384
MAX_TOKENS = 1024
MODEL_ID = f"{REPO}@{REVISION}:avx2-int8:cls-l2:{DIMENSIONS}d:{MAX_TOKENS}t"


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download_model(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, (remote_path, expected_hash) in FILES.items():
        target = directory / filename
        if target.is_file() and digest(target) == expected_hash:
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote_path}"
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if digest(temporary) != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch for {filename}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def load_runtime():
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    return np, ort, Tokenizer


class Granite:
    def __init__(self, directory: Path, cpu_threads: int = 2):
        self.directory = directory
        self.cpu_threads = cpu_threads
        self._lock = Lock()
        self._session = None

    def _load(self, ort, tokenizer_type):
        for filename, (_, expected_hash) in FILES.items():
            path = self.directory / filename
            if not path.is_file() or digest(path) != expected_hash:
                raise RuntimeError(f"Missing or invalid model file {path}. Run download-model.")

        tokenizer = tokenizer_type.from_file(str(self.directory / TOKENIZER))
        if tokenizer.padding is None:
            raise RuntimeError("Granite tokenizer is missing its padding configuration")
        tokenizer.enable_truncation(max_length=MAX_TOKENS)

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.cpu_threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(
            str(self.directory / MODEL), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = tokenizer
        self._session = session

    def encode(self, texts: list[str]):
        np, ort, tokenizer_type = load_runtime()

        with self._lock:
            if self._session is None:
                self._load(ort, tokenizer_type)
            encoded = self._tokenizer.encode_batch(texts)
            inputs = {
                "input_ids": np.asarray([row.ids for row in encoded], dtype=np.int64),
                "attention_mask": np.asarray([row.attention_mask for row in encoded], dtype=np.int64),
            }
            (hidden,) = self._session.run(["last_hidden_state"], inputs)
        vectors = hidden[:, 0, :].astype("<f4", copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if vectors.shape != (len(texts), DIMENSIONS) or not np.all(np.isfinite(norms) & (norms > 0)):
            raise RuntimeError("Granite returned invalid vectors")
        return vectors / norms
