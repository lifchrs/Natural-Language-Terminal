from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever
from datasets import Dataset, Features, Sequence, Value
from functools import partial
from config import batch_size, device, output_dir, data_dir, resource_file
import faiss
import os

def embed(documents, ctx_encoder, ctx_tokenizer):
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def load_text(commands):
    def read_file(filename):
        with open(filename, "r") as file:
            contents = file.read()
        return contents
    return [read_file(f"{os.path.join(data_dir, command)}.txt") for command in commands]
commands = ['apropos', 'apt', 'apt-get', 'awk', 'basename', 'base32', 'base64', 'bash', 'bzip2', 'cat', 'chgrp', 'chmod', 'chpasswd', 'chroot', 'cksum', 'clear_console', 'cmp', 'comm', 'cp', 'cpio', 'csplit', 'curl', 'cut', 'date', 'dd', 'df', 'diff', 'diff3', 'dir', 'dircolors', 'dirname', 'dmesg', 'dpkg', 'du', 'echo', 'egrep', 'eject', 'env', 'expand', 'expr', 'fgrep', 'file', 'find', 'fmt', 'fold', 'free', 'fsck', 'gawk', 'grep', 'groupadd', 'groupdel', 'groupmod', 'groups', 'gzip', 'head', 'htop', 'iconv', 'id']
text = load_text(commands)
dpr_ctx_encoder_model_name = "facebook/dpr-ctx_encoder-multiset-base"
dataset = Dataset.from_dict({"title": commands, "text": text})

ctx_encoder = DPRContextEncoder.from_pretrained(dpr_ctx_encoder_model_name).to(device=device)
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(dpr_ctx_encoder_model_name)
new_features = Features(
    {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
)

dataset = dataset.map(
    partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
    batched=True,
    batch_size=batch_size,
    features=new_features,
)

dataset.save_to_disk(os.path.join(output_dir, resource_file))

index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
dataset.add_faiss_index("embeddings", custom_index=index)

index_path = os.path.join(output_dir, f"{resource_file}.faiss")
dataset.get_index("embeddings").save(index_path)


