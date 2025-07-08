from transformers import PreTrainedTokenizerFast
from tokenizers.implementations import SentencePieceUnigramTokenizer

model_path = "./spm_unigram_tokenizer_200m/tokenizer.model"

sp_tokenizer = SentencePieceUnigramTokenizer(model_file=model_path)


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=None,
    sp_model_kwargs={"model_file": model_path},
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<bos>",
    eos_token="<eos>",
    model_max_length=2048
)

tokenizer.save_pretrained("./spm_unigram_tokenizer_200m")