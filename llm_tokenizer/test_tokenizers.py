import sentencepiece as spm
from transformers import AutoTokenizer
import statistics

# Test texts in different languages (same as before)
test_texts = {
    "English": [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning algorithms can process natural language efficiently.",
        "Tokenization is a fundamental step in natural language processing.",
        "Large language models require sophisticated preprocessing techniques.",
        "OpenAI's GPT models use byte-pair encoding for tokenization."
    ],
    "Russian": [
        "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Алгоритмы машинного обучения могут эффективно обрабатывать естественный язык.",
        "Токенизация является фундаментальным шагом в обработке естественного языка.",
        "Большие языковые модели требуют сложных методов предварительной обработки.",
        "Модели GPT от OpenAI используют кодирование по парам байтов для токенизации."
    ],
    "Kazakh": [
        "Жылдам қоңыр түлкі жалқау иттің үстінен секіреді.",
        "Машиналық оқыту алгоритмдері табиғи тілді тиімді өңдей алады.",    
        "Токенизация табиғи тілді өңдеудегі негізгі қадам болып табылады.",
        "Үлкен тілдік модельдер күрделі алдын ала өңдеу әдістерін талап етеді.",
        "OpenAI-дың GPT модельдері токенизация үшін байт-жұп кодтауды пайдаланады."
    ]
}

def load_tokenizers():
    """Load all tokenizers for comparison."""
    tokenizers = {}
    
    print("Loading tokenizers...")
    
    # Your custom tokenizer
    try:
        sp = spm.SentencePieceProcessor()
        sp.load("/home/sanzhar/llm_project/llm_tokenizer/spm_unigram_tokenizer_200m/tokenizer_multilingual.model")
        tokenizers["Your Custom"] = ("sentencepiece", sp)
        print("✅ Loaded your custom tokenizer")
    except:
        print("❌ Failed to load your custom tokenizer")
    
    # HuggingFace tokenizers
    hf_models = {
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
        "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
        "Gemma-2-9B": "google/gemma-2-9b-it"
    }
    
    for name, model_id in hf_models.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizers[name] = ("huggingface", tokenizer)
            print(f"✅ Loaded {name}")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
    
    return tokenizers

def tokenize_text(tokenizer_type, tokenizer, text):
    """Tokenize text based on tokenizer type."""
    if tokenizer_type == "sentencepiece":
        return tokenizer.encode_as_pieces(text)
    elif tokenizer_type == "huggingface":
        return tokenizer.tokenize(text)
    else:
        return []

def count_words(text):
    """Count words using whitespace splitting."""
    return len(text.split())

def evaluate_tokenizer_performance(name, tokenizer_type, tokenizer):
    """Evaluate a single tokenizer's performance."""
    print(f"\n📊 {name.upper()} EVALUATION")
    print("-" * 50)
    
    # Get vocab size
    if tokenizer_type == "sentencepiece":
        vocab_size = tokenizer.vocab_size()
    elif tokenizer_type == "huggingface":
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = "Unknown"
    
    print(f"Vocabulary size: {vocab_size:,}")
    
    all_compression = []
    all_fertility = []
    all_continued = []
    
    for lang, texts in test_texts.items():
        lang_compression = []
        lang_fertility = []
        lang_continued = []
        
        for text in texts:
            # Tokenize
            tokens = tokenize_text(tokenizer_type, tokenizer, text)
            
            # Calculate metrics
            total_chars = len(text)
            total_tokens = len(tokens)
            total_words = count_words(text)
            
            # Compression ratio
            compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
            
            # Fertility
            fertility = total_tokens / total_words if total_words > 0 else 0
            
            # Continued words (approximate for HF tokenizers)
            words = text.split()
            continued_words = 0
            
            for word in words:
                word_tokens = tokenize_text(tokenizer_type, tokenizer, word.strip('.,!?";:()'))
                if len(word_tokens) > 1:
                    continued_words += 1
            
            continued_ratio = continued_words / total_words if total_words > 0 else 0
            
            lang_compression.append(compression_ratio)
            lang_fertility.append(fertility)
            lang_continued.append(continued_ratio)
        
        # Language averages
        avg_compression = statistics.mean(lang_compression)
        avg_fertility = statistics.mean(lang_fertility)
        avg_continued = statistics.mean(lang_continued)
        
        print(f"{lang:8} | Comp: {avg_compression:.2f} | Fert: {avg_fertility:.2f} | Cont: {avg_continued:.1%}")
        
        all_compression.extend(lang_compression)
        all_fertility.extend(lang_fertility)
        all_continued.extend(lang_continued)
    
    # Overall averages
    overall_compression = statistics.mean(all_compression)
    overall_fertility = statistics.mean(all_fertility)
    overall_continued = statistics.mean(all_continued)
    
    print(f"{'Overall':8} | Comp: {overall_compression:.2f} | Fert: {overall_fertility:.2f} | Cont: {overall_continued:.1%}")
    
    return {
        "compression": overall_compression,
        "fertility": overall_fertility,
        "continued": overall_continued,
        "vocab_size": vocab_size
    }

def compare_tokenizers():
    """Main comparison function."""
    print("=" * 70)
    print("TOKENIZER COMPARISON: YOUR CUSTOM vs INDUSTRY LEADERS")
    print("=" * 70)
    
    tokenizers = load_tokenizers()
    
    if not tokenizers:
        print("❌ No tokenizers loaded successfully!")
        return
    
    results = {}
    
    # Evaluate each tokenizer
    for name, (tokenizer_type, tokenizer) in tokenizers.items():
        results[name] = evaluate_tokenizer_performance(name, tokenizer_type, tokenizer)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Tokenizer':<15} {'Vocab':<8} {'Compression':<12} {'Fertility':<10} {'Continued':<10}")
    print("-" * 70)
    
    for name, metrics in results.items():
        vocab = f"{metrics['vocab_size']:,}" if isinstance(metrics['vocab_size'], int) else str(metrics['vocab_size'])
        print(f"{name:<15} {vocab:<8} {metrics['compression']:<12.2f} {metrics['fertility']:<10.2f} {metrics['continued']:<10.1%}")
    
    # Ranking analysis
    print(f"\n🏆 RANKINGS:")
    
    # Best compression (highest)
    best_comp = max(results.items(), key=lambda x: x[1]['compression'])
    print(f"Best Compression: {best_comp[0]} ({best_comp[1]['compression']:.2f})")
    
    # Best fertility (lowest)
    best_fert = min(results.items(), key=lambda x: x[1]['fertility'])
    print(f"Best Fertility:   {best_fert[0]} ({best_fert[1]['fertility']:.2f})")
    
    # Most balanced continued words (closest to 0.4-0.5)
    best_cont = min(results.items(), key=lambda x: abs(x[1]['continued'] - 0.45))
    print(f"Best Balance:     {best_cont[0]} ({best_cont[1]['continued']:.1%})")
    
    # Overall efficiency score (compression/fertility ratio)
    print(f"\n📈 EFFICIENCY SCORES (Compression/Fertility):")
    for name, metrics in results.items():
        efficiency = metrics['compression'] / metrics['fertility']
        print(f"{name:<15}: {efficiency:.2f}")

def test_specific_examples():
    """Test specific challenging examples across all tokenizers."""
    print("\n" + "=" * 70)
    print("CHALLENGING EXAMPLES COMPARISON")
    print("=" * 70)
    
    tokenizers = load_tokenizers()
    
    examples = [
        "preprocessing",
        "антидискриминационный", 
        "ақпараттандыру",
        "machine-learning",
        "сверхъестественный"
    ]
    
    for example in examples:
        print(f"\nWord: '{example}'")
        print("-" * 40)
        
        for name, (tokenizer_type, tokenizer) in tokenizers.items():
            tokens = tokenize_text(tokenizer_type, tokenizer, example)
            print(f"{name:<15}: {len(tokens)} tokens → {tokens}")

if __name__ == "__main__":
    try:
        compare_tokenizers()
        test_specific_examples()
        print("\n🎉 Comparison completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()