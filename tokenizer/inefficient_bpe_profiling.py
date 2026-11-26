from bpe_tokenizer import *
import time


special_tokens = ["<|endoftext|>"]

def ineffient_bpe_profiling(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    """
    vocabulary, vocab_length = init_vocabulary(special_tokens)
    num_merges = vocab_size - vocab_length
    
    PAT1 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    pretoken_start_time = time.perf_counter()
    pretoken = pre_tokenization(input_path, PAT1)
    pretoken_end_time = time.perf_counter()

    merges_start_time = time.perf_counter()
    merges = merge(pretoken, num_merges)
    merges_end_time = time.perf_counter()

    vocabulary, vocab_length = update_vocabulary(merges, vocabulary, vocab_length)

    pretoken_time = pretoken_end_time - pretoken_start_time
    merges_time = merges_end_time - merges_start_time

    print(f"预分词所用时间：{pretoken_time:.4f}s")
    print(f"子词合并所用时间：{merges_time:.4f}s")

    return vocabulary, merges

# 数据集：tests/fixtures/corpus.en      Special Token：<|endoftext|>
# 目标词表：500 预分词所用时间：0.0257s   子词合并所用时间：0.9156s   全部耗时统计：0.9415s
# 目标词表：2000   预分词所用时间：0.0244s  子词合并所用时间：4.6810s   全部耗时统计：4.7058s
input_path1 = r"/home/coke/cs336/assignment1-basics/tests/fixtures/corpus.en"

# 数据集：/data/TinyStoriesV2-GPT4-valid.txt      Special Token：<|endoftext|>
#目标词表：500  预分词所用时间：4.2157s   子词合并所用时间：2.6446s   全部耗时统计：6.8606s
#目标词表：2000 预分词所用时间：4.2456s   子词合并所用时间：14.4485s  全部耗时统计：18.6946s
input_path2 = r"/home/coke/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"

if __name__ == "__main__":
    total_ineffient_bpe_start =  time.perf_counter()
    ineffient_bpe_profiling(input_path2, 2000, special_tokens)
    total_ineffient_bpe_end =  time.perf_counter()
    total_time = total_ineffient_bpe_end - total_ineffient_bpe_start
    print(f"全部耗时统计：{total_time:.4f}s")


