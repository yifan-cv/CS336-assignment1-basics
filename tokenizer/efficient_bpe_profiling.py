from efficient_bpe import *
import time


special_tokens = ["<|endoftext|>"]


def efficient_bpe_profiling(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocabulary, vocab_length = init_vocabulary(special_tokens)
    num_merges = vocab_size - vocab_length

    PAT1 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 预分词 + 初始 Pair 频次统计
    pretoken_start_time = time.perf_counter()
    pretoken, byte_merges = pre_tokenization(input_path, PAT1)
    pretoken_end_time = time.perf_counter()

    # 增量 Merge
    merges_start_time = time.perf_counter()
    merges = efficient_merge(pretoken, byte_merges, num_merges)
    merges_end_time = time.perf_counter()

    # 更新词表
    vocabulary, vocab_length = update_vocabulary(
        merges, vocabulary, vocab_length
    )

    pretoken_time = pretoken_end_time - pretoken_start_time
    merges_time = merges_end_time - merges_start_time

    print(f"预分词所用时间：{pretoken_time:.4f}s")
    print(f"子词合并所用时间：{merges_time:.4f}s")

    return vocabulary, merges


# 数据集1：corpus.en
# 目标词表：500
# 目标词表：2000
input_path1 = r"/home/coke/cs336/assignment1-basics/tests/fixtures/corpus.en"


# 数据集2：TinyStories Validation
# 目标词表：500
# 目标词表：2000
input_path2 = r"/home/coke/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"


if __name__ == "__main__":

    total_efficient_bpe_start = time.perf_counter()

    efficient_bpe_profiling(input_path2, 5000, special_tokens)

    total_efficient_bpe_end = time.perf_counter()

    total_time = total_efficient_bpe_end - total_efficient_bpe_start

    print(f"全部耗时统计：{total_time:.4f}s")