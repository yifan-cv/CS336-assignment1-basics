#实现朴素的Byte-Level BPE分词器算法
import regex as re

PAT1 = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT2 = r"\S+"

special_tokens = ["<|endoftext|>"]

def init_vocabulary(special_tokens:list[str]) -> tuple[dict[int, bytes], int]:
    """ 
    初始化词表:256个字节+特殊token
    参数：特殊token列表  
    返回：包含初始字节以及所有特殊token的词表、此刻词表的长度
    """
    
    #将初始字节加入词表
    vocabulary = {}
    vocab_length = 0
    for i in range(256):
        vocabulary[i] = bytes([i])
        vocab_length += 1

    #将特殊字节加入词表
    for token in special_tokens:
        token = token.encode("utf-8")
        vocabulary[vocab_length] = token
        vocab_length += 1
    return vocabulary, vocab_length


def pre_tokenization(input_path:str, regular:str) -> dict[tuple[bytes, ...], int]:
    """
    预分词
    参数：文本路径，正则化规则
    返回：字节级别得单词以及对应频次（预分词字典）
    """
    pretoken = {}

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

        #将单词转化为字节序列并在全文中统计该序列出现次数
        for match in re.finditer(regular, content):
            word = match.group()    # "low"
            if word not in special_tokens:
                byte_sequence = tuple(bytes([i]) for i in word.encode('utf-8'))   # "low" -> b'low' -> (b'l', b'o', b'w')
                if byte_sequence not in pretoken:
                    pretoken[byte_sequence] = 1
                else:
                    pretoken[byte_sequence] += 1

    return pretoken # {(b'l', b'o', b'w'): 5}


def merge(pretoken:dict[tuple[bytes,...],int],
          num_merges:int
        ) -> list[tuple[bytes,bytes]]:
    """
    合并：统计 Pair、选择最高频 Pair、更新预分词字典
    参数：预分词字典，合并次数
    返回：合并规则
    """
    merges = []
    for _ in range(num_merges):
        
        #统计 Pair：遍历预分词字典得键，统计各个pair(b'l', b'o')出现的次数
        byte_merges = {}
        for byte_sequence, times in pretoken.items():
            for i in range(len(byte_sequence)-1):
                pair = (byte_sequence[i], byte_sequence[i+1])   # (b'l', b'o')
                if pair not in byte_merges:
                    byte_merges[pair] = times   # {(b'l', b'o'):7,(b'o', b'w'):7}
                else:
                    byte_merges[pair] += times

        #选择最高频 Pair：遍历刚刚生成得pair字典，选择出现次数高且字典序大得pair作为合并规则
        max_times = 0
        for pair, times in byte_merges.items():
            if times > max_times:
                max_times = times
                max_pair = pair     # max_pair:(b'l', b'o')
            elif times == max_times:
                if max_pair < pair:
                    max_pair = pair
        max_token = max_pair[0] + max_pair[1] # (b'l', b'o') -> b'lo'
        merges.append(max_pair) #[(b'l',b'o')]

        #更新预分词字典：将选中得pair合并为一个token(b'st')后更新预分词词典，为下一次合并做准备
        new_pretoken = {}
        for byte_sequence , times in pretoken.items():
            byte_sequence_list = list(byte_sequence)    # [b'l', b'o', b'w']
            i = 0
            while i < len(byte_sequence_list)-1:
                pair = (byte_sequence_list[i], byte_sequence_list[i+1])
                if max_pair == pair:
                    byte_sequence_list[i:i+2] = [max_token]
            
                i += 1

            new_byte_sequence = tuple(byte_sequence_list)   # (b'lo', b'w')
            new_pretoken[new_byte_sequence] = times
        pretoken = new_pretoken


    return merges   #[(b'l',b'o')]

def update_vocabulary(merges:list[tuple[bytes,bytes]],
                      vocabulary:dict[int, bytes],
                      vocab_length:int) -> tuple[dict[int, bytes], int]:
    """
    根据合并规则更新词表
    参数：合并规则，词表，此刻表长
    返回：更新后的词表以及表长
    """
    for pair in merges:
        token = pair[0] + pair[1]
        vocabulary[vocab_length] = token
        vocab_length += 1
    return vocabulary, vocab_length

    

# a,b = init_vocabulary(special_tokens)
# print(a)

# input = r"/home/coke/cs336/assignment1-basics/tokenizer/tinyword.txt"
# c = pre_tokenization(input, PAT2)
# print(c)

# d = merge(c,4)
# print(d)

# e, f = update_vocabulary(d,a,b)
# print(e)
# print(f)

