#实现高效的Byte-Level BPE分词器算法
import regex as re
import os

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


def pre_tokenization(input_path:str | os.PathLike, regular:str
                     ) -> tuple[dict[tuple[bytes, ...], int],
                                dict[tuple[bytes, bytes], int]]:
    """
    预分词 + pair_merges
    参数：文本路径，正则化规则
    返回：字节级别得单词以及对应频次（预分词字典）
    """
    pretoken = {}
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
        special_tokens_escape = [re.escape(i) for i in special_tokens]
        contents = re.split("|".join(special_tokens_escape),content)    #"low<|endoftext|>lower" -> ['low','lower']

        for split_content in contents:
        #将单词转化为字节序列并在全文中统计该序列出现次数
            for match in re.finditer(regular, split_content):
                word = match.group()    # "low"
                if word not in special_tokens:
                    byte_sequence = tuple(bytes([i]) for i in word.encode('utf-8'))   # "low" -> b'low' -> (b'l', b'o', b'w')
                    if byte_sequence not in pretoken:
                        pretoken[byte_sequence] = 1
                    else:
                        pretoken[byte_sequence] += 1


        byte_merges = {}
        for byte_sequence, times in pretoken.items():
            for i in range(len(byte_sequence)-1):
                pair = (byte_sequence[i], byte_sequence[i+1])
                if pair not in byte_merges:
                    byte_merges[pair] = times
                else:
                    byte_merges[pair] += times


    return pretoken, byte_merges    # {(b'l', b'o', b'w'): 5}   {(b'l', b'o'):7,(b'o', b'w'):7}


def efficient_merge(pretoken:dict[tuple[bytes,...],int],
                    byte_merges:dict[tuple[bytes,bytes],int],
                    num_merges:int
                    ) -> list[tuple[bytes,bytes]]:
    merges = []
    for _ in range(num_merges):

        #选择最高频pair
        max_pair = (b'',b'')
        max_times = 0
        
        for pair, times in byte_merges.items():
            if times > max_times:
                max_times = times
                max_pair = pair
            elif times == max_times:
                if max_pair < pair:
                    max_pair = pair

        #没有可合并的 Pair，提前退出
        if max_times == 0:
            break
        
        max_token = max_pair[0] + max_pair[1] # (b's', b't') -> b'st'
        merges.append(max_pair) #[(b's',b't')]

        #更新预分词字典的同时记录最高频bytes前后变化
        change_byte_merges = {}
        new_pretoken = {}
        for byte_sequence, times in pretoken.items():
            byte_sequence_list = list(byte_sequence)    #[b'l',b'i',b's',b't',b'e',b'n']
            i = 0
            while i < len(byte_sequence_list)-1:
                pair = (byte_sequence_list[i], byte_sequence_list[i+1]) #(b's', b't')
                if max_pair == pair:
                    if i > 0: #有前pair情况   
                        before_pair = (byte_sequence_list[i-1],byte_sequence_list[i])   #(b'i',b's')
                        change_byte_merges[before_pair] = change_byte_merges.get(before_pair,0) - times

                        new_before_pair = (byte_sequence_list[i-1], max_token)  #(b'i',b'st')
                        change_byte_merges[new_before_pair] = change_byte_merges.get(new_before_pair, 0) + times

                    if i+1 != len(byte_sequence_list)-1: #有后pair情况
                        after_pair = (byte_sequence_list[i+1],byte_sequence_list[i+2])  #(b't',b'e')
                        change_byte_merges[after_pair] = change_byte_merges.get(after_pair, 0) - times

                        new_after_pair = (max_token, byte_sequence_list[i+2])   #(b'st',b'e')
                        change_byte_merges[new_after_pair] = change_byte_merges.get(new_after_pair, 0) + times

                    change_byte_merges[pair] = change_byte_merges.get(pair, 0) - times #无论有无前后pair #{(b's',b't'):-7...}

                    byte_sequence_list[i:i+2] = [max_token] #更新预分词字典

                i += 1

            new_byte_sequence = tuple(byte_sequence_list)
            new_pretoken[new_byte_sequence] = times
        pretoken = new_pretoken

        #更新byte_merges,为下一次选最大值做准备
        for change_pair, times in change_byte_merges.items():
            byte_merges[change_pair] = byte_merges.get(change_pair,0) + times

            if byte_merges[change_pair] == 0:
                del byte_merges[change_pair]

    return merges

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


    