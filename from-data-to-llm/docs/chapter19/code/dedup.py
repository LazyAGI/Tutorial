import hashlib
from datasketch import MinHash, MinHashLSH

# -----------------------------
# 1. 输入数据
# -----------------------------
code_list = [
    'def add(a, b):\n    return a + b',
    'def multiply(a, b):\n    return a * b',
    'def add(a, b):\n    return a + b',
    'def add(a, b, c):\n    return a + b',
]


# -----------------------------
# 2. 精确去重 (SHA-256)
# -----------------------------
def exact_dedup(codes):
    '''精确去重，返回保留索引和删除日志'''
    seen_hashes = {}
    kept_indices = []
    removed_exact = []

    for idx, code in enumerate(codes):
        h = hashlib.sha256(code.encode('utf-8')).hexdigest()
        if h not in seen_hashes:
            seen_hashes[h] = idx
            kept_indices.append(idx)
        else:
            removed_exact.append((idx, seen_hashes[h]))  # (被删索引, 保留索引)

    return kept_indices, removed_exact


# -----------------------------
# 3. 模糊去重 (MinHash + LSH)
# -----------------------------
def get_minhash(code, num_perm=128):
    '''根据代码文本生成 MinHash 对象'''
    m = MinHash(num_perm=num_perm)
    for token in code.split():
        m.update(token.encode('utf-8'))
    return m


def fuzzy_dedup(codes, indices, threshold=0.6):
    '''基于 MinHash + LSH 的模糊去重'''
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    final_kept_indices = []
    fuzzy_removed_log = []
    minhash_store = {}

    for idx in indices:
        m = get_minhash(codes[idx])
        result = lsh.query(m)

        if result:
            match_idx = int(result[0])
            jac = m.jaccard(minhash_store[match_idx])
            fuzzy_removed_log.append((idx, match_idx, jac))
        else:
            lsh.insert(str(idx), m)
            minhash_store[idx] = m
            final_kept_indices.append(idx)

    return final_kept_indices, fuzzy_removed_log


# -----------------------------
# 4. 执行流程与结果打印
# -----------------------------
if __name__ == '__main__':
    # 执行去重
    exact_kept, exact_removed = exact_dedup(code_list)
    final_indices, fuzzy_removed = fuzzy_dedup(code_list, exact_kept)

    print('=' * 60)
    print('去重过程报告')
    print('=' * 60)

    print('\n[1/2] 精确去重阶段 (SHA-256):')
    for r_idx, k_idx in exact_removed:
        print(f'  - 移除原始索引 {r_idx}: 内容与索引 {k_idx} 完全一致')

    print('\n[2/2] 模糊去重阶段 (LSH - Threshold 0.6):')
    for r_idx, k_idx, jac in fuzzy_removed:
        print(f'  - 移除原始索引 {r_idx}: 相似度 {jac:.3f}，已保留相似样本 {k_idx}')

    print('\n' + '=' * 60)
    title = '最终保留结果清单'
    print(f'{title:^55}')
    print('-' * 60)
    # 修改重点：使用三单引号，并保持内部插值正常
    print(f'''{'原始索引':<10} | {'内容摘要 (前20位)':<25} | {'去重状态'}''')
    print('-' * 60)

    for i in range(len(code_list)):
        content = code_list[i].replace('\n', ' ')[:20] + '...'
        if i in final_indices:
            status = '✅ [保留]'
        elif any(r[0] == i for r in exact_removed):
            status = '❌ [精确重复删除]'
        else:
            status = '❌ [模糊重复删除]'

        print(f'{i:<12} | {content:<25} | {status}')

    print('-' * 60)
    print(f'统计：原始总数 {len(code_list)} -> 最终保留 {len(final_indices)}')
    print('=' * 60)
