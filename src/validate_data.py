"""检查训练/验证/测试标签、ID、句子对是否重复。只报警，不自动修改划分。"""
import argparse
import json
from collections import Counter


def read(path):
    result = []
    for number, line in enumerate(open(path, encoding='utf-8'), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not all(k in obj for k in ('id', 'sentence1', 'sentence2', 'label')):
            raise ValueError(f'{path} 第 {number} 行缺字段')
        if not isinstance(obj['sentence1'], str) or not isinstance(obj['sentence2'], str):
            raise ValueError(f'{path} 第 {number} 行句子必须是字符串')
        if type(obj['label']) is not int or obj['label'] not in (0, 1):
            raise ValueError(f'{path} 第 {number} 行 label 必须是整数 0/1')
        result.append(obj)
    if not result:
        raise ValueError(f'空文件：{path}')
    return result


def key(item):
    # 中文句子对的语义匹配通常是对称的，也检查互换句子顺序的重叠。
    return tuple(sorted((item['sentence1'].strip(), item['sentence2'].strip())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--test', required=True)
    a = p.parse_args()
    parts = {name: read(path) for name, path in [('train', a.train), ('val', a.val), ('test', a.test)]}
    for name, rows in parts.items():
        ids = [str(r['id']) for r in rows]
        dist = Counter(r['label'] for r in rows)
        duplicated = len(ids) - len(set(ids))
        print(f'{name}: 样本={len(rows)}; 标签分布={dict(dist)}; 重复 ID={duplicated}')
        if duplicated:
            raise ValueError(f'{name} 内部有重复 ID，需要先联系数据负责人')
    names = list(parts)
    for i, left in enumerate(names):
        lkeys = set(map(key, parts[left]))
        for right in names[i+1:]:
            overlap = len(lkeys.intersection(map(key, parts[right])))
            print(f'{left} / {right} 重叠句子对={overlap}')
            if overlap:
                print('警告：跨集合存在相同或逆序的句子对；请组长核查并协调，不要各自私改划分。')
    print('数据结构检查完成。')


if __name__ == '__main__':
    main()
