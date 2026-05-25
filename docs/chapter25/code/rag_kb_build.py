import os
import sys
import json
import argparse
from pathlib import Path


from lazyllm.tools.data.pipelines.kc_pipelines import (
    build_batch_chunk_generator_pipeline,
    build_batch_kbc_pipeline,
)


def collect_text_paths(sources_dir: Path) -> list:
    paths = []
    for ext in ('*.txt', '*.md'):
        paths.extend(sources_dir.glob(ext))
    return sorted(paths)


def run_chunk_pipeline(
    text_paths: list,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list:
    chunk_out = output_dir / 'chunks'
    chunk_out.mkdir(parents=True, exist_ok=True)

    ppl = build_batch_chunk_generator_pipeline(
        input_key='text_path',
        output_key='chunk_path',
        output_dir=str(chunk_out),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_method='recursive',
        tokenizer_name='bert-base-uncased',
    )
    data_list = [{'text_path': str(p)} for p in text_paths]
    results = ppl(data_list)
    return results


def run_clean_pipeline(
    chunk_paths: list,
    output_dir: Path,
    llm,
    lang: str = 'zh',
) -> list:
    clean_out = output_dir / 'cleaned'
    clean_out.mkdir(parents=True, exist_ok=True)

    ppl = build_batch_kbc_pipeline(
        input_key='chunk_path',
        output_key='cleaned_chunk_path',
        llm=llm,
        lang=lang,
        output_dir=str(clean_out),
    )
    data_list = [{'chunk_path': p} for p in chunk_paths]
    results = ppl(data_list)
    return results


def export_chunks_to_kb_dir(
    results: list,
    use_cleaned: bool,
    kb_content_dir: Path,
) -> None:
    kb_content_dir.mkdir(parents=True, exist_ok=True)

    path_key = 'cleaned_chunk_path' if use_cleaned else 'chunk_path'
    chunk_key = 'cleaned_chunk' if use_cleaned else 'raw_chunk'

    for item in results:
        path = item.get(path_key) or item.get('chunk_path')
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            data = [data]
        chunks = []
        for rec in data:
            text = rec.get(chunk_key) or rec.get('raw_chunk', '')
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        if not chunks:
            continue
        base_name = (
            Path(path).stem.replace('_chunk', '').replace('_cleaned', '')
        )
        out_file = kb_content_dir / f'{base_name}.txt'
        out_file.write_text('\n\n'.join(chunks), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description='基于 LazyLLM KBC 构建 RAG 知识库',
    )
    parser.add_argument(
        '--sources_dir',
        type=str,
        default=None,
        help='原始文本目录，默认 <output_dir>/sources',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出根目录',
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=256,
        help='分块长度（token 数）',
    )
    parser.add_argument(
        '--chunk_overlap',
        type=int,
        default=50,
        help='分块重叠',
    )
    parser.add_argument(
        '--do_clean',
        action='store_true',
        help='是否使用 LLM 做文档精炼（需配置 LazyLLM 可用 LLM）',
    )
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        choices=('zh', 'en'),
        help='清洗与 prompt 语言',
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = 'rag_kb'
    args.output_dir = Path(args.output_dir)

    if args.sources_dir is None:
        args.sources_dir = args.output_dir / 'sources'
    args.sources_dir = Path(args.sources_dir)

    if not args.sources_dir.is_dir():
        print(f'错误：源目录不存在 {args.sources_dir}')
        print('请先创建并在其中放入 .txt / .md 文件，'
              '或使用默认 rag_kb/sources 并放入示例文件。')
        sys.exit(1)

    text_paths = collect_text_paths(args.sources_dir)
    if not text_paths:
        print(f'错误：在 {args.sources_dir} 下未找到 .txt / .md 文件')
        sys.exit(1)

    print(f'源目录: {args.sources_dir}')
    print(f'输出目录: {args.output_dir}')
    print(f'找到 {len(text_paths)} 个文本文件，开始分块...')

    results = run_chunk_pipeline(
        [str(p) for p in text_paths],
        args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunk_paths = [r.get('chunk_path') for r in results if r.get('chunk_path')]
    print(f'分块完成，得到 {len(chunk_paths)} 个 chunk 文件。')

    use_cleaned = False
    import lazyllm
    if args.do_clean and chunk_paths:
        try:
            llm = lazyllm.OnlineChatModule()
            results = run_clean_pipeline(
                chunk_paths,
                args.output_dir,
                llm=llm,
                lang=args.lang,
            )
            use_cleaned = True
            print('LLM 清洗完成，将导出清洗后内容')
        except Exception as e:
            print(f'LLM 清洗跳过（{e}），将导出原始分块。')

    kb_content_dir = args.output_dir / 'kb_content'
    export_chunks_to_kb_dir(results, use_cleaned, kb_content_dir)
    print(f'已导出到: {kb_content_dir}')

    kb_path = kb_content_dir.resolve()
    doc = lazyllm.Document(dataset_path=str(kb_path))
    retriever = lazyllm.Retriever(
        doc=doc,
        group_name=lazyllm.Document.CoarseChunk,
        similarity='bm25_chinese',
        topk=3,
    )
    demo_result = retriever('玉山箭竹生长在哪里？')
    print('\n示例检索结果（首条内容摘要）:')
    if demo_result and len(demo_result) > 0:
        content = getattr(
            demo_result[0],
            'get_content',
            lambda: str(demo_result[0]),
        )()
        print(content[:200] + ('...' if len(content) > 200 else ''))
    else:
        print('(无召回)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
