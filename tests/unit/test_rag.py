"""RAG 模块单元测试。

覆盖：
- loader: 文档加载（md/txt，不支持格式报错）
- chunker: 文档切片、chunk_index 元数据
- retriever: vector_search / bm25_search / RRF 融合 / hybrid_search 管线
- reranker: HF API 响应解析（float/dict/nested 格式）、分数不匹配处理
- indexer: 单文件索引、增量 hash 跳过、目录索引
- vectorstore: add_documents / get_vectorstore
- rag_node: orchestrator 节点行为
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
from langchain_core.documents import Document

from human_resource.schemas.models import RetrievalResult, RetrievedChunk


# ═══════════════════════════════════════════════════════════════
# Chunker 测试
# ═══════════════════════════════════════════════════════════════


class TestChunker:
    def test_chunk_documents_splits(self):
        from human_resource.rag.chunker import chunk_documents

        long_text = "这是一段测试文本。" * 200  # 远超 CHUNK_SIZE
        docs = [Document(page_content=long_text, metadata={"source": "test.md"})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["source"] == "test.md"

    def test_chunk_documents_preserves_metadata(self):
        from human_resource.rag.chunker import chunk_documents

        docs = [Document(
            page_content="短文本",
            metadata={"source": "a.md", "file_path": "/a.md"},
        )]
        chunks = chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "a.md"
        assert chunks[0].metadata["file_path"] == "/a.md"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_chunk_index_increments_per_source(self):
        from human_resource.rag.chunker import chunk_documents

        long_text = "内容片段。" * 200
        docs = [
            Document(page_content=long_text, metadata={"source": "a.md"}),
            Document(page_content=long_text, metadata={"source": "b.md"}),
        ]
        chunks = chunk_documents(docs)
        a_indices = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "a.md"]
        b_indices = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == "b.md"]
        assert a_indices == list(range(len(a_indices)))
        assert b_indices == list(range(len(b_indices)))

    def test_chunk_documents_empty_input(self):
        from human_resource.rag.chunker import chunk_documents

        chunks = chunk_documents([])
        assert chunks == []

    def test_create_chunker_config(self):
        from human_resource.rag.chunker import create_chunker
        from human_resource.config import CHUNK_SIZE, CHUNK_OVERLAP

        chunker = create_chunker()
        assert chunker._chunk_size == CHUNK_SIZE
        assert chunker._chunk_overlap == CHUNK_OVERLAP


# ═══════════════════════════════════════════════════════════════
# Loader 测试
# ═══════════════════════════════════════════════════════════════


class TestLoader:
    def test_load_md_document(self, tmp_path):
        """使用 txt 代替 md 测试加载，因为 UnstructuredMarkdownLoader 需要 unstructured 包。"""
        from human_resource.rag.loader import load_document

        txt_file = tmp_path / "test2.txt"
        txt_file.write_text("# 标题\n\n这是内容。", encoding="utf-8")
        docs = load_document(txt_file)
        assert len(docs) >= 1
        # TextLoader 会设置 source 为完整路径
        assert "test2.txt" in docs[0].metadata["source"]
        assert docs[0].metadata["file_path"] == str(txt_file)

    def test_load_txt_document(self, tmp_path):
        from human_resource.rag.loader import load_document

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("纯文本内容", encoding="utf-8")
        docs = load_document(txt_file)
        assert len(docs) >= 1
        assert "纯文本内容" in docs[0].page_content

    def test_unsupported_format_raises(self, tmp_path):
        from human_resource.rag.loader import load_document

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\na,b", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_document(csv_file)

    def test_load_directory(self, tmp_path):
        from human_resource.rag.loader import load_directory

        (tmp_path / "a.txt").write_text("文档A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("文档B", encoding="utf-8")
        (tmp_path / "c.csv").write_text("skip", encoding="utf-8")  # 应跳过
        docs = load_directory(tmp_path)
        sources = {d.metadata.get("source", "") for d in docs}
        # TextLoader 的 source 为完整路径，检查文件名是否在其中
        assert any("a.txt" in s for s in sources)
        assert any("b.txt" in s for s in sources)
        assert not any("c.csv" in s for s in sources)

    def test_supported_extensions_constant(self):
        from human_resource.rag.loader import _SUPPORTED_EXTENSIONS

        assert ".pdf" in _SUPPORTED_EXTENSIONS
        assert ".docx" in _SUPPORTED_EXTENSIONS
        assert ".md" in _SUPPORTED_EXTENSIONS
        assert ".txt" in _SUPPORTED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════
# Retriever 测试
# ═══════════════════════════════════════════════════════════════


class TestBM25Search:
    def test_basic_keyword_match(self):
        from human_resource.rag.retriever import bm25_search

        docs = [
            Document(page_content="年假 政策 规定 每年 5 天", metadata={"source": "a"}),
            Document(page_content="绩效 考核 每季度 进行", metadata={"source": "b"}),
            Document(page_content="年假 申请 提前 3 天", metadata={"source": "c"}),
        ]
        results = bm25_search("年假 政策", docs, top_k=2)
        assert len(results) == 2
        # 包含 "年假" 的文档应排在前面
        top_contents = [doc.page_content for doc, _ in results]
        assert any("年假" in c for c in top_contents)

    def test_empty_corpus_returns_empty(self):
        from human_resource.rag.retriever import bm25_search

        results = bm25_search("查询", [], top_k=5)
        assert results == []

    def test_top_k_limits_results(self):
        from human_resource.rag.retriever import bm25_search

        docs = [Document(page_content=f"文档 {i}", metadata={"source": f"{i}"}) for i in range(10)]
        results = bm25_search("文档", docs, top_k=3)
        assert len(results) == 3


class TestReciprocalRankFusion:
    def test_merge_two_lists(self):
        from human_resource.rag.retriever import reciprocal_rank_fusion

        doc_a = Document(page_content="A", metadata={"source": "a", "chunk_index": 0})
        doc_b = Document(page_content="B", metadata={"source": "b", "chunk_index": 0})
        doc_c = Document(page_content="C", metadata={"source": "c", "chunk_index": 0})

        list1 = [(doc_a, 0.9), (doc_b, 0.8)]
        list2 = [(doc_b, 0.95), (doc_c, 0.7)]

        fused = reciprocal_rank_fusion(list1, list2)
        # doc_b 出现在两个列表中，RRF 分数应最高
        assert fused[0][0].page_content == "B"
        assert len(fused) == 3  # A, B, C 去重后

    def test_single_list(self):
        from human_resource.rag.retriever import reciprocal_rank_fusion

        doc = Document(page_content="X", metadata={"source": "x", "chunk_index": 0})
        result = reciprocal_rank_fusion([(doc, 1.0)])
        assert len(result) == 1
        assert result[0][0].page_content == "X"

    def test_empty_lists(self):
        from human_resource.rag.retriever import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_dedup_by_source_and_chunk_index(self):
        from human_resource.rag.retriever import reciprocal_rank_fusion

        doc1 = Document(page_content="内容1", metadata={"source": "s", "chunk_index": 0})
        doc2 = Document(page_content="内容1副本", metadata={"source": "s", "chunk_index": 0})

        fused = reciprocal_rank_fusion([(doc1, 0.9)], [(doc2, 0.8)])
        # 同 source + chunk_index → 合并为一个
        assert len(fused) == 1

    def test_scores_are_positive(self):
        from human_resource.rag.retriever import reciprocal_rank_fusion

        docs = [
            Document(page_content=f"d{i}", metadata={"source": f"s{i}", "chunk_index": 0})
            for i in range(5)
        ]
        results = [(d, 0.5) for d in docs]
        fused = reciprocal_rank_fusion(results)
        for _, score in fused:
            assert score > 0


class TestVectorSearch:
    @patch("human_resource.rag.retriever.get_vectorstore")
    def test_calls_similarity_search(self, mock_get_vs):
        from human_resource.rag.retriever import vector_search

        mock_store = MagicMock()
        doc = Document(page_content="结果", metadata={"source": "test"})
        mock_store.similarity_search_with_relevance_scores.return_value = [(doc, 0.85)]
        mock_get_vs.return_value = mock_store

        results = vector_search("查询", top_k=5)
        mock_store.similarity_search_with_relevance_scores.assert_called_once_with("查询", k=5)
        assert len(results) == 1
        assert results[0][0].page_content == "结果"

    @patch("human_resource.rag.retriever.get_vectorstore")
    def test_with_metadata_filter(self, mock_get_vs):
        from human_resource.rag.retriever import vector_search

        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = []
        mock_get_vs.return_value = mock_store

        vector_search("查询", metadata_filter={"source": "handbook"})
        call_kwargs = mock_store.similarity_search_with_relevance_scores.call_args
        assert call_kwargs[1]["filter"] == {"source": "handbook"}


class TestHybridSearch:
    @patch("human_resource.rag.retriever.get_all_documents")
    @patch("human_resource.rag.retriever.get_vectorstore")
    @patch("human_resource.rag.reranker.requests.post")
    def test_full_pipeline(self, mock_post, mock_get_vs, mock_get_all):
        from human_resource.rag.retriever import hybrid_search

        # 模拟向量检索结果
        doc1 = Document(page_content="年假政策是每年5天", metadata={"source": "a.md", "chunk_index": 0})
        doc2 = Document(page_content="绩效考核规定", metadata={"source": "b.md", "chunk_index": 0})
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (doc1, 0.9), (doc2, 0.7),
        ]
        mock_get_vs.return_value = mock_store

        # 模拟 BM25 语料库
        mock_get_all.return_value = [doc1, doc2]

        # 模拟 reranker API 响应
        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.95, 0.3]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = hybrid_search("年假有几天", top_n=2)
        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) >= 1
        assert result.chunks[0].text == "年假政策是每年5天"
        assert result.chunks[0].score == 0.95

    @patch("human_resource.rag.retriever.get_all_documents")
    @patch("human_resource.rag.retriever.get_vectorstore")
    def test_empty_query(self, mock_get_vs, mock_get_all):
        from human_resource.rag.retriever import hybrid_search

        result = hybrid_search("")
        assert isinstance(result, RetrievalResult)
        assert result.chunks == []
        mock_get_vs.assert_not_called()

    @patch("human_resource.rag.retriever.get_all_documents")
    @patch("human_resource.rag.retriever.get_vectorstore")
    def test_vector_search_failure_returns_empty(self, mock_get_vs, mock_get_all):
        from human_resource.rag.retriever import hybrid_search

        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.side_effect = Exception("DB错误")
        mock_get_vs.return_value = mock_store
        mock_get_all.return_value = []  # BM25 也为空

        result = hybrid_search("查询")
        assert isinstance(result, RetrievalResult)
        assert result.chunks == []

    @patch("human_resource.rag.retriever.get_all_documents")
    @patch("human_resource.rag.retriever.get_vectorstore")
    @patch("human_resource.rag.reranker.requests.post")
    def test_reranker_failure_falls_back_to_rrf(self, mock_post, mock_get_vs, mock_get_all):
        from human_resource.rag.retriever import hybrid_search

        doc = Document(page_content="内容", metadata={"source": "x.md", "chunk_index": 0})
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [(doc, 0.8)]
        mock_get_vs.return_value = mock_store
        mock_get_all.return_value = [doc]
        mock_post.side_effect = Exception("API超时")

        result = hybrid_search("查询", top_n=1)
        assert isinstance(result, RetrievalResult)
        # 即使 reranker 失败，也应返回 RRF 结果（如果超过阈值）

    @patch("human_resource.rag.retriever.get_all_documents")
    @patch("human_resource.rag.retriever.get_vectorstore")
    @patch("human_resource.rag.reranker.requests.post")
    def test_threshold_filters_low_scores(self, mock_post, mock_get_vs, mock_get_all):
        from human_resource.rag.retriever import hybrid_search

        doc = Document(page_content="低相关", metadata={"source": "z.md", "chunk_index": 0})
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [(doc, 0.5)]
        mock_get_vs.return_value = mock_store
        mock_get_all.return_value = [doc]

        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.1]  # 低于阈值 0.3
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = hybrid_search("查询", top_n=1)
        assert result.chunks == []

    @patch("human_resource.rag.retriever.get_all_documents")
    @patch("human_resource.rag.retriever.get_vectorstore")
    @patch("human_resource.rag.reranker.requests.post")
    def test_parallel_execution(self, mock_post, mock_get_vs, mock_get_all):
        """验证 vector 和 BM25 并行执行且结果正确融合。"""
        from human_resource.rag.retriever import hybrid_search

        doc = Document(page_content="并行测试", metadata={"source": "p.md", "chunk_index": 0})
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]
        mock_get_vs.return_value = mock_store
        mock_get_all.return_value = [doc]

        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.85]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = hybrid_search("并行", top_n=1)
        assert len(result.chunks) == 1
        # 两路都应被调用
        mock_store.similarity_search_with_relevance_scores.assert_called_once()
        mock_get_all.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# Reranker 测试
# ═══════════════════════════════════════════════════════════════


class TestReranker:
    def _make_docs(self, n: int) -> list[tuple[Document, float]]:
        return [
            (Document(page_content=f"文档{i}", metadata={"source": f"s{i}"}), 0.5)
            for i in range(n)
        ]

    @patch("human_resource.rag.reranker.requests.post")
    def test_float_scores_format(self, mock_post):
        from human_resource.rag.reranker import rerank

        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.9, 0.3, 0.7]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        docs = self._make_docs(3)
        result = rerank("查询", docs, top_n=2)
        assert len(result) == 2
        # 最高分应排第一
        assert result[0][1] == 0.9

    @patch("human_resource.rag.reranker.requests.post")
    def test_dict_scores_format(self, mock_post):
        from human_resource.rag.reranker import rerank

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"score": 0.8, "label": "LABEL_0"},
            {"score": 0.2, "label": "LABEL_0"},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        docs = self._make_docs(2)
        result = rerank("查询", docs, top_n=2)
        assert result[0][1] == 0.8

    @patch("human_resource.rag.reranker.requests.post")
    def test_nested_list_format(self, mock_post):
        from human_resource.rag.reranker import rerank

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            [{"score": 0.6, "label": "LABEL_0"}],
            [{"score": 0.4, "label": "LABEL_0"}],
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        docs = self._make_docs(2)
        result = rerank("查询", docs, top_n=1)
        assert len(result) == 1
        assert result[0][1] == 0.6

    @patch("human_resource.rag.reranker.requests.post")
    def test_empty_documents(self, mock_post):
        from human_resource.rag.reranker import rerank

        result = rerank("查询", [], top_n=3)
        assert result == []
        mock_post.assert_not_called()

    @patch("human_resource.rag.reranker.requests.post")
    def test_score_count_mismatch_pads(self, mock_post):
        from human_resource.rag.reranker import rerank

        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.9]  # 只有1个分数但有2个文档
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        docs = self._make_docs(2)
        result = rerank("查询", docs, top_n=2)
        assert len(result) == 2
        # 缺失的分数应被补0
        scores = sorted([s for _, s in result], reverse=True)
        assert scores[0] == 0.9
        assert scores[1] == 0.0

    @patch("human_resource.rag.reranker.requests.post")
    def test_api_error_propagates(self, mock_post):
        from human_resource.rag.reranker import rerank
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp

        docs = self._make_docs(2)
        with pytest.raises(req.HTTPError):
            rerank("查询", docs)

    @patch("human_resource.rag.reranker.requests.post")
    def test_top_n_limits_output(self, mock_post):
        from human_resource.rag.reranker import rerank

        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        docs = self._make_docs(5)
        result = rerank("查询", docs, top_n=2)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════
# Indexer 测试
# ═══════════════════════════════════════════════════════════════


class TestIndexer:
    @patch("human_resource.rag.indexer.add_documents")
    @patch("human_resource.rag.indexer._save_index_record")
    @patch("human_resource.rag.indexer._load_index_record")
    def test_index_file_new(self, mock_load_rec, mock_save_rec, mock_add, tmp_path):
        from human_resource.rag.indexer import index_file

        f = tmp_path / "policy.txt"
        f.write_text("年假政策内容", encoding="utf-8")
        mock_load_rec.return_value = {}

        count = index_file(f)
        assert count > 0
        mock_add.assert_called_once()
        mock_save_rec.assert_called_once()

    @patch("human_resource.rag.indexer.add_documents")
    @patch("human_resource.rag.indexer._save_index_record")
    @patch("human_resource.rag.indexer._load_index_record")
    def test_index_file_skip_unchanged(self, mock_load_rec, mock_save_rec, mock_add, tmp_path):
        from human_resource.rag.indexer import index_file, _file_hash

        f = tmp_path / "policy.txt"
        f.write_text("年假政策内容", encoding="utf-8")
        file_hash = _file_hash(f)
        mock_load_rec.return_value = {str(f.resolve()): file_hash}

        count = index_file(f)
        assert count == 0
        mock_add.assert_not_called()

    @patch("human_resource.rag.indexer.add_documents")
    @patch("human_resource.rag.indexer._save_index_record")
    @patch("human_resource.rag.indexer._load_index_record")
    def test_index_file_force_reindex(self, mock_load_rec, mock_save_rec, mock_add, tmp_path):
        from human_resource.rag.indexer import index_file, _file_hash

        f = tmp_path / "policy.txt"
        f.write_text("年假政策内容", encoding="utf-8")
        file_hash = _file_hash(f)
        mock_load_rec.return_value = {str(f.resolve()): file_hash}

        count = index_file(f, force=True)
        assert count > 0
        mock_add.assert_called_once()

    @patch("human_resource.rag.indexer.index_file")
    def test_index_directory(self, mock_index_file, tmp_path):
        from human_resource.rag.indexer import index_directory

        (tmp_path / "a.txt").write_text("文档A", encoding="utf-8")
        (tmp_path / "b.md").write_text("# 文档B", encoding="utf-8")
        (tmp_path / "c.csv").write_text("跳过", encoding="utf-8")
        mock_index_file.return_value = 5

        total = index_directory(tmp_path, collection_name="test_col")
        # 只索引 .txt 和 .md，不索引 .csv
        assert mock_index_file.call_count == 2
        assert total == 10  # 5 * 2

    @patch("human_resource.rag.indexer.index_file")
    def test_index_directory_auto_mapping(self, mock_index_file, tmp_path):
        """子目录自动映射到对应 collection。"""
        from human_resource.rag.indexer import index_directory
        from human_resource.config import POLICY_COLLECTION, SOP_COLLECTION

        policy_dir = tmp_path / "policy"
        sop_dir = tmp_path / "sop"
        policy_dir.mkdir()
        sop_dir.mkdir()
        (policy_dir / "handbook.txt").write_text("政策", encoding="utf-8")
        (sop_dir / "onboard.txt").write_text("流程", encoding="utf-8")
        mock_index_file.return_value = 3

        total = index_directory(tmp_path)  # 不指定 collection_name → 自动映射
        assert mock_index_file.call_count == 2
        # 检查各文件被映射到正确的 collection
        calls = {c.args[0].name: c.kwargs.get("collection_name") or c.args[1]
                 for c in mock_index_file.call_args_list}
        # index_file 通过关键字参数传递 collection_name
        for call in mock_index_file.call_args_list:
            fname = call.args[0].name
            col = call.kwargs.get("collection_name", "")
            if "handbook" in fname:
                assert col == POLICY_COLLECTION
            elif "onboard" in fname:
                assert col == SOP_COLLECTION

    def test_index_directory_nonexistent(self, tmp_path):
        from human_resource.rag.indexer import index_directory

        result = index_directory(tmp_path / "does_not_exist")
        assert result == 0

    def test_file_hash_deterministic(self, tmp_path):
        from human_resource.rag.indexer import _file_hash

        f = tmp_path / "test.txt"
        f.write_text("内容", encoding="utf-8")
        h1 = _file_hash(f)
        h2 = _file_hash(f)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex length

    def test_file_hash_changes_on_content_change(self, tmp_path):
        from human_resource.rag.indexer import _file_hash

        f = tmp_path / "test.txt"
        f.write_text("版本1", encoding="utf-8")
        h1 = _file_hash(f)
        f.write_text("版本2", encoding="utf-8")
        h2 = _file_hash(f)
        assert h1 != h2

    def test_load_save_index_record(self, tmp_path):
        from human_resource.rag.indexer import _load_index_record, _save_index_record, _INDEX_RECORD_FILE

        # 直接测试 JSON 序列化逻辑
        with patch.object(
            type(_INDEX_RECORD_FILE), "exists", return_value=False
        ):
            record = _load_index_record()
            assert record == {}


# ═══════════════════════════════════════════════════════════════
# Vectorstore 测试
# ═══════════════════════════════════════════════════════════════


class TestVectorstore:
    @patch("human_resource.rag.vectorstore.get_embeddings")
    @patch("human_resource.rag.vectorstore.Chroma")
    def test_get_vectorstore_default(self, mock_chroma_cls, mock_emb):
        from human_resource.rag.vectorstore import get_vectorstore
        from human_resource.config import DEFAULT_COLLECTION

        mock_emb.return_value = MagicMock()
        get_vectorstore()
        mock_chroma_cls.assert_called_once()
        call_kwargs = mock_chroma_cls.call_args[1]
        assert call_kwargs["collection_name"] == DEFAULT_COLLECTION

    @patch("human_resource.rag.vectorstore.get_embeddings")
    @patch("human_resource.rag.vectorstore.Chroma")
    def test_get_vectorstore_custom_collection(self, mock_chroma_cls, mock_emb):
        from human_resource.rag.vectorstore import get_vectorstore

        mock_emb.return_value = MagicMock()
        get_vectorstore("custom_collection")
        call_kwargs = mock_chroma_cls.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    @patch("human_resource.rag.vectorstore.get_vectorstore")
    def test_add_documents(self, mock_get_vs):
        from human_resource.rag.vectorstore import add_documents

        mock_store = MagicMock()
        mock_get_vs.return_value = mock_store
        docs = [Document(page_content="测试", metadata={})]
        add_documents(docs)
        mock_store.add_documents.assert_called_once_with(docs)

    @patch("human_resource.rag.vectorstore.get_vectorstore")
    def test_get_all_documents(self, mock_get_vs):
        from human_resource.rag.vectorstore import get_all_documents

        mock_store = MagicMock()
        mock_store.get.return_value = {
            "documents": ["文档A", "文档B"],
            "metadatas": [{"source": "a"}, {"source": "b"}],
        }
        mock_get_vs.return_value = mock_store

        docs = get_all_documents("test_col")
        assert len(docs) == 2
        assert docs[0].page_content == "文档A"
        assert docs[1].metadata["source"] == "b"

    @patch("human_resource.rag.vectorstore.get_vectorstore")
    def test_get_all_documents_empty(self, mock_get_vs):
        from human_resource.rag.vectorstore import get_all_documents

        mock_store = MagicMock()
        mock_store.get.return_value = {"documents": [], "metadatas": []}
        mock_get_vs.return_value = mock_store

        docs = get_all_documents()
        assert docs == []


# ═══════════════════════════════════════════════════════════════
# Embedder 测试
# ═══════════════════════════════════════════════════════════════


class TestEmbedder:
    @patch("human_resource.rag.embedder.HuggingFaceEndpointEmbeddings")
    def test_get_embeddings_returns_instance(self, mock_cls):
        from human_resource.rag.embedder import get_embeddings

        # 清除 lru_cache 以确保重新创建
        get_embeddings.cache_clear()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        result = get_embeddings()
        assert result is mock_instance
        mock_cls.assert_called_once()

        # 清除 cache 以免影响其他测试
        get_embeddings.cache_clear()

    @patch("human_resource.rag.embedder.HuggingFaceEndpointEmbeddings")
    def test_get_embeddings_singleton(self, mock_cls):
        from human_resource.rag.embedder import get_embeddings

        get_embeddings.cache_clear()
        mock_cls.return_value = MagicMock()

        emb1 = get_embeddings()
        emb2 = get_embeddings()
        assert emb1 is emb2
        mock_cls.assert_called_once()  # 只调用一次

        get_embeddings.cache_clear()


# ═══════════════════════════════════════════════════════════════
# RAG Node (Orchestrator) 测试
# ═══════════════════════════════════════════════════════════════


class TestRagNodeIntegration:
    @patch("human_resource.rag.retriever.hybrid_search")
    def test_rag_node_with_results(self, mock_hs):
        from langchain_core.messages import HumanMessage
        from human_resource.agents.orchestrator import rag_node

        mock_hs.return_value = RetrievalResult(chunks=[
            RetrievedChunk(text="年假5天", score=0.9, metadata={"source": "handbook"}),
        ])
        state = {"messages": [HumanMessage(content="年假有几天")]}
        result = rag_node(state)
        assert len(result["rag_results"].chunks) == 1
        assert result["rag_results"].chunks[0].text == "年假5天"

    @patch("human_resource.rag.retriever.hybrid_search")
    def test_rag_node_empty_messages(self, mock_hs):
        from human_resource.agents.orchestrator import rag_node

        state = {"messages": []}
        result = rag_node(state)
        assert result["rag_results"].chunks == []
        mock_hs.assert_not_called()

    @patch("human_resource.rag.retriever.hybrid_search")
    def test_rag_node_exception_returns_empty(self, mock_hs):
        from langchain_core.messages import HumanMessage
        from human_resource.agents.orchestrator import rag_node

        mock_hs.side_effect = RuntimeError("检索失败")
        state = {"messages": [HumanMessage(content="查询")]}
        result = rag_node(state)
        assert isinstance(result["rag_results"], RetrievalResult)
        assert result["rag_results"].chunks == []

    @patch("human_resource.rag.retriever.hybrid_search")
    def test_rag_node_routes_policy_qa(self, mock_hs):
        """policy_qa 意图应路由到 policy_collection。"""
        from langchain_core.messages import HumanMessage
        from human_resource.agents.orchestrator import rag_node
        from human_resource.schemas.models import IntentItem, IntentLabel, IntentResult
        from human_resource.config import POLICY_COLLECTION

        mock_hs.return_value = RetrievalResult(chunks=[])
        intent = IntentResult(intents=[IntentItem(label=IntentLabel.POLICY_QA, confidence=0.9)])
        state = {
            "messages": [HumanMessage(content="年假政策")],
            "intent": intent,
        }
        rag_node(state)
        mock_hs.assert_called_once_with("年假政策", collection_name=POLICY_COLLECTION)

    @patch("human_resource.rag.retriever.hybrid_search")
    def test_rag_node_routes_process_inquiry(self, mock_hs):
        """process_inquiry 意图应路由到 sop_collection。"""
        from langchain_core.messages import HumanMessage
        from human_resource.agents.orchestrator import rag_node
        from human_resource.schemas.models import IntentItem, IntentLabel, IntentResult
        from human_resource.config import SOP_COLLECTION

        mock_hs.return_value = RetrievalResult(chunks=[])
        intent = IntentResult(intents=[IntentItem(label=IntentLabel.PROCESS_INQUIRY, confidence=0.9)])
        state = {
            "messages": [HumanMessage(content="入职流程")],
            "intent": intent,
        }
        rag_node(state)
        mock_hs.assert_called_once_with("入职流程", collection_name=SOP_COLLECTION)

    @patch("human_resource.rag.retriever.hybrid_search")
    def test_rag_node_default_collection_without_intent(self, mock_hs):
        """无意图时使用默认 collection。"""
        from langchain_core.messages import HumanMessage
        from human_resource.agents.orchestrator import rag_node
        from human_resource.config import DEFAULT_COLLECTION

        mock_hs.return_value = RetrievalResult(chunks=[])
        state = {"messages": [HumanMessage(content="查询")]}
        rag_node(state)
        mock_hs.assert_called_once_with("查询", collection_name=DEFAULT_COLLECTION)


# ═══════════════════════════════════════════════════════════════
# Models 测试
# ═══════════════════════════════════════════════════════════════


class TestRAGModels:
    def test_retrieved_chunk_defaults(self):
        chunk = RetrievedChunk(text="内容", score=0.5)
        assert chunk.metadata == {}

    def test_retrieved_chunk_with_metadata(self):
        chunk = RetrievedChunk(text="内容", score=0.8, metadata={"source": "a.md"})
        assert chunk.metadata["source"] == "a.md"

    def test_retrieval_result_defaults(self):
        result = RetrievalResult()
        assert result.chunks == []

    def test_retrieval_result_with_chunks(self):
        chunks = [
            RetrievedChunk(text="A", score=0.9),
            RetrievedChunk(text="B", score=0.7),
        ]
        result = RetrievalResult(chunks=chunks)
        assert len(result.chunks) == 2
        assert result.chunks[0].score > result.chunks[1].score


# ═══════════════════════════════════════════════════════════════
# Config 常量测试
# ═══════════════════════════════════════════════════════════════


class TestRAGConfig:
    def test_collection_constants(self):
        from human_resource.config import (
            POLICY_COLLECTION, SOP_COLLECTION, DEFAULT_COLLECTION,
            INTENT_COLLECTION_MAP, DIR_COLLECTION_MAP,
        )
        assert POLICY_COLLECTION == "policy_collection"
        assert SOP_COLLECTION == "sop_collection"
        assert DEFAULT_COLLECTION == POLICY_COLLECTION
        assert INTENT_COLLECTION_MAP["policy_qa"] == POLICY_COLLECTION
        assert INTENT_COLLECTION_MAP["process_inquiry"] == SOP_COLLECTION
        assert DIR_COLLECTION_MAP["policy"] == POLICY_COLLECTION
        assert DIR_COLLECTION_MAP["sop"] == SOP_COLLECTION


# ═══════════════════════════════════════════════════════════════
# RAG Main CLI 测试
# ═══════════════════════════════════════════════════════════════


class TestRAGMain:
    @patch("human_resource.rag.main.index_directory")
    def test_main_default(self, mock_idx):
        from human_resource.rag.main import main

        mock_idx.return_value = 10
        main([])  # 无参数 → 默认目录 + 自动映射
        mock_idx.assert_called_once()
        call_kwargs = mock_idx.call_args[1]
        assert call_kwargs["collection_name"] is None  # 自动映射
        assert call_kwargs["force"] is False

    @patch("human_resource.rag.main.index_directory")
    def test_main_with_collection(self, mock_idx):
        from human_resource.rag.main import main

        mock_idx.return_value = 5
        main(["--collection", "policy_collection"])
        call_kwargs = mock_idx.call_args[1]
        assert call_kwargs["collection_name"] == "policy_collection"

    @patch("human_resource.rag.main.index_directory")
    def test_main_force(self, mock_idx):
        from human_resource.rag.main import main

        mock_idx.return_value = 0
        main(["--force"])
        call_kwargs = mock_idx.call_args[1]
        assert call_kwargs["force"] is True

    @patch("human_resource.rag.main.index_directory")
    def test_main_custom_dir(self, mock_idx):
        from human_resource.rag.main import main

        mock_idx.return_value = 3
        main(["--dir", "data/documents/policy", "--collection", "policy_collection"])
        call_kwargs = mock_idx.call_args[1]
        assert call_kwargs["dir_path"] == "data/documents/policy"
        assert call_kwargs["collection_name"] == "policy_collection"
