import functools
from typing import Any, List, Sequence
from lazyllm import LOG
from lazyllm.tools.rag import NodeTransform, DocNode
from typing import Union


class MagicPDFTransform(NodeTransform):
    """
    专门用于magic-pdf解析结果的节点转换方法
    可自定义节点转化方法
    现根据章节标题和限定长度进行节点聚合
    """
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, document: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        return 

    def batch_forward(self, documents, node_group, **kwargs):
        nodes = ConsolidationTextNodeParser.parse_nodes(documents)
        for node in nodes:
            node.excluded_embed_metadata_keys = ['bbox', 'lines']
            node._group = node_group
        return nodes


class ConsolidationTextNodeParser:
    """
    遍历 nodes，将所有非 title 类型的节点合并。
    
    metadata:
        - 有 text_level 字段的为 title
        - 有 title 字段的为正文
    """

    @classmethod
    def class_name(cls) -> str:
        return "ConsolidationTextNodeParser"

    @staticmethod
    def parse_nodes(nodes: List[DocNode], **kwargs: Any) -> List[DocNode]:
        """
        解析节点，合并非 title 类型的文本节点。
        """
        for node in nodes:
            node._metadata['bbox'] = [{'page': node.metadata['page'], 'bbox': node.metadata['bbox']}]
        grouped_nodes = ConsolidationTextNodeParser._group_nodes(nodes)
        return [node for group in grouped_nodes for node in ConsolidationTextNodeParser._merge_text_nodes(group)]

    @staticmethod
    def _group_nodes(nodes: List["DocNode"]) -> List[List["DocNode"]]:
        """
        根据 text_level 进行分组，确保每组不会超过 4096 个字符。
        """
        grouped_nodes = []
        current_group = []

        for node in nodes:
            if node.metadata.get("text_level", 0): 
                if current_group:
                    grouped_nodes.append(current_group)
                current_group = [node]
            elif len("\n\n".join(n._content for n in current_group + [node])) >= 4096:
                grouped_nodes.append(current_group)
                current_group = [node]
            else:
                current_group.append(node)

        if current_group:
            grouped_nodes.append(current_group)

        return grouped_nodes

    @staticmethod
    def _merge_text_nodes(nodes: List["DocNode"]) -> List["DocNode"]:
        """
        合并同一组中的文本节点，将内容和元数据合并到前一个节点中。
        """
        merged_nodes = []
        for node in nodes:
            if not merged_nodes:
                merged_nodes.append(node)
            else:
                last_node = merged_nodes[-1]
                last_node._content += f"\n\n{node._content}"
                
                if last_node.metadata.get("bbox"):
                    last_node.metadata["bbox"].extend(node.metadata.get("bbox", []))
                    # last_node.metadata["bbox"] = ConsolidationTextNodeParser._merge_bbox(
                    #     last_node.metadata["bbox"], node.metadata.get("bbox", [])
                    # )
                
                if last_node.metadata.get("lines"):
                    last_node.metadata["lines"].extend(node.metadata.get("lines", []))

        return merged_nodes

    @staticmethod
    def _merge_bbox(top_bbox: List, bottom_bbox: List) -> List:
        """
        合并两个坐标框（bbox）。
        bbox 格式: [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
        """
        assert len(top_bbox) == 4, f"每个 bbox 必须包含 4 个值\n top_bbox: {top_bbox}"
        if len(bottom_bbox) != 4:
            bottom_bbox = top_bbox

        assert top_bbox[0] <= top_bbox[2] and top_bbox[1] <= top_bbox[3], f"top_bbox 格式不正确：{top_bbox}"
        assert bottom_bbox[0] <= bottom_bbox[2] and bottom_bbox[1] <= bottom_bbox[3], "bottom_bbox 格式不正确"

        # 确保 top_bbox 的 y 坐标较小
        if top_bbox[1] > bottom_bbox[1]:
            # LOG.warning(f"BBox 顺序需要检查：交换 top_bbox: {top_bbox} 和 bottom_bbox: {top_bbox}")
            top_bbox, bottom_bbox = bottom_bbox, top_bbox

        # 合并两个 bbox
        new_bbox = [
            max(top_bbox[0], bottom_bbox[0]),  # left_top_x
            top_bbox[1],  # left_top_y
            max(top_bbox[2], bottom_bbox[2]),  # right_bottom_x
            bottom_bbox[3],  # right_bottom_y
        ]
        return new_bbox