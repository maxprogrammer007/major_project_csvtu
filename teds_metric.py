"""
Standalone TEDS (Tree-Edit-Distance-based Similarity) Metric
============================================================
Reference: IBM PubTabNet (Zhong et al., 2020)
Supports both:
- TEDS-S (structure_only=True): Evaluates tree edit distance on HTML table structure (rows, cols, spans) with cell contents stripped.
- TEDS (structure_only=False): Evaluates tree edit distance on both structure and text tokens.
"""

from collections import deque
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    def rename(self, node1, node2):
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                # String character distance
                c1 = node1.content or []
                c2 = node2.content or []
                if c1 == c2:
                    return 0.0
                # Simple Levenshtein or token distance
                return 1.0
        return 0.0


class TEDS(object):
    def __init__(self, structure_only=False, ignore_nodes=None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        if not pred or not true:
            return 0.0

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        try:
            pred_doc = html.fromstring(pred, parser=parser)
            true_doc = html.fromstring(true, parser=parser)
        except Exception:
            return 0.0

        pred_tables = pred_doc.xpath("//table")
        true_tables = true_doc.xpath("//table")

        if not pred_tables or not true_tables:
            return 0.0

        pred_table = pred_tables[0]
        true_table = true_tables[0]

        if self.ignore_nodes:
            etree.strip_tags(pred_table, *self.ignore_nodes)
            etree.strip_tags(true_table, *self.ignore_nodes)

        n_nodes_pred = len(pred_table.xpath(".//*"))
        n_nodes_true = len(true_table.xpath(".//*"))
        n_nodes = max(n_nodes_pred, n_nodes_true)

        if n_nodes == 0:
            return 1.0

        tree_pred = self.load_html_tree(pred_table)
        tree_true = self.load_html_tree(true_table)

        distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
        return max(0.0, 1.0 - (float(distance) / n_nodes))
