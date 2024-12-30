#!/usr/bin/env python3
import os
import sys
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint, pformat

PARSE = f'{os.environ["HOME"]}/link-5.12.5/link-parser/link-parser'
PNG_FILE = f'{os.environ["HOME"]}/Desktop/link-parser.png'

r"""
    +-------------------Xp------------------+
    +------------>WV----------->+           |
    |       +---------Ix--------+           |
    |       +----SIs----+       |           |
    +-->Qd--+    +--Dmu-+       +-Pp-+-J-+  |
    |       |    |      |       |    |   |  |
LEFT-WALL may.v the force.n-u be.v with you . 

"""

class Node:
    """
        Each node is a language token, including LEFT-WALL and punctuation.
    """
    def __init__(self, element):
        self.id = element.id
        self.tier_i = 0
        self.col_i = element.col_i
        self.x = element.x
        self.source_x, self.target_x = element.get_xx()
        self.text, self.tag = element.text, element.tag
        self.label = element.text
        if self.tag is not None and len(self.tag) > 0:
            self.label += f'.{element.tag}'

    def __lt__(self, other):
        """
            The __lt__ function is used for sorting in a set()
        """
        return self.__hash__() < other.__hash__()

    def __str__(self):
        return self.text

    def __repr__(self):
        """
            The __repr__ function is used in pprint debug formatting.
        """
        coords = f'{self.tier_i}:{self.col_i}'
        if hasattr(self, 'tag') and self.tag is not None and len(self.tag) > 0:
            return f'{self.__hash__()}:{self.text},{self.tag}|{coords}'
        if hasattr(self, 'text'):
            return f'{self.__hash__()}:{self.text}|{coords}'
        return f'{self.__hash__()}:{coords}'

    def get(self, field, default=None):
        """
            Provide the value of the requested field or the given default
            value if the property is unknown.
        """
        if hasattr(self, field):
            return getattr(self, field)
        return default

    def matches(self, where):
        """
            This function returns a bool depending on whether or not the Node
            matches the given where clause.
        """
        def cmp(op, ref_val, val):
            if isinstance(ref_val, list):
                ref_val = [float(v) for v in ref_val]
            elif isinstance(ref_val, str):
                if ref_val.isdigit() or all(e.isdigit() for e in ref_val.split('.')):
                    ref_val = float(ref_val)
            if isinstance(val, str):
                if val.isdigit() or all(e.isdigit() for e in val.split('.')):
                    val = float(val)
            match op:
                case '$eq':
                    return val == ref_val
                case '$ne':
                    return val != ref_val
                case '$lte':
                    return val <= ref_val
                case '$gte':
                    return val >= ref_val
                case '$gt':
                    return val > ref_val
                case '$lt':
                    return val < ref_val
                case '$between':
                    return ref_val[0] <= val <= ref_val[1]
                case _:
                    raise Exception(f'Unrecognized operator: {op}')
            return False
        for field, condition in where.items():
            op, reference_value = list(condition.items())[0]
            if not hasattr(self, field):
                raise Exception(f'Unrecognized {__class__.__name__} field: {field}')
            if not cmp(op, reference_value, getattr(self, field)):
                return False
        return True


class Edge(Node):
    """
        The Grammar Node represents link-parser relationship nodes.
    """
    def __init__(self, element):
        super().__init__(element)
        self.source_id = None
        self.target_id = None

    def set_source(self, node):
        if self.source_x == node.x:
            self.source_id = node.id
            return True
        return False

    def set_target(self, node):
        if self.target_x == node.x:
            self.target_id = node.id
            return True
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}({super().__str__()})'

    def __lt__(self, other):
        return super().__lt__(other)


class Element:
    """
        This is an intermediate class which holds the raw text parsing
        result while parsing the link-parser text output.
    """
    def __init__(self, **conf):
        """
            Initialized on a dict of:
                text - the label given by link-parser which further
                    derives the part of speech or special flags.
                coords - the coordinates of the element in the link-parser text
                    in terms of:
                        row_N - line number
                        col_i - element number on line
        """
        self.x = -1
        self.col_i, self.row_i = conf['col_i'], conf['row_i']
        self.source_x = conf.get('source_x', None)
        self.target_x = conf.get('target_x', None)
        if len(conf['text']) > 1 and '.' in conf['text']:
            self.text, self.tag = conf['text'].split('.', 1)
        else:
            self.text, self.tag = conf['text'], None
        self.flag = ""
        if '[' in self.text:
            self.text, self.flag = self.text.split('[', 1)
            self.flag = self.flag.rstrip(']')

    @property
    def id(self):
        if self.tag is None or len(self.tag) == 0:
            return f'{self.col_i}.{self.row_i}:{self.text}'
        return f'{self.col_i}.{self.row_i}:{self.text}.{self.tag}'

    def set_x(self, x):
        self.x = x

    def get_xx(self):
        return self.source_x, self.target_x

    def __repr__(self):
        if self.source_x is None:
            if self.tag is None:
                return pformat({'id': self.id,
                                'label': self.text})
            return pformat({'id': self.id,
                            'label': self.text,
                            'tag': self.tag})
        return pformat({'id': self.id,
                        'label': self.text,
                        'source_x': self.source_x,
                        'target_x': self.target_x})


class NodeDB:
    """
        Temporary database of Nodes parsed from the link-parser text diagram.
    """
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def add(self, item):
        """
            The item added may be a Node or an Edge.
            If an Edge is added it is assumed that the source and target Nodes
            for the Edge have already been added and the source and target ID
            properties will be set upon adding them.
        """
        if isinstance(item, Edge):
            nodes = self.query({'x': {'$eq': item.source_x}})
            item.set_source(next(nodes))
            nodes = self.query({'x': {'$eq': item.target_x}})
            item.set_target(next(nodes))
            self.edges.add(item)
        elif isinstance(item, Node):
            self.nodes.add(item)
        else:
            raise ValueError(f'{item} is not a Node or an Edge')

    def query(self, where):
        """
            Produce an iterator of Nodes that match the given where clause.
        """
        for node in self.nodes:
            if node.matches(where):
                yield node

    def __repr__(self):
        """
           Produce a sparse stringification of the database for debug purposes.
        """
        return pformat(self.graph()) 

    def graph(self):
        nodes = [{'id': node.get('id', node.get('text')),
                  'label': node.get('text'),
                  'tag': node.get('tag', None)}
                 for node in sorted(self.nodes, key=lambda node: node.x)]
        edges = [{'source_id': edge.source_id,
                  'target_id': edge.target_id,
                  'label': edge.label}
                 for edge in sorted(self.edges, key=lambda edge: edge.source_x)]
        return {"nodes": nodes, "edges": edges}


def get_linkgrammar_text(sentence):
    """
        This take a raw Enlish sentence and returns the link-parser diagram
        text in response.
    """
    if not os.path.exists(PARSE):
        raise Exception(f'link parser executable not found: {PARSE}')

    special_commands = [
        'width=99999',  # prevent line wrapped diagram
    ]
    commands = '\n!'.join(special_commands)

    p = subprocess.run([PARSE],
                       input=f'!{commands}\n\n{" ".join(sentence)}',
                       cwd=os.path.dirname(PARSE),
                       encoding='UTF-8',
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)

    if p.returncode > 0:
        print(p.returncode, file=sys.stderr)
        print(p.stdout, file=sys.stderr)
        return None

    skipping = True
    output = []
    for line in p.stdout.split("\n"):
        line = line.rstrip()
        text = line.lstrip()
        if len(text) == 0:
            if len(output):
                break
            continue
        if text[0] == '+' and text[-1] == '+':
            output = []
            skipping = False
        if skipping:
            continue
        if text.startswith('Press RETURN for'):
            break
        if text.startswith('linkparser> Freeing dictionary'):
            break
        output.append(line)

    if len(output) == 0:
        return None

    return "\n".join(output)


def extract_elements(line_n, line):
    """
        For a given line from the link-parser diagram this extracts
        any elements.
    """
    elements = []
    line = line.replace('|', " ")
    if line_n == 0:
        tokens = line.split(' ')
        for token_n, token in enumerate(tokens):
            elements.append(Element(text = token, col_i = token_n, row_i = line_n))
        return elements
    tokens = line.split('+')
    if len(tokens) > 0:
        xs = [x_i for x_i, c in enumerate(line) if c == '+' for _ in (0, 1)]
        xs = [0, *xs, len(line) - 1]
        for token_n, token in enumerate([t for t in tokens
                                         if len(t) > 0]):
            source_x = xs.pop(0)
            target_x = xs.pop(0)
            if len(token.strip()) == 0:
                continue
            label = token.replace('-', "").replace('>', "")
            elements.append(Element(text = label,
                                    col_i = len(elements),
                                    row_i = line_n,
                                    source_x = source_x,
                                    target_x = target_x))
    if len(elements) == 0:
        return None
    return elements


def create_nodedb(linkage_txt):
    """
        For a given link-parser diagram this will produce a database
        of Node objects.
    """
    element_lists = []
    for line_n, line in enumerate(reversed(linkage_txt.split("\n"))):
        if line_n == 1:
            col_xs = [x_i for x_i, c in enumerate(line) if c != ' ']
            for element_i, x in enumerate(col_xs):
                element_lists[-1][element_i].set_x(x)
        else: 
            elements = extract_elements(line_n, line)
            if elements is None:
                continue
            element_lists.append(elements)
    db = NodeDB()
    for row_i, elements in enumerate(element_lists):
        for col_i, element in enumerate(elements):
            if row_i == 0:
                db.add(Node(element))
            else:
                db.add(Edge(element))

    return db


def generate_graph(db):
    """
        Produce a networkx.DiGraph for the given NodeDB.
    """
    dg = nx.DiGraph()
    graph = db.graph()

    for node in graph['nodes']:
        dg.add_node(node['id'], label=node['label'], tag=node['tag'])

    for edge in graph['edges']:
        dg.add_edge(edge['source_id'], edge['target_id'], label=edge['label'])

    pos = nx.shell_layout(dg)
    labels = nx.get_node_attributes(dg, 'label')
    edge_labels = nx.get_edge_attributes(dg, 'label')

    return dg, pos, labels, edge_labels


def parse(sentence):
    """
        Parse the given sentence and produce a PNG file.
    """
    diagram = get_linkgrammar_text(sentence)

    if diagram is None:
        print('No link grammar', file=sys.stderr)
        return 1

    print(diagram, '\n')

    db = create_nodedb(diagram)

    if db is None:
        print('No nodes found', file=sys.stderr)
        return 1

    pprint(db)

    dg, pos, labels, edge_labels = generate_graph(db)

    nx.draw(dg, pos,
            with_labels = False,
            node_size   = 3000,
            font_size   = 8,
            font_weight = 'bold',
            node_color  = 'lightblue')
    nx.draw_networkx_labels(dg, pos,
                            labels = labels,
                            font_size = 10)
    nx.draw_networkx_edge_labels(dg, pos,
                                 edge_labels = edge_labels,
                                 font_size = 10)

    plt.savefig(PNG_FILE)

    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f'USAGE: {sys.argv[0]} <sentence>', file=sys.stderr)
        sys.exit(1)
    sys.exit(parse(sys.argv[1:]))
