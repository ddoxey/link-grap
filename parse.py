#!/usr/bin/env python3
import os
import sys
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint, pformat

PARSE = f'{os.environ["HOME"]}/link-4.1b/parse'
PNG_FILE = f'{os.environ["HOME"]}/Desktop/link-parser.png'

"""
   +-----------------------Xp-----------------------+
   |       +--------Ix-------+                      |
   |       +----SIs---+      |    +-----Osn----+    |
   +---Qd--+    +-D*u-+      +-MVp+-J-+   +-Ds-+    |
   |       |    |     |      |    |   |   |    |    |
EFT-WALL may.v the force.n be.v with you my child.n .


[8]                                     Xp
                                        / \
[7]    (l)------------------------------+ +----------------------------------------(r)
        |                                                                           |
[6]     |                    Ix                                                     |
        |                    / \                                                    |
[5]     |        (l)---------+ +--------(r)                                         |
        |         |                      |                                          |
[4]     |         |      SIs             |                        Osn               |
        |         |      / \             |                        / \               |
[3]     |        (l)-----+ +----(r)      |          (l)-----------+ +---------(r)   |
        |         |              |       |           |                         |    |
[2]     |   Qd    |        D*u   |       |   MVp     |      J            Ds    |    |
        |   / \   |        / \   |       |   / \     |     / \           / \   |    |
[1]    (l)--+ +--(r)  (l)--+ +--(r)     (l)--+ +-(r) | (l)-+ +--(r) (l)--+ +--(r)   |
        |         |    |         |       |         \ | /         |   |         |    |
[0] LEFT-WALL   may.v the     force.n   be.v        with        you  my     child.n .


        Nodes:

            Grammar  - link type
            Left     - left edge
            Right    - right edge
            Language - human language leaf node
"""

class Node:
    """
        Each node is based on the actual coordinates of the element
        parsed from the raw text of the link-parse ouput text.
        The line number translates to y, the column numbers become x1
        for the left edge, x for the center, x2 for the right edge.
        The tier number is calculated later as the edges are added
        later as appearing in between the tiers with Grammar and
        Language nodes.
    """
    def __init__(self, element):
        self.text, self.pos = element.text, element.pos
        self.y = element.y
        self.tier_n = 0
        self.x = element.x
        self.x1 = element.x1
        self.x2 = element.x2

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
        coords = f'{self.tier_n}:{self.y}:{self.x1},{self.x},{self.x2}'
        if hasattr(self, 'pos') and len(self.pos):
            return f'{self.__hash__()}:{self.text},{self.pos}|{coords}'
        if hasattr(self, 'text'):
            return f'{self.__hash__()}:{self.text}|{coords}'
        return f'{self.__hash__()}:{coords}'

    @property
    def id(self):
        """
            The id property makes a convenient unique key to identify nodes
            and edges on the graph.
        """
        return str(round(float(f'{self.tier_n:02d}{self.y:02d}{self.x:3.3f}')))

    def matches(self, where):
        """
            This function returns a bool depending on whether or not the Node
            matches the given where clause.
        """
        def cmp(op, ref_val, val):
            match op:
                case '$eq':
                    return float(val) == float(ref_val)
                case '$ne':
                    return float(val) != float(ref_val)
                case '$lte':
                    return float(val) <= float(ref_val)
                case '$gte':
                    return float(val) >= float(ref_val)
                case '$gt':
                    return float(val) > float(ref_val)
                case '$lt':
                    return float(val) < float(ref_val)
                case '$between':
                    return float(ref_val[0]) <= float(val) <= float(ref_val[1])
                case _:
                    raise Exception(f'Unrecognized operator: {op}')
            return False
        for field, condition in where.items():
            op, reference_value = list(condition.items())[0]
            match field:
                case 'y':
                    if not cmp(op, reference_value, self.y):
                        return False
                case 'x1':
                    if not cmp(op, reference_value, self.x1):
                        return False
                case 'x':
                    if not cmp(op, reference_value, self.x):
                        return False
                case 'x2':
                    if not cmp(op, reference_value, self.x2):
                        return False
                case 'tier_n':
                    if not cmp(op, reference_value, self.tier_n):
                        return False
                case 'id':
                    if not cmp(op, reference_value, self.id):
                        return False
                case _:
                    raise Exception(f'Unrecognized field: {field}')
        return True


class Language(Node):
    """
        The Language Nodes represents the human readable language
        node (based on the original text) and will appear as leaf
        nodes on the grap.
    """
    def __init__(self, element):
        super().__init__(element)
    def __repr__(self):
        return f'{self.__class__.__name__}({super().__str__()})'
    def __lt__(self, other):
        return super().__lt__(other)

class Grammar(Node):
    """
        The Grammar Node represents link-parser relationship nodes.
    """
    def __init__(self, element):
        super().__init__(element)
    def __repr__(self):
        return f'{self.__class__.__name__}({super().__str__()})'
    def __lt__(self, other):
        return super().__lt__(other)

class Left(Node):
    """
        The Left and Right Nodes represent the connections between
        the Grammar and Language nodes and will ultimately be the edges
        in the graph.
    """
    def __init__(self, **kwargs):
        self.tier_n  = kwargs['tier_n']
        self.x       = kwargs['x']
        self.y       = kwargs['y']
        self.from_id = kwargs['from_id']
        self.to_id   = kwargs['to_id']
    def __str__(self):
        return f'{self.__class__.__name__}({super().__str__()})'

class Right(Left):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                        y - line number
                        x1 - the leading column number
                        x2 - the last column number
        """
        if len(conf['text']) > 1 and '.' in conf['text']:
            self.text, self.pos = conf['text'].split('.', 1)
        else:
            self.text, self.pos = conf['text'], ""
        self.flag = ""
        if '[' in self.text:
            self.text, self.flag = self.text.split('[', 1)
            self.flag = self.flag.rstrip(']')
        self.y, self.x1, self.x2 = conf['coords']
        self.x = self.x1 + ( self.x2 - self.x1 ) / 2

    def __repr__(self):
        return f'[{self.y},{self.x1},{self.x2}]' \
               '( ' \
               f'{self.text},{self.flag},{self.pos}' \
               ' )'

class NodeDB:
    """
        Temporary database of Nodes parsed from the link-parser text diagram.
    """
    def __init__(self):
        self.db = set()
        self.depth = 0
        self.tier_ns = set()
    @property
    def tiers(self):
        """
            The tiers property is a sorted list of tier numbers
            represented in the Node database.
            The tier the vertical row that a node appears in. The even
            tiers will contain Language and Grammar nodes. The odd
            number tiers will contain the Left and Right relationship
            nodes.
        """
        return sorted(list(self.tier_ns))
    def add(self, node):
        """
            When a node is added to the database its tier
            number is set.
        """
        if node.tier_n == 0:
            if len(self.tier_ns):
                node.tier_n = max(list(self.tier_ns))
            if node.y > self.depth:
                node.tier_n += 2
        self.depth = max(node.y, self.depth)
        self.tier_ns.add(node.tier_n)
        self.db.add(node)
    def extend(self, nodes):
        """
            Add a collection of nodes in a single whack.
        """
        for node in nodes:
            self.add(node)
    def nodes(self):
        """
            Produce an iterator of "node" Nodes, which only includes
            the Language and Grammar nodes and excluces Left and Right Nodes.
        """
        for node in sorted(list(self.db)):
            if not isinstance(node, Left) and not isinstance(node, Right):
                yield node
    def edges(self):
        """
            Produce an iterator of "edge" Nodes, which only includes
            the Left and Right Nodes.
        """
        for node in sorted(list(self.db)):
            if isinstance(node, Left) or isinstance(node, Right):
                yield node
    def query(self, where):
        """
            Produce an iterator of Nodes that match the given where clause.
        """
        for node in self.db:
            if node.matches(where):
                yield node
    def count(self, where):
        """
            Return a count of the number of nodes matching.
        """
        return len(list(self.query(where)))
    def __repr__(self):
        """
           Produce a sparse stringification of the database for debug purposes.
        """
        output = []
        for node in sorted(self.db):
            record = {
                'tier_n': node.tier_n,
                'x': node.x,
                'y': node.y,
                'type': node.__class__.__name__
            }
            if hasattr(node, 'text'):
                record['text'] = node.text
            if hasattr(node, 'x1'):
                record['x1'] = node.x1
            if hasattr(node, 'x2'):
                record['x2'] = node.x2
            output.append(record)
        return pformat(output, width=120, sort_dicts=True, compact=True)


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
        indent = 0
        tokens = line.split(' ')
        for token_n, token in enumerate(tokens):
            from_i = indent + token_n
            to_i = from_i + len(token) - 1
            elements.append(Element(
                                text   = token,
                                coords = [line_n, from_i, to_i]
                            ))
            indent += len(token)
        return elements
    tokens = line.split('+')
    if len(tokens) > 0:
        indent = 0
        for token_n, token in enumerate([t for t in tokens
                                         if len(t) > 0]):
            indent += len(token)
            if len(token.strip()) == 0:
                continue
            label = token.replace('-', "")
            from_i = indent - len(token) + token_n - 1
            to_i = from_i + len(token) + 1
            elements.append(Element(
                                text = label,
                                coords = [line_n, from_i, to_i]
                            ))
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
        elements = extract_elements(line_n, line)
        if elements is None:
            continue
        element_lists.append(elements)
    db = NodeDB()
    for row_n, elements in enumerate(element_lists):
        for element in elements:
            if row_n == 0:
                db.add(Language(element))
            else:
                db.add(Grammar(element))

    edge_nodes = []

    for node in db.query({'tier_n': {'$eq': 0}}):
        for left_ancestor in db.query(
            {'tier_n': {'$gt': 0},
                 'x1': {'$between': [node.x - 1, node.x + 1]}}):
            edge_nodes.append(Left(tier_n  = left_ancestor.tier_n - 1,
                                   x       = node.x,
                                   y       = left_ancestor.y,
                                   from_id = left_ancestor.id,
                                   to_id   = node.id))
        for right_ancestor in db.query(
            {'tier_n': {'$gt': 0},
                 'x2': {'$between': [node.x - 1, node.x + 1]}}):
            edge_nodes.append(Right(tier_n  = right_ancestor.tier_n - 1,
                                    x       = node.x,
                                    y       = right_ancestor.y,
                                    from_id = right_ancestor.id,
                                    to_id   = node.id))
    db.extend(edge_nodes)

    return db


def linkgrammar_layout(db):
    """
        Produces a map of nodes to (x,y) positions
        for use with networkx.draw().
    """
    pos = {}
    labels = {}

    for node in db.nodes():
        pos[node] = [node.x, node.tier_n]

    for edge in db.edges():
        a = next(db.query({'id': {'$eq': edge.from_id}}))
        b = next(db.query({'id': {'$eq': edge.to_id}}))
        labels[(a, b)] = edge.__class__.__name__[0]

    return pos, labels

def generate_graph(db, labels):
    """
        Produce a networkx.DiGraph for the given NodeDB.
    """
    dg = nx.DiGraph()

    for node in db.nodes():
        dg.add_node(node)

    for (n1, n2), _ in labels.items():
        dg.add_edge(n1, n2)

    return dg


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

    print(db)

    pos, labels = linkgrammar_layout(db)

    dg = generate_graph(db, labels)

    nx.draw(dg, pos,
            with_labels = True,
            node_size   = 1500,
            font_size   = 8,
            font_weight = 'bold')
    nx.draw_networkx_edge_labels(dg, pos, edge_labels=labels)

    plt.savefig(PNG_FILE)

    return 0


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f'USAGE: {sys.argv[0]} <sentence>', file=sys.stderr)
        sys.exit(1)
    sys.exit(parse(sys.argv[1:]))
