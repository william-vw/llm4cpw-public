import pandas as pd
import networkx as nx
import py_bipartite_matching as pbm

# https://py-bipartite-matching.readthedocs.io/en/latest/
# https://github.com/FranciscoMoretti/py_bipartite_matching

def get_bipartite_matchings(matches, get_all=False):
    """
    Gets bipartite mappings based on the given matches.
    :param matches: [ ( n_G1, n_G2, [..] ), ( n_G1, n_G2, [..] ), .. ]
    :param get_all: whether all possible bipartite mappings are returned, or only the maximal one
    :return: list of matches, i.e., [ [ ( n_G1, n_G2, [..] ), ( n_G1, n_G2, [..] ), .. ] ]
    """

    B = nx.Graph()

    # ensure unique names for both bipartite's
    B.add_nodes_from([ f"{m[0]}_0" for m in matches ], bipartite=0)
    B.add_nodes_from([ f"{m[1]}_1" for m in matches ], bipartite=1)

    # allow for round-tripping
    or_edge_map = { ( f"{m[0]}_0", f"{m[1]}_1" ): m for m in matches }
    edges = list(or_edge_map.keys())
    B.add_edges_from(edges)

    if get_all:
        matchings = pbm.enum_maximum_matchings(B)
    else: 
        top_nodes = [ n for n in B.nodes if B.nodes[n]['bipartite'] == 0 ]
        matchings = [ nx.bipartite.maximum_matching(B, top_nodes) ]

    return [ [ or_edge_map[(n0, n1)] for n0, n1 in matching.items() if (n0, n1) in or_edge_map ] for matching in matchings ]


if __name__ == "__main__":
    best_matches = [('StartEvent_1 ()', 'StartEvent_1 ()', 1), ('Gateway_0x1ins9 (Exclusive Gateway)', 'Gateway_0x1ins9 (Exclusive Gateway)', 1.0), ('Gateway_0x1ins9 (Exclusive Gateway)', 'Gateway_1004qyo (Exclusive Gateway)', 1.0), ('Activity_0hgpgfv (A)', 'Activity_0hgpgfv (A)', 1), ('Activity_0ylqqik (B)', 'Activity_0ylqqik (B)', 1), ('Gateway_0p0dm0x (Exclusive Merge)', 'Gateway_0p0dm0x (Exclusive Merge)', 1.0), ('Gateway_0p0dm0x (Exclusive Merge)', 'Gateway_079an4i (Exclusive Merge)', 1.0), ('Event_0vqis76 ()', 'Event_0vqis76 ()', 1), ('Gateway_1004qyo (Exclusive Gateway)', 'Gateway_0x1ins9 (Exclusive Gateway)', 1.0), ('Gateway_1004qyo (Exclusive Gateway)', 'Gateway_1004qyo (Exclusive Gateway)', 1.0), ('Activity_0c2bvi7 (C)', 'Activity_0c2bvi7 (C)', 1), ('Activity_0istgcx (D)', 'Activity_0istgcx (D)', 1), ('Gateway_079an4i (Exclusive Merge)', 'Gateway_0p0dm0x (Exclusive Merge)', 1.0), ('Gateway_079an4i (Exclusive Merge)', 'Gateway_079an4i (Exclusive Merge)', 1.0)]

    # best_matches = [('ActivityB', 'Activity2', 1.0), ('ActivityB', 'Activity3', 1.0), ('ActivityC', 'Activity2', 1.0), ('ActivityC', 'Activity3', 1.0)]

    df = pd.DataFrame(best_matches, columns=['left', 'right', 'score'])
    print("duplicates:\n", df[df['left'].duplicated(keep=False)])

    matches = get_bipartite_matchings(best_matches, get_all=True)
    print("num matches:", len(matches))
    # for match in matches:
    #     print(match)
    #     print()