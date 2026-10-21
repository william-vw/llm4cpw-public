"""
Comparing the similarity of two BPMN Graphs
Implementation for "Similarity of business process models: Metrics and evaluation"

Based originally on https://github.com/alireza-hsi/LLM4QBP
"""

from __future__ import annotations
import os

from collections import defaultdict
from itertools import chain

import pandas as pd
from Levenshtein import distance

from libs.cmp4proc.bpmn_highlighter.model import Diagram, DiagramElement
from libs.cmp4proc.bpmn_highlighter.parser import EVENT_TYPES, GATEWAY_TYPES, TASK_TYPES, BpmnSource, parse_diagram

from .utils import jaccard_simil
from .best_matches import best_matches_per_column_row
from .bipartite_matches import get_bipartite_matchings

NODE_TYPES = TASK_TYPES | EVENT_TYPES | GATEWAY_TYPES

def calculate_similarity(source1: Diagram | BpmnSource, source2: Diagram | BpmnSource, verbose=False, all_equiv_gw_evt_mappings=False, incl_gw_dir=False, simil_matrix_folder=None):
    cmp = CompareBPMN(verbose, all_equiv_gw_evt_mappings, incl_gw_dir, simil_matrix_folder)
    return cmp.calculate_similarity(source1, source2)

class CompareBPMN(object):

    def __init__(self, verbose=False, all_equiv_gw_evt_mappings=False, incl_gw_dir=False, simil_matrix_folder=None):
        self.verbose = verbose
        self.all_equiv_gw_evt_mappings = all_equiv_gw_evt_mappings
        self.incl_gw_dir = incl_gw_dir
        self.simil_matrix_folder = simil_matrix_folder

    def _flow_maps(
        self, diagram: Diagram
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """
        Build { <node id>: [ <incoming sequence flow id>, ... ] } and the
        equivalent outgoing map, from the diagram's sequenceFlow elements'
        source_ref/target_ref.
        """
        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in diagram.edges():
            if edge.type != "sequenceFlow":
                continue
            outgoing[edge.source_ref].append(edge.target_ref) #edge.id)
            incoming[edge.target_ref].append(edge.source_ref) #edge.id)
        return incoming, outgoing

    def _node_attributes(
        self,
        element: DiagramElement,
        incoming: dict[str, list[str]],
        outgoing: dict[str, list[str]],
    ) -> dict:
        """Build the attribute dict expected downstream for one node element."""
        return {
            'type': element.type,
            'node_name': element.label or '',
            'incoming_flow': incoming.get(element.id, []),
            'outgoing_flow': outgoing.get(element.id, []),
        }

    def _preprocess_nodes(self, diagram: Diagram):
        incoming, outgoing = self._flow_maps(diagram)
        nodes = [
            (element.id, self._node_attributes(element, incoming, outgoing))
            for element in diagram.elements.values()
            if element.type in NODE_TYPES
        ]

        if self.incl_gw_dir:
            nodes = self._set_gateway_dirs(nodes)

        return nodes

    def _set_gateway_dirs(self, nodes):
        processed_nodes = []
        for node in nodes:
            node_id, attributes = node

            if self._is_gateway(node):
                incoming = attributes['incoming_flow']
                outgoing = attributes['outgoing_flow']

                # Compute the difference between incoming and outgoing counts.
                diff = len(incoming) - len(outgoing)

                # Use diff if we have any flow info;
                if incoming or outgoing:
                    # Positive diff means more incoming → likely a merge.
                    if diff > 0:
                        dir = 'merge'
                    # Negative diff means more outgoing → likely a split.
                    elif diff < 0:
                        dir = 'split'
                    else:
                        # If there is more than one incoming flow, we assume merge; otherwise, split.
                        if len(incoming) > 1:
                            dir = 'merge'
                        else:
                            dir = 'split'
                # otherwise, fall back to a generic (assume split)
                else:
                    dir = 'split'

                attributes['dir'] = dir

            processed_nodes.append((node_id, attributes))

        return processed_nodes

    def get_bpmn_nodes(self, diagram: Diagram):
        """
        Gets the nodes from the parsed BPMN diagram.
        :param diagram: a bpmn_highlighter.model.Diagram
        :return: [ ( <node id>, <attribute list> ), ... ]
        """

        nodes = self._preprocess_nodes(diagram)
        return nodes

    def get_bpmn_edges(self, diagram: Diagram) -> list[tuple[str, str]]:
        """
        Gets the (sequence flow) edges from the parsed BPMN diagram.
        :param diagram: a bpmn_highlighter.model.Diagram
        :return: [ ( <source node id>, <target node id>, <edge id> ), ... ]
        """

        return [
            (edge.source_ref, edge.target_ref, edge.id)
            for edge in diagram.edges()
            if edge.type == "sequenceFlow" and edge.source_ref and edge.target_ref
        ]

    def _id(self, node):
        return node[0]

    def _attr(self, node, attr):
        return node[1][attr]

    def _label(self, node):
        name = self._attr(node, 'node_name')
        type = self._attr(node, 'type')
        return (name if name != "" else self._id(node)) + f" ({type})"

    def _is_event(self, node):
        return self._attr(node, 'type') in EVENT_TYPES

    def _is_gateway(self, node):
        return self._attr(node, 'type') in GATEWAY_TYPES

    def _is_task(self, node):
        return self._attr(node, 'type') in TASK_TYPES

    def _src(self, edge):
        return edge[0]

    def _tgt(self, edge):
        return edge[1]

    def calculate_similarity(self, source1: Diagram | BpmnSource, source2: Diagram | BpmnSource):
        """ Calculate the similarity between two BPMN diagrams.
        :param source1: a Diagram already parsed via parse_diagram(), or any
            BpmnSource parse_diagram() accepts (path, XML str/bytes,
            file-like object, or an lxml Element/ElementTree).
        :param source2: same as source1.
        :return: ( <node similarity score>, <structural similarity score>, <graph edit distance>, <diff> )
        """
        try:
            diagram1 = source1 if isinstance(source1, Diagram) else parse_diagram(source1)
            nodes1 = self.get_bpmn_nodes(diagram1)
            edges1 = self.get_bpmn_edges(diagram1)

            diagram2 = source2 if isinstance(source2, Diagram) else parse_diagram(source2)
            nodes2 = self.get_bpmn_nodes(diagram2)
            edges2 = self.get_bpmn_edges(diagram2)

            nodes = { self._id(node): node for node in chain(nodes1, nodes2) }

            if len(nodes2) == 0 or len(nodes1) == 0:
                return [ ( 0, 0, 0, [] ) ]

            # get similarities for tasks
            simil_matrix = self.create_simil_matrix(nodes1, nodes2, TASK_TYPES)
            self.calc_simil_tasks(nodes1, nodes2, simil_matrix)
            if self.simil_matrix_folder is not None:
                self.save_simil_matrix(nodes1, nodes2, simil_matrix, "task")
            # get single best mapping between tasks based on similarity matrix
            task_equiv_map = self.get_equiv_maps(simil_matrix, True, 0.5)[0]
            # if self.verbose:
                # print("equivalent tasks:", task_equiv_map)

            # get similarities for gateways & events, given the task mapping
            simil_matrix = self.create_simil_matrix(nodes1, nodes2, EVENT_TYPES.union(GATEWAY_TYPES))
            self.calc_simil_others(nodes1, nodes2, task_equiv_map, simil_matrix)
            if self.simil_matrix_folder is not None:
                self.save_simil_matrix(nodes1, nodes2, simil_matrix, "other")

            # get best mappings between gateways & events, given updated similarity matrix
            other_equiv_maps = self.get_equiv_maps(simil_matrix, self.all_equiv_gw_evt_mappings, 0)
            if self.verbose:
                print("# gw & evt mappings:", len(other_equiv_maps))

            best = { 'out': [], 'struct_sim': 0 }
            for other_equiv_map in other_equiv_maps:
                equiv_map = task_equiv_map + other_equiv_map

                node_sim = self.calc_node_simil(equiv_map, len(nodes1), len(nodes2))
                struct_sim, ged, diff = self.calc_struct_simil(edges1, nodes1, edges2, nodes2, equiv_map)

                result = node_sim, struct_sim, ged, diff
                if struct_sim > best['struct_sim']:
                    best['struct_sim'] = struct_sim
                    best['out'] = [ [ result, equiv_map ] ]
                elif struct_sim == best['struct_sim']:
                    best['out'].append([ result, equiv_map ])

            if self.verbose:
                print("# results:", len(best['out']))

            # let's make the equiv maps more readable
            for out in best['out']:
                out[1] = [ (self._label(nodes[e[0]]), self._label(nodes[e[1]]), e[2]) for e in out[1] ]
            
            return best['out']

        except BaseException as e:
            print("error in calculate similarity: ", e)
            raise e

    def create_simil_matrix(self, nodes1, nodes2, types):
        # use nodes2 as columns, and nodes1 as rows
        col_idxes = [ self._id(n2) for n2 in nodes2 if self._attr(n2, 'type') in types ]
        row_idxes = [ self._id(n1) for n1 in nodes1 if self._attr(n1, 'type') in types ]
        
        simil_matrix = pd.DataFrame([ [0.0] * len(col_idxes) ] * len(row_idxes), columns=col_idxes, index=row_idxes)
        return simil_matrix

    def save_simil_matrix(self, nodes1, nodes2, simil_matrix, name):
        display_simil_matrix = simil_matrix.rename(index={
            self._id(node): self._label(node) for node in nodes1
        }, columns={
            self._id(node): self._label(node) for node in nodes2
        })

        simil_matrix_path = os.path.join(self.simil_matrix_folder, f"simil_matrix_{name}.csv")
        # display_simil_matrix.to_excel(simil_matrix_path, index=True)
        display_simil_matrix.to_csv(simil_matrix_path, index=True)
        if self.verbose:
            print(f"similarity matrix saved to {simil_matrix_path}.")

    def calc_simil_tasks(self, nodes1, nodes2, simil_matrix):
        for node1 in ( n1 for n1 in nodes1 if self._is_task(n1) ):
            for node2 in ( n2 for n2 in nodes2 if self._is_task(n2) ):
                # calculate syntactic similarity for nodes
                syn_sim = self.calc_syn_sim(node1, node2)
                # NOTE: currently no semantic similarity (based on synonyms) or attribute similarity
                node_sim = syn_sim
                simil_matrix.loc[self._id(node1), self._id(node2)] = node_sim

    def calc_simil_others(self, nodes1, nodes2, task_equiv_map, simil_matrix):
        task_map = { task1 : task2 for task2, task1, _ in task_equiv_map }
        node_types = { self._id(node): self._attr(node, 'type') for node in chain(nodes1, nodes2) }

        for node1 in ( n1 for n1 in nodes1 if not self._is_task(n1) ):
            for node2 in ( n2 for n2 in nodes2 if not self._is_task(n2) ):
                type_sim = self.calc_type_sim(node1, node2)
                # score=-1 if they don't share a type; score=0 if they do but they don't share any ctx
                ctx_sim = self.calc_ctx_sim(node1, node2, node_types, task_map) if type_sim > 0 else -1
                # print(self._id(node1), self._id(node2), ctx_sim)
                node_sim = ctx_sim
                simil_matrix.loc[self._id(node1), self._id(node2)] = node_sim

    def calc_syn_sim(self, node1, node2):
        """
        Calculate syntactic similarity
        :param node1
        :param node2
        """
        # if (self.ignore_event_names and self._is_event(node1) and self._is_event(node2)) or \
        #    (self.ignore_gateway_names and self._is_gateway(node1) and self._is_gateway(node2)):
        #    return 1

        node1 = self._attr(node1, 'node_name')
        node2 = self._attr(node2, 'node_name')
        if len(node1) == 0 and len(node2) == 0:
            return 1
        else:
            syn_sim = 1 - distance(node1, node2) / max(len(node1), len(node2))
            return syn_sim

    def calc_type_sim(self, node1, node2):
        """
        Calculate type similarity
        :param node1:
        :param node2:

        This implements the simplest option, namely 1 if they have the same type, and 0 otherwise.
        """
        if self._attr(node1, 'type') == self._attr(node2, 'type'):
            if self.incl_gw_dir and self._is_gateway(node1) and self._is_gateway(node2):
                return 1 if self._attr(node1, 'dir') == self._attr(node2, 'dir') else 0
            return 1
        else:
            return 0

    def calc_ctx_sim(self, node1, node2, node_types, task_map):
        """
        For a given direction: 
            - if neither sets of connected tasks have edges, ignore the direction in the score 
        (i.e., 'count' is not incremented).
            - else, take the jaccard similarity of the set of connected tasks
        
        If count is 0, means none of the directions had connected tasks; 
        the nodes are similar in that neither of them are connected to tasks.
        A score of 1 is returned in that case. 
        (An alternative could be to return any score that is higher than zero; 
        but, then even a perfect alignment  would not yield a score of 1).
        Else, cases where one node has a direction with some connected tasks, 
        and the other node does not, would return the same score (0) as the above case.
        But, these nodes are arguably less similar.
        """

        total = 0; count = 0
        dirs = 'incoming_flow', 'outgoing_flow'
        for dir in dirs:
            s1 = { n for n in self._attr(node1, dir) if node_types[n] in TASK_TYPES }
            s2 = { n for n in self._attr(node2, dir) if node_types[n] in TASK_TYPES }
            # nothing to compare (there are no dir nodes, or they are not tasks)
            if len(s1) == 0 and len(s2) == 0: 
                continue
            count += 1 # stuff to compare, so count as a score
            # map n1's dir nodes to n2 nodes
            s1 = { task_map[n] for n in s1 if n in task_map }
            score = jaccard_simil(s1, s2) if len(s1) > 0 or len(s2) > 0 else 0
            total += score
        
        return total / count if count > 0 else 1

    def get_equiv_maps(self, simil_matrix, get_all, threshold):
        """ Gets equivalency mappings between nodes.
        :return: [ ( <node2 id>, <node1 id>, <node similarity score> ), ... ]
        """

        # get best matches between nodes1 and nodes2 given the similarity matrix
        # i.e., node1 is matched to node2 if there is no better match for node1 nor node2
        matches = best_matches_per_column_row(simil_matrix, threshold)
        # print(matches)

        if self.verbose:
            df = pd.DataFrame(matches, columns=['left', 'right', 'score'])
            overlaps = df[df['left'].duplicated(keep=False)]
            if len(overlaps) > 0:
                print("overlapping matches:\n", overlaps)

        # there may be multiple best matches for a given node1 or node2
        # we turn this into a bipartite matching problem; each mapping pairs node1's to one best node2 match
        # if get_all is true, we return all maximum mappings (each mapping as many nodes as possible)
        # else, we only return one maximum mapping
        return get_bipartite_matchings(matches, get_all)

    def calc_node_simil(self, equiv_map, node1_num, node2_num):
        """
        Computing the node similarity given the equivalence mapping and number of nodes
        :param equiv_map
        :param node1_num
        :param node2_num
        """
        return 2 * sum(map(lambda x: x[2], equiv_map)) / (node1_num + node2_num)

    def calc_struct_simil(self, edges1, nodes1, edges2, nodes2, equiv_map):
        """
        Calculate the structural similarity (simged) and GED.
        :param equiv_map: [ ( <node2 id>, <node1 id>, <score> ), ... ]
        """

        diff = { 'nodes': { 'add': [], 'del': [] }, 'edges': { 'add': [], 'del': [] } }
        sn = self.get_sn(nodes1, nodes2, equiv_map, diff)
        se = self.get_se(edges1, edges2, equiv_map, diff)
        sim_sum = sum(map(lambda x: 1 - x[2], equiv_map))
        ged = len(sn) + len(se) + 2 * sim_sum

        snv = len(sn) / (len(nodes1) + len(nodes2))
        sev = len(se) / (len(edges1) + len(edges2))
        sbv = 2 * sim_sum / (len(nodes1) + len(nodes2) - len(sn))
        simged = 1 - (snv + sev + sbv) / 3

        return simged, ged, diff

    def get_sn(self, nodes1, nodes2, equiv_map, diff):
        """
        Get the set of all inserted and deleted nodes (sn).
        :param equiv_map: [ ( <node2 id>, <node1 id>, <score> ), ... ]
        """
        mapped_nodes2 = set(map(lambda x: self._id(x), equiv_map))
        nodes2 = set(map(lambda x: self._id(x), nodes2))
        # in graph2 but not in graph1 = "inserted"
        insert_nodes = list(nodes2 - mapped_nodes2)
        diff['nodes']['add'] = insert_nodes

        mapped_nodes1 = set(map(lambda x: x[1], equiv_map))
        nodes1 = set(map(lambda x: self._id(x), nodes1))
        # in graph1 but not in graph2 = "deleted"
        delete_nodes = list(nodes1 - mapped_nodes1)
        diff['nodes']['del'] = delete_nodes

        sn = delete_nodes + insert_nodes
        return sn

    def get_se(self, edges1, edges2, equiv_map, diff):
        """
        Get the set of all inserted and deleted edges (se).
        :param edges1: [ ( <source node1 id>, <target node1 id>, <edge id> ), ... ]
        :param edges2: [ ( <source node2 id>, <target node2 id>, <edge id> ), ... ]
        :param equiv_map: [ ( <node2 id>, <node1 id>, <score> ), ... ]
        """

        # { <node2 id>: <node1 id> }
        edges1_dict = { ( self._src(edge), self._tgt(edge) ): edge for edge in edges1 }
        equiv_dict = { item[0]: item[1] for item in equiv_map }

        insert_edges = []
        for edge2 in edges2: # ( <source node2 id>, <target node2 id> )
            # edge was fully mapped in equiv mapping
            if self._src(edge2) in equiv_dict and self._tgt(edge2) in equiv_dict:
                edge1 = ( equiv_dict[self._src(edge2)], equiv_dict[self._tgt(edge2)] )
                # in graph2 but not in graph1: "inserted"
                if edge1 not in edges1_dict:
                    insert_edges.append(edge2)
                else:
                    # in graph2 _and_ graph1
                    # remaining edges1_dict will be those in graph1 but not in graph2 = "deleted"
                    del edges1_dict[edge1]
            else:
                # in graph2 but not in graph1: "inserted"
                insert_edges.append(edge2)

        delete_edges = list(edges1_dict.values())
        diff['edges']['add'] = insert_edges
        diff['edges']['del'] = delete_edges

        return insert_edges + delete_edges
