#!/usr/bin/env python3
"""Convert a BPMN 2.0 XML file to a process tree."""
import argparse

import pm4py
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
import libs.pm4py.classic as classic


def bpmn_to_svg(bpmn_path: str, svg_path: str = None) -> None:
    bpmn_model = bpmn_importer.apply(bpmn_path)
    parameters = bpmn_visualizer.Variants.CLASSIC.value.Parameters
    gviz = classic.apply(bpmn_model, parameters={parameters.FORMAT: "svg", parameters.ENABLE_SWIMLANES: False})
    
    if svg_path is not None:
        bpmn_visualizer.save(gviz, svg_path)

    return gviz


def bpmn_to_process_tree(bpmn_path: str, ptml_path: str = None):
    bpmn_graph = pm4py.read_bpmn(bpmn_path)
    process_tree = pm4py.convert_to_process_tree(bpmn_graph)

    if ptml_path:
        pm4py.write_ptml(process_tree, ptml_path)

    return process_tree


def bpmn_to_petri_net(bpmn_path: str, pnml_path: str = None):
    bpmn_graph = pm4py.read_bpmn(bpmn_path)
    net, im, fm = pm4py.convert_to_petri_net(bpmn_graph)

    if pnml_path:
        pm4py.write_pnml(net, im, fm, pnml_path)

    return net, im, fm