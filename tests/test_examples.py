import copy
import importlib.util
import json
import pathlib
import subprocess
import sys
import pytest

ROOT=pathlib.Path(__file__).resolve().parents[1]


def normalized(graph):
    graph=copy.deepcopy(graph)
    graph['7']['inputs']['enabled']=True
    graph['10']['inputs']['filename_prefix']='paired'
    return graph


@pytest.mark.parametrize('prefix',['raw','turbo','raw_reference'])
def test_paired_api_only_changes_enable_and_output(prefix):
    a=json.loads((ROOT/'examples'/f'{prefix}_native.api.json').read_text())
    b=json.loads((ROOT/'examples'/f'{prefix}_spectrum.api.json').read_text())
    assert normalized(a)==normalized(b)
    assert a['7']['inputs']['enabled'] is False
    assert b['7']['inputs']['enabled'] is True
    assert a['2']['inputs']['type']=='krea2'
    assert a['6']['class_type']=='EmptySD3LatentImage'


@pytest.mark.parametrize('path',list((ROOT/'examples').glob('*.json')))
def test_ui_links_and_api_references(path):
    graph=json.loads(path.read_text())
    if '.api.' in path.name:
        for node in graph.values():
            for value in node['inputs'].values():
                if isinstance(value,list):
                    assert value[0] in graph
        return
    nodes={n['id']:n for n in graph['nodes']}
    for link,origin,out,target,slot,typ in graph['links']:
        assert nodes[target]['inputs'][slot]['link']==link
        assert link in nodes[origin]['outputs'][out]['links']
        assert nodes[target]['inputs'][slot]['type']==nodes[origin]['outputs'][out]['type']==typ
    sampler=nodes[8]
    assert sampler['widgets_values'][1]=='fixed'
    assert nodes[2]['widgets_values'][1]=='krea2'


def test_example_generator_reproduces_published_graphs(tmp_path):
    spec=importlib.util.spec_from_file_location('examples_generator',ROOT/'tools/generate_examples.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=tmp_path;module.main()
    for file in tmp_path.glob('*.json'):
        assert file.read_bytes()==(ROOT/'examples'/file.name).read_bytes()


def test_benchmark_cli_loads_without_comfy_or_gpu():
    result=subprocess.run([sys.executable,str(ROOT/'tools/benchmark.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--comfy-root' in result.stdout
