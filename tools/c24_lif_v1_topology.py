"""Discover AIFI and the first PAN downsample from the original C2 graph."""
from copy import deepcopy
from lif_down_topology import locate as lif_locate, refs

def locate(config):
    result=lif_locate(config)
    layers=config['backbone']+config['head']
    aifi=[i for i,row in enumerate(layers) if row[2] in ('AIFI','SCCAAIFI')]
    if len(aifi)!=1 or layers[aifi[0]][3]!=[1024,8]:raise ValueError('Original single AIFI geometry required')
    i=aifi[0];projection=refs(layers[i],i)
    if len(projection)!=1 or layers[projection[0]][2]!='Conv':raise ValueError('AIFI must consume projected P5')
    result.update(aifi=i,decoder_inputs=refs(layers[result['decoder']],result['decoder']))
    return result

def variant_config(base):
    config=deepcopy(base);top=locate(base)
    layers=config['backbone']+config['head']
    layers[top['aifi']][2]='SCCAAIFI'
    layers[top['p3_to_p4']['downsample']][2]='LIFDown'
    return config
