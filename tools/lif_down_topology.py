"""Discover the first bottom-up edge from the Decoder's finest RepC3 input, without layer constants."""
from copy import deepcopy


def refs(layer, index):
    source = layer[0]
    return [index + r if r < 0 else r for r in (source if isinstance(source, list) else [source])]


def locate(config):
    layers = config['backbone'] + config['head']
    decoder = [i for i, row in enumerate(layers) if row[2] == 'RTDETRDecoder']
    if len(decoder) != 1:
        raise ValueError('Expected one native RTDETRDecoder')
    p3, p4, p5 = refs(layers[decoder[0]], decoder[0])
    if any(layers[i][2] != 'RepC3' for i in (p3, p4, p5)):
        raise ValueError('Decoder inputs must be C2 RepC3 outputs')
    edges = []
    for fine, coarse in ((p3, p4), (p4, p5)):
        concat_refs = refs(layers[coarse], coarse)
        if len(concat_refs) != 1 or layers[concat_refs[0]][2] != 'Concat':
            raise ValueError('Expected PAN RepC3 after Concat')
        concat = concat_refs[0]
        candidates = [i for i in refs(layers[concat], concat)
                      if layers[i][2] in ('Conv', 'LIFDown') and refs(layers[i], i) == [fine]
                      and layers[i][3] == [256, 3, 2]]
        if len(candidates) != 1:
            raise ValueError('Expected unique k3/s2 C2 downsample')
        down = candidates[0]
        edges.append(dict(input=fine, downsample=down, concat=concat,
                          lateral=[i for i in refs(layers[concat], concat) if i != down], output=coarse))
    return dict(decoder=decoder[0], p3_to_p4=edges[0], p4_to_p5=edges[1])


def variant_config(baseline):
    target = deepcopy(baseline)
    index = locate(baseline)['p3_to_p4']['downsample'] - len(baseline['backbone'])
    target['head'][index][2] = 'LIFDown'
    return target
