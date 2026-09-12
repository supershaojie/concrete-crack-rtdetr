"""Validate source edges before deriving state mappings; no blanket index shift."""
from copy import deepcopy
from lif_down_topology import locate, refs


def graph(config):
    return config['backbone'] + config['head']


def inspect(config):
    rows = graph(config)
    pan = locate(config)
    cs = [i for i, r in enumerate(rows) if r[2] == 'CSCEFv51']
    if len(cs) != 1:
        raise ValueError('Expected exactly one original CSCEFv51')
    cs = cs[0]
    lateral, semantic = refs(rows[cs], cs)
    p3 = pan['p3_to_p4']['input']
    concat, = refs(rows[p3], p3)
    y4, = refs(rows[semantic], semantic)
    if not (rows[lateral][2] == 'Conv' and refs(rows[lateral], lateral) == [5]
            and rows[semantic][2] == 'nn.Upsample' and rows[y4][2] == 'Conv'
            and refs(rows[concat], concat) == [semantic, cs]
            and pan['p3_to_p4']['lateral'] == [y4]
            and sum(r[2] == 'AIFI' for r in rows) == 1):
        raise ValueError('Original CSCEF lateral/Y4/Concat/PAN edges changed')
    return dict(pan, cscef=cs, lateral=lateral, semantic=semantic, y4=y4,
                p3_concat=concat, decoder_inputs=refs(rows[pan['decoder']], pan['decoder']))


def mapping(source, target):
    """Collapse ONLY the verified identity-residual node, then check every common row and edge."""
    a, b = graph(source), graph(target)
    src_cs = [i for i, r in enumerate(a) if r[2] == 'CSCEFv51']
    dst_cs = [i for i, r in enumerate(b) if r[2] == 'CSCEFv51']
    if src_cs: inspect(source)
    if dst_cs: inspect(target)
    if len(src_cs) > 1 or len(dst_cs) > 1 or src_cs and not dst_cs:
        raise ValueError('Unsupported source graph')
    indices = [i for i in range(len(b)) if src_cs or i not in dst_cs]
    if len(indices) != len(a): raise ValueError('Unexpected graph size')
    result = dict(zip(range(len(a)), indices))
    bypass = {i: refs(b[i], i)[0] for i in dst_cs if not src_cs}
    def normalized(row):
        row = deepcopy(row[1:])
        if row[1] == 'LIFDown': row[1] = 'Conv'
        return row
    for i, j in result.items():
        if normalized(a[i]) != normalized(b[j]):
            raise ValueError(f'Common layer semantics changed: {i}->{j}')
        expected = [result[r] if r >= 0 else r for r in refs(a[i], i)]
        actual = [bypass.get(r, r) for r in refs(b[j], j)]
        if expected != actual: raise ValueError(f'Common source edges changed: {i}->{j}')
    return result


def state_mapping(source, target, keys):
    layers = mapping(source, target)
    result = {}
    for key in keys:
        prefix, index, suffix = key.split('.', 2)
        if prefix != 'model' or int(index) not in layers:
            raise ValueError('Unexpected state namespace: ' + key)
        result[key] = f'model.{layers[int(index)]}.{suffix}'
    return result
