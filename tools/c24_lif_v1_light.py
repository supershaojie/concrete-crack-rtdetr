"""Read-only LIGHT reader: verify manifest and every structured reference."""
import hashlib
import json
from pathlib import PurePosixPath
import tarfile
from c24_lif_v1_common import require, sha256, write_json
from c24_lif_v1_pack import verify_archive


class LightReader:
    def __init__(self, path):
        require(path.stat().st_size <= 8_000_000, 'LIGHT exceeds hard limit')
        with tarfile.open(path) as archive:
            require(sum(m.size for m in archive.getmembers()) <= 64_000_000, 'Expanded archive budget exceeded')
        self.manifest = verify_archive(path)
        with tarfile.open(path) as archive:
            self.payload = {n: archive.extractfile(n).read() for n in self.manifest}
        self.references = set()

    def read(self, name):
        def reference(ref, origin, active):
            target = ref['path']
            if target not in self.manifest:
                target = (PurePosixPath(origin).parent / target).as_posix()
            require(target in self.manifest, 'Missing structured reference: ' + target)
            actual = self.manifest[target]
            require(ref['bytes'] == actual['bytes'] and ref['sha256'] == actual['sha256'], 'Structured reference hash/size mismatch')
            self.references.add((origin, target))
            return document(target, active)
        def resolve(value, origin, active):
            if isinstance(value, dict):
                if 'structured_shard' in value:
                    require(set(value) == {'structured_shard'}, 'Ambiguous shard wrapper')
                    return reference(value['structured_shard'], origin, active)
                if 'structured_shards' in value:
                    parts = [reference(p, origin, active) for p in value['structured_shards']]
                    require(all(isinstance(p, list) for p in parts), 'List shard type mismatch')
                    result = [v for part in parts for v in part]
                    require(len(result) == value.get('original_items', value.get('items')), 'List shard length/order contract')
                    return result
                return {k: resolve(v, origin, active) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v, origin, active) for v in value]
            return value
        def document(path, active):
            require(path not in active and len(active) < 40, 'Cyclic/deep shard reference')
            return resolve(json.loads(self.payload[path]), path, active | {path})
        return document(name, set())


def audit(path, output):
    require(path.stat().st_size == 433147 and sha256(path) ==
            'f725285bd46204ac5df302cfe330df304527b7af03eafe9a9ecdd277ba103f05', 'Unexpected latest LIGHT identity')
    reader = LightReader(path)
    require(len(reader.manifest) == 320, 'Latest manifest count changed')
    for name in reader.manifest:
        if name.endswith('.json'):
            reader.read(name)
    report = reader.read('metadata/preflight/preflight.json')
    stages = {s['name']: s for s in report['stages']}
    summary = dict(archive_sha256=sha256(path), archive_bytes=path.stat().st_size, manifest_entries=len(reader.manifest),
        verified_structured_references=len(reader.references), runtime=report['runtime'],
        blocking_summary=reader.read('metadata/preflight/blocking_summary.json'),
        stages=[dict(name=s['name'],status=s['status']) for s in report['stages']], fusion={}, capacity={})
    for mode in ('amp','half'):
        combo=stages['fusion_cuda_'+mode]['result']['cases'][0]
        parents=stages['original_parent_cutoff_cuda_'+mode]['result']['parents']
        parent=parents['c24']['numerics']
        fields=('candidate_ids_a','candidate_ids_b','boundary')
        same={k:combo[k]==parent[k] for k in fields}
        # Boundary includes a whole-encoder perturbation which can differ outside P3.
        swaps_equal=all(a['changed_scores']==b['changed_scores'] and a['swaps']==b['swaps'] for a,b in zip(combo['boundary'],parent['boundary']))
        require(same['candidate_ids_a'] and same['candidate_ids_b'] and swaps_equal, 'Claimed C24 natural candidate proof does not match archive')
        summary['fusion'][mode]=dict(status=combo['status'],natural_relation=combo['natural_relation'],
            full_A_ids_equal_parent=same['candidate_ids_a'],full_B_ids_equal_parent=same['candidate_ids_b'],
            changed_scores_and_swaps_equal=swaps_equal,whole_boundary_equal=same['boundary'],
            full_encoder_fingerprints_equal=combo['comparison_fingerprints']==parent['comparison_fingerprints'],
            combo_fingerprints=combo['comparison_fingerprints'],parent_fingerprints=parent['comparison_fingerprints'],
            only_a=combo['boundary'][0]['only_a'],only_b=combo['boundary'][0]['only_b'],
            natural_output_boxes=next(v for v in combo['natural_row_comparison'] if v['key']=='output_boxes'),
            shared_P3_certificate='MISSING_IN_HISTORICAL_LIGHT',outside_competitors='MISSING_IN_HISTORICAL_LIGHT')
        write_json(output/('historical_'+mode+'.json'),dict(scope='HISTORICAL_READ_ONLY_MISSING_P3_CERTIFICATE',combo=combo,parent=parent))
    for name in ('native_initialization_B16_640','nonzero_branch_stress_B16_640'):
        row=stages[name]['result'];c=row['calibration']
        require(row['status']==c['status']=='PASSED' and row['source_model_unchanged'] and
                c['actual_optimizer_steps']==c['consecutive_updates']==2 and c['initial_scale']==65536. and
                [a['status'] for a in c['attempts']]==['OVERFLOW_SKIPPED']*4+['UPDATED']*2,'Unexpected native B16 outcome')
        summary['capacity'][name]=dict(status=row['status'],source_model_unchanged=row['source_model_unchanged'],
            input=c['input'],initial_scale=c['initial_scale'],attempts=c['attempts'],actual_optimizer_steps=c['actual_optimizer_steps'],
            save_load_model_optimizer_scaler=c['save_load_model_optimizer_scaler'],
            peak_allocated_bytes=c['peak_allocated_bytes'],peak_reserved_bytes=c['peak_reserved_bytes'])
    write_json(output/'latest_light_audit.json',summary)
    write_json(output/'verified_manifest.json',reader.manifest)
    return summary


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();r=audit(args.archive,args.output)
    print(json.dumps(dict(manifest=r['manifest_entries'],references=r['verified_structured_references'],blockers=r['blocking_summary'],
                         fusion={m:{k:v for k,v in c.items() if k.endswith('equal') or k.endswith('equal_parent')} for m,c in r['fusion'].items()}),indent=2))
