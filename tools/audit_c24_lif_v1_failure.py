"""Read-only, manifest-verified audit of the supplied server LIGHT archive."""
import argparse
import json
import tarfile
from pathlib import Path
from c24_lif_v1_common import require, sha256, write_json
from c24_lif_v1_pack import verify_archive


def audit(path, output):
    manifest = verify_archive(path)
    require(path.stat().st_size == 260391 and sha256(path) ==
            '2acde183614281cce7549fa7374c6124fc1b5d64547bf0e06841df495123d156', 'Unexpected failure archive')
    require(len(manifest) == 52, 'Unexpected manifest count')
    with tarfile.open(path, 'r:gz') as archive:
        records = {n: json.load(archive.extractfile(n)) for n in manifest if n.endswith('.json')}
        preflight = records['metadata/preflight/preflight.json']
        # Preserve the actual server evidence, never substitute source_docs/checks.json.
        for name, data in records.items():
            if name.startswith('metadata/'):
                write_json(output / 'original_server' / name.removeprefix('metadata/'), data)
        summary = dict(archive_bytes=path.stat().st_size, archive_sha256=sha256(path),
                       manifest_members=len(manifest), manifest_verification='PASSED',
                       evidence_namespace='metadata/preflight', runtime=records['metadata/environment.json'],
                       first_unaccepted_stage='fusion_cuda_amp', terminal_exception_stage='server_B16_640',
                       unresolved_stages=[], original_loss_finite='UNKNOWN_NOT_RECORDED',
                       original_nonfinite_gradient_names='UNKNOWN_NOT_RECORDED',
                       earliest_faulty_operator='UNKNOWN', only_scaler_overflow='UNPROVEN',
                       capacity_native_init='NOT_RUN', capacity_nonzero_stress='BLOCKED',
                       amp_calibration='NOT_RUN', training_dispatched=False)
        for stage in preflight['stages']:
            if stage['status'] == 'PASSED':
                continue
            actual = records['metadata/preflight/' + stage['name'] + '.json']
            row = dict(name=stage['name'], status=stage['status'], error=actual.get('error'))
            if 'fusion_' in stage['name']:
                case = actual['result']['cases'][0]
                row.update(natural_relation=case['natural_relation'],
                           position_mismatch=sum(x != y for a, b in zip(case['candidate_ids_a'], case['candidate_ids_b']) for x, y in zip(a, b)),
                           replacements=[dict(only_a=b['only_a'], only_b=b['only_b']) for b in case['boundary']],
                           continuous_max_abs=max(r.get('max_abs', 0) for r in case['continuous']),
                           operator_status=case['operator_status'], parent_control=case['parent_control'],
                           replay={k: v['status'] for k, v in case['replay'].items()})
            summary['unresolved_stages'].append(row)
        write_json(output / 'original_blocking_summary.json', summary)
        write_json(output / 'archive_manifest_verified.json', manifest)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive, args.output), ensure_ascii=False, indent=2))
