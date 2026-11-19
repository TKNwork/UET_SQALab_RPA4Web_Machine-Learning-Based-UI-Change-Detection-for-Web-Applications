"""Generate visibility candidates from unmatched geometry detections."""


def candidates(geometry, content):
    result = []
    evidence = geometry['evidence_detail']
    for side, subtype in [('before', 'disappeared'), ('after', 'appeared')]:
        covered = {
            member
            for pair in content['matched_pairs']
            for member in pair.get(side + '_members', [pair[side + '_id']])
        }
        members = {
            group['representative']: group['members']
            for group in evidence['duplicate_' + side]['groups']
        }
        for node in geometry['unmatched_' + side]:
            if covered.intersection(members.get(node['id'], [node['id']])):
                continue
            result.append({
                'side': side, 'type': subtype, 'node_id': node['id'],
                'bbox': node['bbox'], 'status': 'candidate',
                'evidence': 'unmatched_after_geometry_and_content',
            })
    return result
