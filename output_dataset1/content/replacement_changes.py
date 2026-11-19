"""Content replacement candidates and explicit human-reviewed GT eligibility.

No GT in candidate generation. Pixel evidence cannot certify unchanged font,
color, element identity or unintentional replacement; those remain review fields.
"""
import copy
from collections import Counter

from content import pipeline as c
from content.text_changes import normalize_text

PROFILE = 'content-replacement-v1'
CONFIG = {'position_tolerance_px': 3., 'size_tolerance_px': 3.,
          'text_visual_fraction': .025, 'visual_fraction': .12, 'visual_mae': 10.,
          'threshold_status': 'provisional_not_calibrated_on_real_data'}
REVIEW_FIELDS = ('content_type', 'content_replaced', 'same_element',
                 'presentation_unchanged', 'unintentional_confirmed', 'evidence')


def raw_text(pair, side, detections):
    """Recover case-sensitive OCR from raw class_label preserved by normalization."""
    members = set(pair.get(side+'_members', [pair[side+'_id']]))
    values = [e for e in detections if e['id'] in members and e.get('source') == 'ocr']
    labels = {normalize_text(e.get('class_label', '')) for e in values}
    if len(labels) == 1:
        return labels.pop(), 'raw_ocr_label'
    # Containers/segmentation can have composite OCR. Preserve uncertainty about case.
    return normalize_text(pair['content'][side+'_text']), 'normalized_or_composite_ocr'


def type_hint(pair, before, after, before_text, after_text):
    if before_text or after_text:
        return 'text'
    labels = []
    for side, detections in [('before', before), ('after', after)]:
        members = set(pair.get(side+'_members', [pair[side+'_id']]))
        labels += [str(e.get('class_name', e.get('class_label', ''))).lower()
                   for e in detections if e['id'] in members]
    # Detector labels are hints only; never infer icon/image from bbox size.
    hints = {'icon' if any(word in label for word in ('icon', 'symbol')) else
             'image' if any(word in label for word in ('image', 'photo', 'picture')) else
             'unknown' for label in labels}
    return next(iter(hints)) if len(hints) == 1 else 'unknown'


def select_replacements(inference):
    result = copy.deepcopy(inference)
    cfg = CONFIG
    for pair in result['matched_pairs'] + result['uncertain_pairs']:
        info = pair['content']
        bt, bs = raw_text(pair, 'before', result['before_detections'])
        at, ats = raw_text(pair, 'after', result['after_detections'])
        b, a = pair['before_bbox'], pair['after_bbox']
        position = max(abs((a[i]+a[i+2]-b[i]-b[i+2])/2) for i in (0, 1))
        size = max(abs((a[i+2]-a[i])-(b[i+2]-b[i])) for i in (0, 1))
        stable = position <= cfg['position_tolerance_px'] and size <= cfg['size_tolerance_px']
        visual = info['visual']
        if bt and at:
            changed = bt != at and visual.get('changed_fraction', 0) >= cfg['text_visual_fraction']
            evidence = 'different_text_and_pixels' if changed else 'same_text' if bt == at else 'text_without_pixel_confirmation'
        elif bt or at:
            changed = False
            evidence = 'one_sided_ocr_not_proof_of_replacement'
        else:
            changed = visual.get('changed_fraction', 0) >= cfg['visual_fraction'] and visual.get('mae', 0) >= cfg['visual_mae']
            evidence = 'visual_replacement_or_style_candidate' if changed else 'insufficient_visual_evidence'
        info['legacy_classification'] = {key: info[key] for key in ('state', 'kind', 'reason')}
        info.update(state='changed' if changed and stable else 'unchanged' if evidence == 'same_text' else 'uncertain',
                    kind='ocr_and_visual' if bt and at and changed else 'visual_replacement_candidate' if changed else 'none',
                    reason=evidence, before_text=bt, after_text=at)
        pair['replacement'] = {'content_type_hint': type_hint(pair, result['before_detections'], result['after_detections'], bt, at),
            'before_text_source': bs, 'after_text_source': ats,
            'position_delta_px': position, 'size_delta_px': size,
            'geometry_stable': stable, 'presentation_status': 'requires_font_color_and_style_review',
            'candidate_status': 'pure_candidate' if changed and stable else 'mixed_candidate' if changed else 'uncertain_or_unchanged',
            'semantic_confirmed': None, 'unintentional_confirmed': None}
    result['predictions'] = [dict(pair, prediction_index=i) for i, pair in enumerate(
        pair for pair in result['matched_pairs'] if pair['replacement']['candidate_status'] == 'pure_candidate')]
    result['groups'] = c.groups(result['predictions'], result['config'])
    result['replacement_profile'] = PROFILE
    result['replacement_config'] = dict(CONFIG)
    result['candidate_counts'] = dict(Counter(p['replacement']['candidate_status'] for p in result['matched_pairs']))
    result['not_semantically_confirmed'] = True
    return result


def audit_gt(targets, reviews):
    if not isinstance(reviews, dict):
        raise ValueError('Per-sample GT review must be an object')
    audited = []
    for target in targets:
        entry = reviews.get(str(target['original_gt_index']), {})
        if not isinstance(entry, dict):
            raise ValueError('Per-GT review must be an object')
        if set(entry) - set(REVIEW_FIELDS):
            raise ValueError('Unexpected GT review fields')
        for key in REVIEW_FIELDS[1:-1]:
            if entry.get(key) is not None and type(entry[key]) is not bool:
                raise ValueError(f'{key} must be boolean or null')
        kind = entry.get('content_type', 'unknown')
        if kind not in ('text', 'image', 'icon', 'unknown'):
            raise ValueError('Invalid reviewed content type')
        meaningful = kind != 'unknown' or any(entry.get(k) is not None for k in REVIEW_FIELDS[1:-1])
        if meaningful and (not isinstance(entry.get('evidence'), str) or not entry['evidence'].strip()):
            raise ValueError('A GT review requires an evidence note')
        if entry.get('content_replaced') is False or entry.get('same_element') is False or entry.get('unintentional_confirmed') is False:
            status = 'excluded'
        elif entry.get('content_replaced') is True and entry.get('presentation_unchanged') is False:
            status = 'mixed'
        elif all(entry.get(k) is True for k in REVIEW_FIELDS[1:-1]) and kind != 'unknown':
            status = 'pure'
        else:
            status = 'needs_review'
        audited.append({**target, 'eligibility': status, 'reviewed_content_type': kind,
                        'review': {key: entry.get(key) for key in REVIEW_FIELDS},
                        'original_multilabel': len(target['original']['type']) > 1})
    unknown = set(reviews) - {str(t['original_gt_index']) for t in targets}
    if unknown:
        raise ValueError(f'Review refers to nonexistent content GT indices: {sorted(unknown)}')
    return audited


def evaluate_reviewed(result, gt_data, audited):
    """Keep original bbox indices; exclude unreviewed GT only from reviewed score."""
    pure = {t['original_gt_index'] for t in audited if t['eligibility'] == 'pure'}
    restricted = copy.deepcopy(gt_data)
    for i, box in enumerate(restricted['bboxes']):
        if i not in pure:
            box['type'] = [label for label in box['type'] if label != 'content']
    ev = c.evaluate(result, restricted)
    ev.update(metric='candidate_localization_on_reviewed_pure_gt',
              semantic_correctness_confirmed=False,
              coverage=ev['covered']/ev['gt_count'] if ev['gt_count'] else None,
              eligibility_counts=dict(Counter(t['eligibility'] for t in audited)))
    if not ev['gt_count']:
        ev['pass'] = None
    return ev
