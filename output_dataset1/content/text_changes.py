"""Text-only research profile; historical content-v1 remains unchanged."""
import copy
import re
import unicodedata
from difflib import SequenceMatcher

from content import pipeline as c

PROFILE = 'visible-text-v1'


def normalize_text(text):
    # Preserve case, punctuation, accents and numbers; ignore OCR whitespace splits.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text or '')).strip()


def select_text_changes(inference):
    """GT-free projection from content inference; no metadata/selector inputs.

    OCR missing on one side cannot establish insertion/deletion: keep uncertain.
    This profile measures visible text, not the semantic meaning of the sentence.
    """
    result = copy.deepcopy(inference)
    for pair in result['matched_pairs'] + result['uncertain_pairs']:
        info = pair['content']
        before, after = (normalize_text(info[key]) for key in ('before_text', 'after_text'))
        info['legacy_classification'] = {key: info[key] for key in ('state', 'kind', 'reason')}
        info.update(state='uncertain', kind='none', reason='missing_two_sided_text')
        info['text_profile'] = PROFILE
        info['normalized_before_text'], info['normalized_after_text'] = before, after
        if before and after:
            if before == after:
                info.update(state='unchanged', reason='same_visible_text')
            elif info['visual'].get('changed_fraction', 0) >= result['config']['ocr_min_visual_fraction']:
                info.update(state='changed', kind='ocr_and_visual', reason='different_visible_text_with_visual_evidence')
                if SequenceMatcher(None, before, after).ratio() >= .9:
                    info['quality_flags'].append('small_text_edit_or_ocr_noise_requires_review')
            else:
                info['reason'] = 'text_difference_without_visual_confirmation'
    result['predictions'] = [dict(pair, prediction_index=i) for i, pair in enumerate(
        pair for pair in result['matched_pairs'] if pair['content']['state'] == 'changed')]
    result['groups'] = c.groups(result['predictions'], result['config'])
    result['text_profile'] = PROFILE
    result['text_profile_limitations'] = ['Missing OCR is not proof of text insertion/deletion',
        'OCR and pixel evidence do not independently certify text transcription correctness',
        'Unmatched and uncertain correspondences are retained for review, not scored predictions']
    return result
