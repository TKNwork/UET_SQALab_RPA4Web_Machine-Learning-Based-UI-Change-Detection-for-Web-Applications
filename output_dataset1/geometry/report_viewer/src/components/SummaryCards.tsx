import React from 'react';
import type { Summary } from '../types';
import { Layers, CheckCircle2, AlertCircle, Type, ShieldCheck } from 'lucide-react';

interface SummaryCardsProps {
  summary: Summary;
  filteredCount: number;
}

export const SummaryCards: React.FC<SummaryCardsProps> = ({ summary, filteredCount }) => {
  const {
    before_elements,
    after_elements,
    matched_pairs,
    unmatched_before,
    unmatched_after,
    evidence_breakdown
  } = summary;

  return (
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
      gap: '0.85rem', 
      marginBottom: '1.25rem' 
    }}>
      
      {/* Total Elements Before / After */}
      <div className="glass-panel" style={{ padding: '0.85rem 1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.775rem', fontWeight: 600 }}>
          <span>PHẦN TỬ ĐẦU VÀO</span>
          <Layers size={16} color="#6366f1" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginTop: '0.4rem' }}>
          <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc' }}>
            {before_elements} <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 500 }}>Before</span>
          </span>
          <span style={{ color: '#475569', fontSize: '1.1rem' }}>/</span>
          <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#38bdf8' }}>
            {after_elements} <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 500 }}>After</span>
          </span>
        </div>
        <div style={{ fontSize: '0.725rem', color: '#64748b', marginTop: '0.2rem' }}>
          Được trích xuất từ Object Detection
        </div>
      </div>

      {/* Matched Pairs */}
      <div className="glass-panel" style={{ padding: '0.85rem 1rem', borderColor: 'rgba(16, 185, 129, 0.25)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.775rem', fontWeight: 600 }}>
          <span>TỔNG CẶP ĐÃ MATCH</span>
          <CheckCircle2 size={16} color="#10b981" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginTop: '0.4rem' }}>
          <span style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>
            {matched_pairs} <span style={{ fontSize: '0.775rem', color: '#94a3b8', fontWeight: 500 }}>cặp</span>
          </span>
          <span style={{ fontSize: '0.75rem', background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', padding: '0.1rem 0.4rem', borderRadius: '4px', fontWeight: 600 }}>
            {filteredCount} hiển thị
          </span>
        </div>
        <div style={{ fontSize: '0.725rem', color: '#64748b', marginTop: '0.2rem' }}>
          Tỷ lệ ghép thành công: {((matched_pairs / Math.max(before_elements, 1)) * 100).toFixed(1)}%
        </div>
      </div>

      {/* Unmatched Status */}
      <div className="glass-panel" style={{ padding: '0.85rem 1rem', borderColor: 'rgba(239, 68, 68, 0.25)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.775rem', fontWeight: 600 }}>
          <span>CHƯA TÌM ĐƯỢC TƯƠNG ỨNG</span>
          <AlertCircle size={16} color="#ef4444" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginTop: '0.4rem' }}>
          <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#ef4444' }} title="Chưa match ở Before">
            {unmatched_before} <span style={{ fontSize: '0.725rem', color: '#fca5a5', fontWeight: 500 }}>Before</span>
          </span>
          <span style={{ color: '#475569', fontSize: '1.1rem' }}>/</span>
          <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#3b82f6' }} title="Chưa match ở After">
            {unmatched_after} <span style={{ fontSize: '0.725rem', color: '#93c5fd', fontWeight: 500 }}>After</span>
          </span>
        </div>
        <div style={{ fontSize: '0.725rem', color: '#94a3b8', marginTop: '0.2rem' }}>
          Không bị gán ghép gượng ép
        </div>
      </div>

      {/* OCR anchor */}
      <div className="glass-panel" style={{ padding: '0.85rem 1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.775rem', fontWeight: 600 }}>
          <span>TIER 1 — OCR FIRST</span>
          <Type size={16} color="#10b981" />
        </div>
        <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#34d399', marginTop: '0.4rem' }}>
          {evidence_breakdown.ocr_anchors} <span style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: 500 }}>matches</span>
        </div>
        <div style={{ fontSize: '0.725rem', color: '#64748b', marginTop: '0.2rem' }}>
          Ghép bằng văn bản OCR (sim ≥ 0.85)
        </div>
      </div>

      {/* Tier 2 & Tier 3 Breakdown */}
      <div className="glass-panel" style={{ padding: '0.85rem 1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.775rem', fontWeight: 600 }}>
          <span>TIER 2 & 3 BREAKDOWN</span>
          <ShieldCheck size={16} color="#f59e0b" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.6rem', marginTop: '0.4rem' }}>
          <span style={{ fontSize: '1.1rem', fontWeight: 700, color: '#06b6d4' }} title="Container evidence">
            Container: {evidence_breakdown.container_matches}
          </span>
          <span style={{ color: '#475569' }}>|</span>
          <span style={{ fontSize: '1.1rem', fontWeight: 700, color: '#fbbf24' }} title="Multi-evidence assignment">
            Assignment: {evidence_breakdown.evidence_assignments}
          </span>
        </div>
        <div style={{ fontSize: '0.725rem', color: '#94a3b8', marginTop: '0.2rem' }}>
          {evidence_breakdown.no_description_diagnostics ? `Diagnostic: ${evidence_breakdown.no_description_diagnostics} thiếu desc` : 'Fallback ghép bằng IoU & vị trí'}
        </div>
      </div>

    </div>
  );
};
