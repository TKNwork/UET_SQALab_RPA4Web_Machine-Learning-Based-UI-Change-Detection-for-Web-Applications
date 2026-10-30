import React from 'react';
import { Info } from 'lucide-react';

export const LegendBar: React.FC = () => {
  return (
    <div className="glass-panel" style={{ padding: '0.75rem 1.25rem', marginBottom: '1.25rem', fontSize: '0.8rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        
        {/* Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#94a3b8', fontWeight: 600 }}>
          <Info size={15} color="#6366f1" />
          <span>Chú giải Quy ước Màu sắc &amp; Ký hiệu:</span>
        </div>

        {/* Legend Items */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', flexWrap: 'wrap' }}>
          
          {/* Stable / Small */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#64748b', border: '1px solid #94a3b8' }}></span>
            <span style={{ color: '#cbd5e1' }}>Thay đổi nhỏ (≤2px / ≤5%)</span>
          </div>

          {/* Significant Shift */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#f97316', border: '1px solid #fdba74' }}></span>
            <span style={{ color: '#fed7aa', fontWeight: 600 }}>Dịch chuyển đáng kể (&gt;8px)</span>
          </div>

          {/* Significant Size Change */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#a855f7', border: '1px solid #d8b4fe' }}></span>
            <span style={{ color: '#e9d5ff', fontWeight: 600 }}>Kích thước đáng kể (&gt;15%)</span>
          </div>

          {/* Both Shift & Size */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#ec4899', border: '1px solid #fbcfe8' }}></span>
            <span style={{ color: '#fbcfe8', fontWeight: 600 }}>Cả 2 (&gt;8px &amp; &gt;15%)</span>
          </div>

          {/* Unmatched Before */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '14px', height: '12px', borderRadius: '3px', border: '2px dashed #ef4444', background: 'rgba(239, 68, 68, 0.15)' }}></span>
            <span style={{ color: '#fca5a5' }}>Chưa match (Before)</span>
          </div>

          {/* Unmatched After */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ width: '14px', height: '12px', borderRadius: '3px', border: '2px dashed #3b82f6', background: 'rgba(59, 130, 246, 0.15)' }}></span>
            <span style={{ color: '#93c5fd' }}>Chưa match (After)</span>
          </div>

        </div>

      </div>
    </div>
  );
};
