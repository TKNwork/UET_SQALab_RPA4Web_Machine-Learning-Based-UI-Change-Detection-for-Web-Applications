import React from 'react';
import type { ViewMode } from '../types';
import { Columns, Layers, FileText, Code2, Sparkles } from 'lucide-react';

interface HeaderProps {
  pairId: string;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  onOpenReportModal: () => void;
  onOpenJsonModal: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  pairId,
  viewMode,
  setViewMode,
  onOpenReportModal,
  onOpenJsonModal
}) => {
  return (
    <header className="glass-panel" style={{ padding: '1rem 1.5rem', marginBottom: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        
        {/* Title & Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ 
            width: '42px', 
            height: '42px', 
            borderRadius: '12px', 
            background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(99, 102, 241, 0.4)'
          }}>
            <Sparkles size={22} color="#ffffff" />
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <h1 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#f8fafc', letterSpacing: '-0.02em' }}>
                UI Geometry Change Inspector
              </h1>
              <span style={{ 
                background: 'rgba(99, 102, 241, 0.15)', 
                color: '#818cf8', 
                border: '1px solid rgba(99, 102, 241, 0.3)', 
                padding: '0.15rem 0.6rem', 
                borderRadius: '6px', 
                fontSize: '0.75rem',
                fontWeight: 600
              }}>
                Mẫu {pairId}
              </span>
            </div>
            <p style={{ fontSize: '0.825rem', color: '#94a3b8', marginTop: '0.1rem' }}>
              Visualizing Object Detection Pairs & Semantic-First Hierarchical Element Matching
            </p>
          </div>
        </div>

        {/* View Mode Toggle & Raw Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          
          {/* Mode Segmented Switch */}
          <div style={{ 
            display: 'flex', 
            background: 'rgba(15, 23, 42, 0.8)', 
            padding: '3px', 
            borderRadius: '10px',
            border: '1px solid rgba(255, 255, 255, 0.08)'
          }}>
            <button
              onClick={() => setViewMode('side-by-side')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.45rem 0.85rem',
                borderRadius: '8px',
                border: 'none',
                background: viewMode === 'side-by-side' ? '#6366f1' : 'transparent',
                color: viewMode === 'side-by-side' ? '#ffffff' : '#94a3b8',
                fontWeight: 600,
                fontSize: '0.825rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <Columns size={16} />
              Hai ảnh song song
            </button>

            <button
              onClick={() => setViewMode('overlay')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                padding: '0.45rem 0.85rem',
                borderRadius: '8px',
                border: 'none',
                background: viewMode === 'overlay' ? '#6366f1' : 'transparent',
                color: viewMode === 'overlay' ? '#ffffff' : '#94a3b8',
                fontWeight: 600,
                fontSize: '0.825rem',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <Layers size={16} />
              Chồng ảnh (Overlay)
            </button>
          </div>

          {/* Quick Raw File Views */}
          <button
            onClick={onOpenReportModal}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.45rem 0.75rem',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              background: 'rgba(30, 41, 59, 0.6)',
              color: '#cbd5e1',
              fontSize: '0.825rem',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            title="Xem báo cáo văn bản matching_report.txt"
          >
            <FileText size={15} color="#38bdf8" />
            Report TXT
          </button>

          <button
            onClick={onOpenJsonModal}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.45rem 0.75rem',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              background: 'rgba(30, 41, 59, 0.6)',
              color: '#cbd5e1',
              fontSize: '0.825rem',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            title="Xem kết quả JSON gốc"
          >
            <Code2 size={15} color="#a855f7" />
            JSON Output
          </button>

        </div>
      </div>
    </header>
  );
};
