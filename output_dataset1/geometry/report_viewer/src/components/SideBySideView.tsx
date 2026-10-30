import React, { useState, useRef } from 'react';
import type { MatchItem, UnmatchedItem, FilterState } from '../types';
import { ZoomIn, ZoomOut, RotateCcw, Lock, Unlock } from 'lucide-react';

interface SideBySideViewProps {
  beforeImgUrl: string;
  afterImgUrl: string;
  matches: MatchItem[];
  unmatchedBefore: UnmatchedItem[];
  unmatchedAfter: UnmatchedItem[];
  filteredMatches: MatchItem[];
  filteredUnmatchedBefore: UnmatchedItem[];
  filteredUnmatchedAfter: UnmatchedItem[];
  selectedItem: MatchItem | UnmatchedItem | null;
  hoveredItemId: string | null;
  onSelectItem: (item: MatchItem | UnmatchedItem | null) => void;
  onHoverItem: (id: string | null) => void;
  filterState: FilterState;
}

const NATURAL_WIDTH = 1536;
const NATURAL_HEIGHT = 826;

export const SideBySideView: React.FC<SideBySideViewProps> = ({
  beforeImgUrl,
  afterImgUrl,
  filteredMatches,
  filteredUnmatchedBefore,
  filteredUnmatchedAfter,
  selectedItem,
  hoveredItemId,
  onSelectItem,
  onHoverItem,
}) => {
  const [zoomBefore, setZoomBefore] = useState<number>(1);
  const [panBefore, setPanBefore] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [zoomAfter, setZoomAfter] = useState<number>(1);
  const [panAfter, setPanAfter] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [isSync, setIsSync] = useState<boolean>(true);
  const [isDraggingBefore, setIsDraggingBefore] = useState<boolean>(false);
  const [dragStartBefore, setDragStartBefore] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [isDraggingAfter, setIsDraggingAfter] = useState<boolean>(false);
  const [dragStartAfter, setDragStartAfter] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [tooltipData, setTooltipData] = useState<{
    x: number;
    y: number;
    match?: MatchItem;
    unmatched?: UnmatchedItem;
  } | null>(null);

  const containerBeforeRef = useRef<HTMLDivElement>(null);
  const containerAfterRef = useRef<HTMLDivElement>(null);

  const selectedMatchId = selectedItem && 'before_id' in selectedItem ? selectedItem.before_id : null;
  const selectedUnmatchedId = selectedItem && !('before_id' in selectedItem) ? selectedItem.id : null;

  // Sync zoom/pan handler
  const updateZoomPan = (newZoom: number, newPan: { x: number; y: number }, source: 'before' | 'after') => {
    const clampedZoom = Math.min(Math.max(0.5, newZoom), 8);
    if (isSync) {
      setZoomBefore(clampedZoom);
      setZoomAfter(clampedZoom);
      setPanBefore(newPan);
      setPanAfter(newPan);
    } else if (source === 'before') {
      setZoomBefore(clampedZoom);
      setPanBefore(newPan);
    } else {
      setZoomAfter(clampedZoom);
      setPanAfter(newPan);
    }
  };

  const handleWheelBefore = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    updateZoomPan(zoomBefore * factor, panBefore, 'before');
  };

  const handleWheelAfter = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    updateZoomPan(zoomAfter * factor, panAfter, 'after');
  };

  const resetViews = () => {
    setZoomBefore(1);
    setZoomAfter(1);
    setPanBefore({ x: 0, y: 0 });
    setPanAfter({ x: 0, y: 0 });
  };

  // Helper for style/color of matched elements
  const getMatchStyle = (m: MatchItem) => {
    if (m.match_type === 'uncertain') return { stroke: '#eab308', fill: 'rgba(234,179,8,.18)', labelColor: '#eab308' };
    const { dx, dy, relative_dw: dw, relative_dh: dh } = m.delta;
    const shiftDist = Math.sqrt(dx * dx + dy * dy);
    const wPct = Math.abs(dw) * 100;
    const hPct = Math.abs(dh) * 100;

    const isShift = shiftDist > 8;
    const isSize = wPct > 15 || hPct > 15;

    if (isShift && isSize) {
      return { stroke: '#d946ef', fill: 'rgba(217, 70, 239, 0.18)', labelColor: '#f0abfc' };
    }
    if (isShift) {
      return { stroke: '#f97316', fill: 'rgba(249, 115, 22, 0.18)', labelColor: '#ffedd5' };
    }
    if (isSize) {
      return { stroke: '#a855f7', fill: 'rgba(168, 85, 247, 0.18)', labelColor: '#e9d5ff' };
    }
    return { stroke: '#64748b', fill: 'rgba(100, 116, 139, 0.12)', labelColor: '#cbd5e1' };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* Top Toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(15, 23, 42, 0.8)', padding: '0.6rem 1rem', borderRadius: '10px', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={() => setIsSync(!isSync)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.4rem 0.75rem',
              borderRadius: '6px',
              border: 'none',
              background: isSync ? 'rgba(99, 102, 241, 0.2)' : 'rgba(255, 255, 255, 0.05)',
              color: isSync ? '#818cf8' : '#94a3b8',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            {isSync ? <Lock size={15} /> : <Unlock size={15} />}
            {isSync ? 'Đồng bộ Zoom & Pan: Bật' : 'Đồng bộ Zoom & Pan: Tắt'}
          </button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            onClick={() => updateZoomPan(zoomBefore * 1.2, panBefore, 'before')}
            style={{ background: 'rgba(30, 41, 59, 0.8)', border: 'none', color: '#cbd5e1', padding: '0.4rem 0.6rem', borderRadius: '6px', cursor: 'pointer' }}
            title="Phóng to"
          >
            <ZoomIn size={16} />
          </button>
          <button
            onClick={() => updateZoomPan(zoomBefore * 0.8, panBefore, 'before')}
            style={{ background: 'rgba(30, 41, 59, 0.8)', border: 'none', color: '#cbd5e1', padding: '0.4rem 0.6rem', borderRadius: '6px', cursor: 'pointer' }}
            title="Thu nhỏ"
          >
            <ZoomOut size={16} />
          </button>
          <button
            onClick={resetViews}
            style={{ background: 'rgba(30, 41, 59, 0.8)', border: 'none', color: '#cbd5e1', padding: '0.4rem 0.75rem', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.8rem' }}
          >
            <RotateCcw size={15} /> Reset ({Math.round(zoomBefore * 100)}%)
          </button>
        </div>
      </div>

      {/* Main Side-by-Side Canvas Container */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', width: '100%' }}>
        
        {/* BEFORE IMAGE PANEL */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#cbd5e1', fontSize: '0.85rem', fontWeight: 600 }}>
            <span>BEFORE (Baseline)</span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{filteredMatches.length} matches | {filteredUnmatchedBefore.length} chưa match</span>
          </div>

          <div
            ref={containerBeforeRef}
            onWheel={handleWheelBefore}
            onMouseDown={(e) => {
              if (e.button !== 0) return;
              setIsDraggingBefore(true);
              setDragStartBefore({ x: e.clientX - panBefore.x, y: e.clientY - panBefore.y });
            }}
            onMouseMove={(e) => {
              if (!isDraggingBefore) return;
              const newPan = { x: e.clientX - dragStartBefore.x, y: e.clientY - dragStartBefore.y };
              if (isSync) {
                setPanBefore(newPan);
                setPanAfter(newPan);
              } else {
                setPanBefore(newPan);
              }
            }}
            onMouseUp={() => setIsDraggingBefore(false)}
            onMouseLeave={() => setIsDraggingBefore(false)}
            style={{
              position: 'relative',
              height: '580px',
              background: '#020617',
              borderRadius: '12px',
              overflow: 'hidden',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              cursor: isDraggingBefore ? 'grabbing' : 'grab'
            }}
          >
            <div
              style={{
                position: 'absolute',
                transformOrigin: '0 0',
                transform: `translate(${panBefore.x}px, ${panBefore.y}px) scale(${zoomBefore})`,
                width: `${NATURAL_WIDTH}px`,
                height: `${NATURAL_HEIGHT}px`
              }}
            >
              <img
                src={beforeImgUrl}
                alt="Before Panel"
                style={{ width: '100%', height: '100%', objectFit: 'contain', pointerEvents: 'none' }}
              />

              <svg
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'auto' }}
                viewBox={`0 0 ${NATURAL_WIDTH} ${NATURAL_HEIGHT}`}
              >
                {/* Unmatched Before */}
                {filteredUnmatchedBefore.map((item) => {
                  const [x1, y1, x2, y2] = item.bbox;
                  const isSelected = selectedUnmatchedId === item.id;
                  const isHovered = hoveredItemId === item.id;
                  return (
                    <g key={`b-ub-${item.id}`}>
                      <rect
                        x={x1}
                        y={y1}
                        width={x2 - x1}
                        height={y2 - y1}
                        fill={isSelected ? 'rgba(239, 68, 68, 0.25)' : 'none'}
                        stroke="#ef4444"
                        strokeWidth={isSelected || isHovered ? 3 : 1.5}
                        strokeDasharray="4 4"
                        style={{ cursor: 'pointer' }}
                        onMouseEnter={(e) => {
                          onHoverItem(item.id);
                          setTooltipData({ x: e.clientX, y: e.clientY, unmatched: item });
                        }}
                        onMouseLeave={() => {
                          onHoverItem(null);
                          setTooltipData(null);
                        }}
                        onClick={() => onSelectItem(item)}
                      />
                      <text
                        x={x1 + 3}
                        y={y1 > 16 ? y1 - 4 : y1 + 14}
                        fill="#fca5a5"
                        fontSize="11"
                        fontWeight="600"
                        fontFamily="monospace"
                        style={{ pointerEvents: 'none' }}
                      >
                        {item.id}
                      </text>
                    </g>
                  );
                })}

                {/* Matched Pairs */}
                {filteredMatches.map((m) => {
                  const [x1, y1, x2, y2] = m.before_bbox;
                  const isSelected = selectedMatchId === m.before_id;
                  const isHovered = hoveredItemId === m.before_id || hoveredItemId === m.after_id;
                  const style = getMatchStyle(m);

                  return (
                    <g key={`b-match-${m.before_id}`}>
                      <rect
                        x={x1}
                        y={y1}
                        width={x2 - x1}
                        height={y2 - y1}
                        fill={isSelected ? 'rgba(56, 189, 248, 0.25)' : style.fill}
                        stroke={isSelected ? '#38bdf8' : style.stroke}
                        strokeWidth={isSelected || isHovered ? 3 : 1.5}
                        style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                        onMouseEnter={(e) => {
                          onHoverItem(m.before_id);
                          setTooltipData({ x: e.clientX, y: e.clientY, match: m });
                        }}
                        onMouseLeave={() => {
                          onHoverItem(null);
                          setTooltipData(null);
                        }}
                        onClick={() => onSelectItem(m)}
                      />
                      <rect
                        x={x1}
                        y={y1 > 16 ? y1 - 16 : y1}
                        width={Math.min(x2 - x1, 80)}
                        height="14"
                        fill="rgba(15, 23, 42, 0.85)"
                        rx="3"
                        style={{ pointerEvents: 'none' }}
                      />
                      <text
                        x={x1 + 4}
                        y={y1 > 16 ? y1 - 5 : y1 + 10}
                        fill={style.labelColor}
                        fontSize="10"
                        fontWeight="700"
                        fontFamily="monospace"
                        style={{ pointerEvents: 'none' }}
                      >
                        {m.before_id}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        </div>

        {/* AFTER IMAGE PANEL */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#cbd5e1', fontSize: '0.85rem', fontWeight: 600 }}>
            <span>AFTER (Inject)</span>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{filteredMatches.length} matches | {filteredUnmatchedAfter.length} chưa match</span>
          </div>

          <div
            ref={containerAfterRef}
            onWheel={handleWheelAfter}
            onMouseDown={(e) => {
              if (e.button !== 0) return;
              setIsDraggingAfter(true);
              setDragStartAfter({ x: e.clientX - panAfter.x, y: e.clientY - panAfter.y });
            }}
            onMouseMove={(e) => {
              if (!isDraggingAfter) return;
              const newPan = { x: e.clientX - dragStartAfter.x, y: e.clientY - dragStartAfter.y };
              if (isSync) {
                setPanBefore(newPan);
                setPanAfter(newPan);
              } else {
                setPanAfter(newPan);
              }
            }}
            onMouseUp={() => setIsDraggingAfter(false)}
            onMouseLeave={() => setIsDraggingAfter(false)}
            style={{
              position: 'relative',
              height: '580px',
              background: '#020617',
              borderRadius: '12px',
              overflow: 'hidden',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              cursor: isDraggingAfter ? 'grabbing' : 'grab'
            }}
          >
            <div
              style={{
                position: 'absolute',
                transformOrigin: '0 0',
                transform: `translate(${panAfter.x}px, ${panAfter.y}px) scale(${zoomAfter})`,
                width: `${NATURAL_WIDTH}px`,
                height: `${NATURAL_HEIGHT}px`
              }}
            >
              <img
                src={afterImgUrl}
                alt="After Panel"
                style={{ width: '100%', height: '100%', objectFit: 'contain', pointerEvents: 'none' }}
              />

              <svg
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'auto' }}
                viewBox={`0 0 ${NATURAL_WIDTH} ${NATURAL_HEIGHT}`}
              >
                {/* Unmatched After */}
                {filteredUnmatchedAfter.map((item) => {
                  const [x1, y1, x2, y2] = item.bbox;
                  const isSelected = selectedUnmatchedId === item.id;
                  const isHovered = hoveredItemId === item.id;
                  return (
                    <g key={`a-ua-${item.id}`}>
                      <rect
                        x={x1}
                        y={y1}
                        width={x2 - x1}
                        height={y2 - y1}
                        fill={isSelected ? 'rgba(59, 130, 246, 0.25)' : 'none'}
                        stroke="#3b82f6"
                        strokeWidth={isSelected || isHovered ? 3 : 1.5}
                        strokeDasharray="4 4"
                        style={{ cursor: 'pointer' }}
                        onMouseEnter={(e) => {
                          onHoverItem(item.id);
                          setTooltipData({ x: e.clientX, y: e.clientY, unmatched: item });
                        }}
                        onMouseLeave={() => {
                          onHoverItem(null);
                          setTooltipData(null);
                        }}
                        onClick={() => onSelectItem(item)}
                      />
                      <text
                        x={x1 + 3}
                        y={y1 > 16 ? y1 - 4 : y1 + 14}
                        fill="#93c5fd"
                        fontSize="11"
                        fontWeight="600"
                        fontFamily="monospace"
                        style={{ pointerEvents: 'none' }}
                      >
                        {item.id}
                      </text>
                    </g>
                  );
                })}

                {/* Matched Pairs */}
                {filteredMatches.map((m) => {
                  const [x1, y1, x2, y2] = m.after_bbox;
                  const isSelected = selectedMatchId === m.before_id;
                  const isHovered = hoveredItemId === m.before_id || hoveredItemId === m.after_id;
                  const style = getMatchStyle(m);

                  return (
                    <g key={`a-match-${m.after_id}`}>
                      <rect
                        x={x1}
                        y={y1}
                        width={x2 - x1}
                        height={y2 - y1}
                        fill={isSelected ? 'rgba(56, 189, 248, 0.25)' : style.fill}
                        stroke={isSelected ? '#38bdf8' : style.stroke}
                        strokeWidth={isSelected || isHovered ? 3 : 1.5}
                        style={{ cursor: 'pointer', transition: 'all 0.15s ease' }}
                        onMouseEnter={(e) => {
                          onHoverItem(m.after_id);
                          setTooltipData({ x: e.clientX, y: e.clientY, match: m });
                        }}
                        onMouseLeave={() => {
                          onHoverItem(null);
                          setTooltipData(null);
                        }}
                        onClick={() => onSelectItem(m)}
                      />
                      <rect
                        x={x1}
                        y={y1 > 16 ? y1 - 16 : y1}
                        width={Math.min(x2 - x1, 80)}
                        height="14"
                        fill="rgba(15, 23, 42, 0.85)"
                        rx="3"
                        style={{ pointerEvents: 'none' }}
                      />
                      <text
                        x={x1 + 4}
                        y={y1 > 16 ? y1 - 5 : y1 + 10}
                        fill={style.labelColor}
                        fontSize="10"
                        fontWeight="700"
                        fontFamily="monospace"
                        style={{ pointerEvents: 'none' }}
                      >
                        {m.after_id}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Floating Hover Tooltip */}
      {tooltipData && (
        <div
          style={{
            position: 'fixed',
            left: `${tooltipData.x + 15}px`,
            top: `${tooltipData.y + 15}px`,
            zIndex: 100,
            background: 'rgba(15, 23, 42, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: '10px',
            padding: '0.75rem 1rem',
            boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
            pointerEvents: 'none',
            maxWidth: '300px',
            fontSize: '0.8rem',
            backdropFilter: 'blur(8px)'
          }}
        >
          {tooltipData.match ? (
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '0.3rem' }}>
                <span style={{ fontWeight: 700, color: '#38bdf8' }}>
                  {tooltipData.match.before_id} ➔ {tooltipData.match.after_id}
                </span>
                <span style={{
                  background: 'rgba(99, 102, 241, 0.2)',
                  color: '#a5b4fc',
                  padding: '0.1rem 0.4rem',
                  borderRadius: '4px',
                  fontSize: '0.7rem',
                  fontWeight: 600
                }}>
                  {tooltipData.match.match_type === 'ocr_anchor' ? 'OCR anchor' : 'Evidence assignment'}
                </span>
              </div>

              <div style={{ marginBottom: '0.3rem', color: '#e2e8f0' }}>
                <strong>Nhãn:</strong> "{tooltipData.match.before_class}"
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.25rem', color: '#94a3b8', fontSize: '0.75rem', marginBottom: '0.4rem' }}>
                <div>dx: <span style={{ color: '#38bdf8', fontWeight: 600 }}>{tooltipData.match.delta.dx > 0 ? `+${tooltipData.match.delta.dx}` : tooltipData.match.delta.dx} px</span></div>
                <div>dy: <span style={{ color: '#38bdf8', fontWeight: 600 }}>{tooltipData.match.delta.dy > 0 ? `+${tooltipData.match.delta.dy}` : tooltipData.match.delta.dy} px</span></div>
                <div>dw: <span style={{ color: '#a855f7', fontWeight: 600 }}>{tooltipData.match.delta.dw > 0 ? `+${tooltipData.match.delta.dw}` : tooltipData.match.delta.dw} px</span></div>
                <div>dh: <span style={{ color: '#a855f7', fontWeight: 600 }}>{tooltipData.match.delta.dh > 0 ? `+${tooltipData.match.delta.dh}` : tooltipData.match.delta.dh} px</span></div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#cbd5e1', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '0.3rem' }}>
                <span>IoU: <strong>{tooltipData.match.matching_features.iou.toFixed(3)}</strong></span>
                <span>Cost: <strong>{tooltipData.match.match_cost.toFixed(3)}</strong></span>
              </div>
            </div>
          ) : tooltipData.unmatched ? (
            <div>
              <div style={{ fontWeight: 700, color: '#ef4444', marginBottom: '0.3rem' }}>
                Chưa tìm được tương ứng ({tooltipData.unmatched.side.toUpperCase()})
              </div>
              <div style={{ color: '#e2e8f0' }}>ID: <span style={{ fontFamily: 'monospace' }}>{tooltipData.unmatched.id}</span></div>
              <div style={{ color: '#94a3b8', fontSize: '0.75rem', marginTop: '0.2rem' }}>
                BBox: [{tooltipData.unmatched.bbox.join(', ')}]
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};
