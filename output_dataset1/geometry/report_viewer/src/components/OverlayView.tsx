import React, { useState, useRef } from "react";
import type { MatchItem, UnmatchedItem, FilterState } from "../types";
import { ZoomIn, ZoomOut, RotateCcw, Sliders, ArrowRight } from "lucide-react";

interface OverlayViewProps {
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

export const OverlayView: React.FC<OverlayViewProps> = ({
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
  const [opacity, setOpacity] = useState<number>(50); // 0..100
  const [showAllArrows, setShowAllArrows] = useState<boolean>(true);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [imgSize, setImgSize] = useState<{ width: number; height: number }>({
    width: 1536,
    height: 826,
  });

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = e.currentTarget;
    if (naturalWidth && naturalHeight) {
      setImgSize({ width: naturalWidth, height: naturalHeight });
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const newZoom = Math.min(Math.max(0.5, zoom * zoomFactor), 8);
    setZoom(newZoom);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Left click only
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Helper for status color
  const getMatchColor = (match: MatchItem) => {
    if (match.match_type === 'uncertain') return { stroke: '#eab308', fill: 'rgba(234,179,8,.18)', label: 'Uncertain' };
    const { dx, dy, relative_dw: dw, relative_dh: dh } = match.delta;
    const shiftDist = Math.sqrt(dx * dx + dy * dy);
    const wRelPct = Math.abs(dw) * 100;
    const hRelPct = Math.abs(dh) * 100;
    const isShift = shiftDist > 8;
    const isSize = wRelPct > 15 || hRelPct > 15;

    if (isShift && isSize) return { stroke: "#d946ef", fill: "rgba(217, 70, 239, 0.15)", label: "Shift + Size" }; // Magenta
    if (isShift) return { stroke: "#f97316", fill: "rgba(249, 115, 22, 0.15)", label: "Shift" }; // Orange
    if (isSize) return { stroke: "#a855f7", fill: "rgba(168, 85, 247, 0.15)", label: "Size" }; // Purple
    return { stroke: "#64748b", fill: "rgba(100, 116, 139, 0.1)", label: "Minor" }; // Gray
  };

  // Check if selected item is a match or unmatched
  const selectedMatchId = selectedItem && "before_id" in selectedItem ? selectedItem.before_id : null;
  const selectedUnmatchedId = selectedItem && !("before_id" in selectedItem) ? selectedItem.id : null;

  return (
    <div className="flex flex-col gap-4 bg-slate-900/60 p-4 rounded-xl border border-slate-800 shadow-xl">
      {/* Control bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-800/80 p-3 rounded-lg border border-slate-700">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-slate-300 font-medium">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <span>Độ trong suốt After (Top layer):</span>
            <input
              type="range"
              min="0"
              max="100"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-32 accent-cyan-400 cursor-pointer"
            />
            <span className="w-10 font-mono text-cyan-300 font-semibold">{opacity}%</span>
          </div>

          <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer select-none border-l border-slate-700 pl-4">
            <input
              type="checkbox"
              checked={showAllArrows}
              onChange={(e) => setShowAllArrows(e.target.checked)}
              className="rounded accent-cyan-500 bg-slate-700 border-slate-600 w-4 h-4"
            />
            <span>Hiển thị tất cả mũi tên dịch chuyển</span>
          </label>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom((z) => Math.min(z * 1.25, 8))}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition"
            title="Phóng to"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(z * 0.8, 0.5))}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition"
            title="Thu nhỏ"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={resetView}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 transition flex items-center gap-1 text-xs text-slate-300 font-medium"
            title="Đặt lại góc nhìn"
          >
            <RotateCcw className="w-4 h-4" /> Reset ({Math.round(zoom * 100)}%)
          </button>
        </div>
      </div>

      {/* Overlay Canvas Workspace */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="relative w-full h-[650px] bg-slate-950 rounded-lg overflow-hidden cursor-grab active:cursor-grabbing border border-slate-800 select-none"
      >
        <div
          className="absolute origin-top-left transition-transform duration-75 ease-out"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            width: `${imgSize.width}px`,
            height: `${imgSize.height}px`,
          }}
        >
          {/* Base Layer: Before Image */}
          <img
            src={beforeImgUrl}
            alt="Before Baseline"
            className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none"
            onLoad={handleImageLoad}
          />

          {/* Overlay Layer: After Image */}
          <img
            src={afterImgUrl}
            alt="After Inject"
            ref={imgRef}
            className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none"
            style={{ opacity: opacity / 100 }}
          />

          {/* SVG Overlay Layer for Vector Arrows & BBoxes */}
          <svg
            className="absolute top-0 left-0 w-full h-full pointer-events-auto"
            viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}
          >
            <defs>
              {/* Vector arrow markers */}
              <marker
                id="arrowhead-selected"
                markerWidth="8"
                markerHeight="8"
                refX="6"
                refY="4"
                orient="auto"
              >
                <polygon points="0 0, 8 4, 0 8" fill="#38bdf8" />
              </marker>
              <marker
                id="arrowhead-shift"
                markerWidth="8"
                markerHeight="8"
                refX="6"
                refY="4"
                orient="auto"
              >
                <polygon points="0 0, 8 4, 0 8" fill="#f97316" />
              </marker>
            </defs>

            {/* Unmatched Before elements (Red dashed) */}
            {filteredUnmatchedBefore.map((item) => {
              const [x1, y1, x2, y2] = item.bbox;
              const isSelected = selectedUnmatchedId === item.id;
              const isHovered = hoveredItemId === item.id;
              return (
                <rect
                  key={`overlay-ub-${item.id}`}
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth={isSelected || isHovered ? 3 : 1.5}
                  strokeDasharray="4 4"
                  className="cursor-pointer hover:stroke-red-300 transition-all"
                  onMouseEnter={() => onHoverItem(item.id)}
                  onMouseLeave={() => onHoverItem(null)}
                  onClick={() => onSelectItem(item)}
                />
              );
            })}

            {/* Unmatched After elements (Blue dashed) */}
            {filteredUnmatchedAfter.map((item) => {
              const [x1, y1, x2, y2] = item.bbox;
              const isSelected = selectedUnmatchedId === item.id;
              const isHovered = hoveredItemId === item.id;
              return (
                <rect
                  key={`overlay-ua-${item.id}`}
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth={isSelected || isHovered ? 3 : 1.5}
                  strokeDasharray="4 4"
                  className="cursor-pointer hover:stroke-blue-300 transition-all"
                  onMouseEnter={() => onHoverItem(item.id)}
                  onMouseLeave={() => onHoverItem(null)}
                  onClick={() => onSelectItem(item)}
                />
              );
            })}

            {/* Matched Pairs BBoxes & Displacement Vectors */}
            {filteredMatches.map((m) => {
              const isSelected = selectedMatchId === m.before_id;
              const isHovered = hoveredItemId === m.before_id || hoveredItemId === m.after_id;
              const colors = getMatchColor(m);

              const [bX1, bY1, bX2, bY2] = m.before_bbox;
              const [aX1, aY1, aX2, aY2] = m.after_bbox;

              const cx_b = m.before_geometry.cx;
              const cy_b = m.before_geometry.cy;
              const cx_a = m.after_geometry.cx;
              const cy_a = m.after_geometry.cy;

              const shiftDist = Math.sqrt(
                Math.pow(cx_a - cx_b, 2) + Math.pow(cy_a - cy_b, 2)
              );

              const drawArrow = (showAllArrows && shiftDist > 1) || isSelected || isHovered;

              return (
                <g key={`overlay-match-${m.before_id}-${m.after_id}`}>
                  {/* Before BBox */}
                  <rect
                    x={bX1}
                    y={bY1}
                    width={bX2 - bX1}
                    height={bY2 - bY1}
                    fill={isSelected ? "rgba(56, 189, 248, 0.2)" : colors.fill}
                    stroke={isSelected ? "#38bdf8" : colors.stroke}
                    strokeWidth={isSelected || isHovered ? 2.5 : 1.5}
                    className="cursor-pointer transition-all"
                    onMouseEnter={() => onHoverItem(m.before_id)}
                    onMouseLeave={() => onHoverItem(null)}
                    onClick={() => onSelectItem(m)}
                  />

                  {/* After BBox */}
                  <rect
                    x={aX1}
                    y={aY1}
                    width={aX2 - aX1}
                    height={aY2 - aY1}
                    fill="none"
                    stroke={isSelected ? "#38bdf8" : colors.stroke}
                    strokeWidth={isSelected || isHovered ? 2 : 1}
                    strokeDasharray="3 3"
                    className="cursor-pointer transition-all pointer-events-none"
                  />

                  {/* Displacement Vector Line from Before Center to After Center */}
                  {drawArrow && (
                    <g className="pointer-events-none">
                      <line
                        x1={cx_b}
                        y1={cy_b}
                        x2={cx_a}
                        y2={cy_a}
                        stroke={isSelected ? "#38bdf8" : colors.stroke === "#64748b" ? "#f97316" : colors.stroke}
                        strokeWidth={isSelected ? 3 : isHovered ? 2.5 : 1.8}
                        strokeDasharray={shiftDist > 30 ? "6 3" : undefined}
                        markerEnd={isSelected ? "url(#arrowhead-selected)" : "url(#arrowhead-shift)"}
                      />
                      <circle cx={cx_b} cy={cy_b} r={isSelected ? 4 : 3} fill="#ef4444" />
                      <circle cx={cx_a} cy={cy_a} r={isSelected ? 4 : 3} fill="#3b82f6" />
                    </g>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Selected Item Detail Card on Overlay */}
      {selectedItem && (
        <div className="bg-slate-800/90 border border-slate-700 p-4 rounded-xl flex items-center justify-between text-sm">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              <ArrowRight className="w-5 h-5" />
            </div>
            {"before_id" in selectedItem ? (
              <div>
                <div className="font-semibold text-slate-200 flex items-center gap-2">
                  <span>Match Pair:</span>
                  <span className="font-mono text-cyan-300">{selectedItem.before_id}</span>
                  <span className="text-slate-500">➔</span>
                  <span className="font-mono text-cyan-300">{selectedItem.after_id}</span>
                  <span className="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300 uppercase">
                    {selectedItem.match_type}
                  </span>
                </div>
                <div className="text-xs text-slate-400 mt-1 flex gap-4">
                  <span>
                    Dịch chuyển (dx, dy):{" "}
                    <strong className="text-amber-400 font-mono">
                      ({selectedItem.delta.dx > 0 ? `+${selectedItem.delta.dx}` : selectedItem.delta.dx}px,{" "}
                      {selectedItem.delta.dy > 0 ? `+${selectedItem.delta.dy}` : selectedItem.delta.dy}px)
                    </strong>
                  </span>
                  <span>
                    Khoảng cách:{" "}
                    <strong className="text-amber-400 font-mono">
                      {Math.round(
                        Math.sqrt(
                          selectedItem.delta.dx * selectedItem.delta.dx +
                            selectedItem.delta.dy * selectedItem.delta.dy
                        ) * 10
                      ) / 10}
                      px
                    </strong>
                  </span>
                  <span>
                    IoU: <strong className="text-emerald-400 font-mono">{selectedItem.matching_features.iou}</strong>
                  </span>
                  <span>
                    Cost: <strong className="text-slate-300 font-mono">{selectedItem.match_cost}</strong>
                  </span>
                </div>
              </div>
            ) : (
              <div>
                <div className="font-semibold text-slate-200">
                  Phần tử chưa match ({selectedItem.side === "before" ? "Phía Before" : "Phía After"}):{" "}
                  <span className="font-mono text-amber-300">{selectedItem.id}</span>
                </div>
                <div className="text-xs text-slate-400 mt-1 font-mono">
                  BBox: [{selectedItem.bbox.join(", ")}] | Label: {selectedItem.class_label}
                </div>
              </div>
            )}
          </div>

          <button
            onClick={() => onSelectItem(null)}
            className="px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs font-medium transition"
          >
            Đóng lựa chọn
          </button>
        </div>
      )}
    </div>
  );
};
