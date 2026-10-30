import React from 'react';
import type { FilterState, FilterPreset } from '../types';
import { Search, Filter, EyeOff, RotateCcw, ArrowUpDown, Tag } from 'lucide-react';

interface FilterBarProps {
  filterState: FilterState;
  onFilterChange: (state: FilterState) => void;
  onResetFilters: () => void;
  filteredCount: number;
  totalCount: number;
}

export const FilterBar: React.FC<FilterBarProps> = ({
  filterState,
  onFilterChange,
  onResetFilters,
  filteredCount,
  totalCount,
}) => {
  const handlePresetChange = (preset: FilterPreset) => {
    if (preset === 'shift_gt_8') {
      onFilterChange({
        ...filterState,
        preset: 'shift_gt_8',
        minShiftPx: 8,
      });
    } else if (preset === 'size_gt_15') {
      onFilterChange({
        ...filterState,
        preset: 'size_gt_15',
        minSizePct: 15,
      });
    } else {
      onFilterChange({
        ...filterState,
        preset,
      });
    }
  };

  return (
    <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 shadow-xl flex flex-col gap-4">
      {/* Top row: Presets & Reset */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Presets */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold uppercase text-slate-400 flex items-center gap-1.5 mr-1">
            <Filter className="w-3.5 h-3.5 text-cyan-400" />
            Bộ lọc nhanh:
          </span>

          <button onClick={() => handlePresetChange('edges8')} aria-pressed={filterState.preset === 'edges8'} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300">Geometry: tâm / cạnh / kích thước</button>
          <button
            onClick={() => handlePresetChange('shift_gt_8')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              filterState.preset === 'shift_gt_8'
                ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40 shadow-sm'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <ArrowUpDown className="w-3.5 h-3.5 text-orange-400" />
            <span>Dịch chuyển tâm &gt; 8 px</span>
          </button>

          <button
            onClick={() => handlePresetChange('size_gt_15')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              filterState.preset === 'size_gt_15'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <ArrowUpDown className="w-3.5 h-3.5 text-purple-400" />
            <span>Thay đổi kích thước &gt; 15%</span>
          </button>

          <button
            onClick={() => handlePresetChange('unmatched')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5 ${
              filterState.preset === 'unmatched'
                ? 'bg-red-500/20 text-red-300 border border-red-500/40 shadow-sm'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <span>Chưa tìm được tương ứng</span>
          </button>

          <button
            onClick={() => handlePresetChange('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              filterState.preset === 'all'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            Tất cả matches
          </button>
        </div>

        {/* Counter & Reset */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-400 font-mono bg-slate-950 px-2.5 py-1 rounded-md border border-slate-800">
            Hiển thị: <strong className="text-cyan-300">{filteredCount}</strong> / {totalCount} matches
          </span>

          <button
            onClick={onResetFilters}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition flex items-center gap-1"
            title="Đặt lại tất cả bộ lọc"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Bottom row: Search, Tier, Threshold Inputs & Minor toggle */}
      <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-slate-800/80 text-xs">
        {/* Search input */}
        <div className="relative flex-1 min-w-[200px]">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Tìm theo ID (ví dụ: fused_0) hoặc nhãn..."
            value={filterState.searchQuery}
            onChange={(e) => onFilterChange({ ...filterState, searchQuery: e.target.value })}
            className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Tier dropdown */}
        <div className="flex items-center gap-2">
          <Tag className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400 font-medium">Bằng chứng matching:</span>
          <select
            value={filterState.tier}
            onChange={(e) => onFilterChange({ ...filterState, tier: e.target.value as any })}
            className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">Tất cả bằng chứng</option>
            <option value="ocr_anchor">OCR anchor</option>
            <option value="container_child_anchors">Container evidence</option>
            <option value="evidence_assignment">Multi-evidence assignment</option>
          </select>
        </div>

        {/* Shift threshold input */}
        <div className="flex items-center gap-2 bg-slate-950 px-2.5 py-1 rounded-lg border border-slate-800">
          <span className="text-slate-400 font-medium">Ngưỡng dịch chuyển:</span>
          <input
            type="number"
            min="0"
            max="100"
            value={filterState.minShiftPx}
            onChange={(e) =>
              onFilterChange({
                ...filterState,
                minShiftPx: Number(e.target.value),
                preset: 'custom',
              })
            }
            className="w-12 bg-slate-900 border border-slate-700 rounded px-1 py-0.5 text-center font-mono text-cyan-300 focus:outline-none"
          />
          <span className="text-slate-500 font-mono">px</span>
        </div>

        {/* Hide minor changes toggle */}
        <label className="flex items-center gap-2 text-slate-300 cursor-pointer select-none ml-auto">
          <input
            type="checkbox"
            checked={filterState.hideMinorChanges}
            onChange={(e) => onFilterChange({ ...filterState, hideMinorChanges: e.target.checked })}
            className="rounded accent-cyan-500 bg-slate-950 border-slate-700 w-4 h-4"
          />
          <EyeOff className="w-3.5 h-3.5 text-slate-400" />
          <span>Ẩn thay đổi không đáng kể (&lt;0.5px, &lt;1%)</span>
        </label>
      </div>
    </div>
  );
};
