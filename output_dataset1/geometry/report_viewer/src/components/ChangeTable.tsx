import React, { useState } from "react";
import type { MatchItem, UnmatchedItem, SortField, SortOrder } from "../types";
import { ArrowUpDown, ArrowUp, ArrowDown, Eye, CheckCircle2, AlertCircle } from "lucide-react";

interface ChangeTableProps {
  filteredMatches: MatchItem[];
  filteredUnmatchedBefore: UnmatchedItem[];
  filteredUnmatchedAfter: UnmatchedItem[];
  selectedItem: MatchItem | UnmatchedItem | null;
  hoveredItemId: string | null;
  onSelectItem: (item: MatchItem | UnmatchedItem | null) => void;
  onHoverItem: (id: string | null) => void;
}

export const ChangeTable: React.FC<ChangeTableProps> = ({
  filteredMatches,
  filteredUnmatchedBefore,
  filteredUnmatchedAfter,
  selectedItem,
  hoveredItemId,
  onSelectItem,
  onHoverItem,
}) => {
  const [activeTab, setActiveTab] = useState<"matches" | "unmatched_before" | "unmatched_after">("matches");
  const [sortField, setSortField] = useState<SortField>("d_position");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc"); // Default to desc for metrics
    }
  };

  // Helper calculation for shift distance in px
  const getShiftDist = (match: MatchItem) => {
    return Math.sqrt(match.delta.dx * match.delta.dx + match.delta.dy * match.delta.dy);
  };

  // Helper calculation for maximum relative size change %
  const getSizeChangePct = (match: MatchItem) => {
    return Math.max(Math.abs(match.delta.relative_dw), Math.abs(match.delta.relative_dh)) * 100;
  };

  // Sorted matched pairs
  const sortedMatches = [...filteredMatches].sort((a, b) => {
    let valA = 0;
    let valB = 0;

    switch (sortField) {
      case "d_position":
        valA = getShiftDist(a);
        valB = getShiftDist(b);
        break;
      case "d_size":
        valA = getSizeChangePct(a);
        valB = getSizeChangePct(b);
        break;
      case "iou":
        valA = a.matching_features.iou;
        valB = b.matching_features.iou;
        break;
      case "match_cost":
        valA = a.match_cost;
        valB = b.match_cost;
        break;
      case "id":
        return sortOrder === "asc"
          ? a.before_id.localeCompare(b.before_id)
          : b.before_id.localeCompare(a.before_id);
    }

    return sortOrder === "asc" ? valA - valB : valB - valA;
  });

  const selectedMatchId = selectedItem && "before_id" in selectedItem ? selectedItem.before_id : null;

  return (
    <div className="bg-slate-900/80 rounded-xl border border-slate-800 shadow-xl overflow-hidden flex flex-col">
      {/* Table Header Controls & Tabs */}
      <div className="p-4 bg-slate-800/60 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-slate-200 text-base flex items-center gap-2">
            <span>Bảng danh sách thay đổi phần tử</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-400 font-mono">
              {activeTab === "matches"
                ? `${sortedMatches.length} matches`
                : activeTab === "unmatched_before"
                ? `${filteredUnmatchedBefore.length} phần tử`
                : `${filteredUnmatchedAfter.length} phần tử`}
            </span>
          </h3>
        </div>

        {/* Tab switches */}
        <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setActiveTab("matches")}
            className={`px-3 py-1.5 rounded-md transition flex items-center gap-1.5 ${
              activeTab === "matches"
                ? "bg-cyan-500/20 text-cyan-300 font-semibold shadow-sm border border-cyan-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>Đã match ({filteredMatches.length})</span>
          </button>

          <button
            onClick={() => setActiveTab("unmatched_before")}
            className={`px-3 py-1.5 rounded-md transition flex items-center gap-1.5 ${
              activeTab === "unmatched_before"
                ? "bg-red-500/20 text-red-300 font-semibold shadow-sm border border-red-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <AlertCircle className="w-3.5 h-3.5 text-red-400" />
            <span>Chưa match Before ({filteredUnmatchedBefore.length})</span>
          </button>

          <button
            onClick={() => setActiveTab("unmatched_after")}
            className={`px-3 py-1.5 rounded-md transition flex items-center gap-1.5 ${
              activeTab === "unmatched_after"
                ? "bg-blue-500/20 text-blue-300 font-semibold shadow-sm border border-blue-500/30"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <AlertCircle className="w-3.5 h-3.5 text-blue-400" />
            <span>Chưa match After ({filteredUnmatchedAfter.length})</span>
          </button>
        </div>
      </div>

      {/* Table Data View */}
      <div className="overflow-x-auto max-h-[500px] scrollbar-thin scrollbar-thumb-slate-700">
        {activeTab === "matches" ? (
          <table className="w-full text-left text-xs text-slate-300 border-collapse">
            <thead className="bg-slate-950/80 sticky top-0 z-10 border-b border-slate-800 text-slate-400 font-medium">
              <tr>
                <th className="py-3 px-4 w-12 text-center">STT</th>
                <th
                  onClick={() => handleSort("id")}
                  className="py-3 px-4 cursor-pointer hover:text-slate-200 transition"
                >
                  <div className="flex items-center gap-1">
                    <span>ID Before ➔ After</span>
                    {sortField === "id" ? (
                      sortOrder === "asc" ? (
                        <ArrowUp className="w-3.5 h-3.5 text-cyan-400" />
                      ) : (
                        <ArrowDown className="w-3.5 h-3.5 text-cyan-400" />
                      )
                    ) : (
                      <ArrowUpDown className="w-3.5 h-3.5 text-slate-600" />
                    )}
                  </div>
                </th>
                <th className="py-3 px-4">Nhãn / Mô tả</th>
                <th className="py-3 px-4">Bằng chứng matching</th>
                <th className="py-3 px-4 text-right">dx, dy (px)</th>
                <th
                  onClick={() => handleSort("d_position")}
                  className="py-3 px-4 text-right cursor-pointer hover:text-slate-200 transition"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Độ dịch chuyển</span>
                    {sortField === "d_position" ? (
                      sortOrder === "asc" ? (
                        <ArrowUp className="w-3.5 h-3.5 text-orange-400" />
                      ) : (
                        <ArrowDown className="w-3.5 h-3.5 text-orange-400" />
                      )
                    ) : (
                      <ArrowUpDown className="w-3.5 h-3.5 text-slate-600" />
                    )}
                  </div>
                </th>
                <th
                  onClick={() => handleSort("d_size")}
                  className="py-3 px-4 text-right cursor-pointer hover:text-slate-200 transition"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Thay đổi W, H</span>
                    {sortField === "d_size" ? (
                      sortOrder === "asc" ? (
                        <ArrowUp className="w-3.5 h-3.5 text-purple-400" />
                      ) : (
                        <ArrowDown className="w-3.5 h-3.5 text-purple-400" />
                      )
                    ) : (
                      <ArrowUpDown className="w-3.5 h-3.5 text-slate-600" />
                    )}
                  </div>
                </th>
                <th
                  onClick={() => handleSort("iou")}
                  className="py-3 px-4 text-right cursor-pointer hover:text-slate-200 transition"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>IoU</span>
                    {sortField === "iou" ? (
                      sortOrder === "asc" ? (
                        <ArrowUp className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <ArrowDown className="w-3.5 h-3.5 text-emerald-400" />
                      )
                    ) : (
                      <ArrowUpDown className="w-3.5 h-3.5 text-slate-600" />
                    )}
                  </div>
                </th>
                <th
                  onClick={() => handleSort("match_cost")}
                  className="py-3 px-4 text-right cursor-pointer hover:text-slate-200 transition"
                >
                  <div className="flex items-center justify-end gap-1">
                    <span>Match Cost</span>
                    {sortField === "match_cost" ? (
                      sortOrder === "asc" ? (
                        <ArrowUp className="w-3.5 h-3.5 text-cyan-400" />
                      ) : (
                        <ArrowDown className="w-3.5 h-3.5 text-cyan-400" />
                      )
                    ) : (
                      <ArrowUpDown className="w-3.5 h-3.5 text-slate-600" />
                    )}
                  </div>
                </th>
                <th className="py-3 px-4 text-center">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {sortedMatches.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-slate-500 font-sans">
                    Không tìm thấy phần tử match phù hợp với bộ lọc hiện tại.
                  </td>
                </tr>
              ) : (
                sortedMatches.map((m, idx) => {
                  const isSelected = selectedMatchId === m.before_id;
                  const isHovered = hoveredItemId === m.before_id || hoveredItemId === m.after_id;
                  const shiftDist = getShiftDist(m);
                  const dwPct = Math.round(m.delta.relative_dw * 100);
                  const dhPct = Math.round(m.delta.relative_dh * 100);

                  const isShiftGt8 = shiftDist > 8;
                  const isSizeGt15 = Math.abs(dwPct) > 15 || Math.abs(dhPct) > 15;

                  // Label display
                  const descText =
                    m.description && m.description.length > 0
                      ? m.description.join(" ")
                      : m.before_class || "ui_element";

                  return (
                    <tr
                      key={`row-match-${m.before_id}-${m.after_id}`}
                      onMouseEnter={() => onHoverItem(m.before_id)}
                      onMouseLeave={() => onHoverItem(null)}
                      onClick={() => onSelectItem(m)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-cyan-950/60 text-cyan-200 font-medium border-l-4 border-l-cyan-400"
                          : isHovered
                          ? "bg-slate-800/80 text-slate-100"
                          : "hover:bg-slate-800/40"
                      }`}
                    >
                      <td className="py-2.5 px-4 text-center text-slate-500 font-sans">{idx + 1}</td>

                      {/* IDs */}
                      <td className="py-2.5 px-4 font-semibold">
                        <span className="text-slate-200">{m.before_id}</span>
                        <span className="text-slate-500 mx-1.5">➔</span>
                        <span className="text-slate-200">{m.after_id}</span>
                      </td>

                      {/* Label / Description */}
                      <td className="py-2.5 px-4 max-w-[200px] truncate font-sans text-slate-300" title={descText}>
                        {descText}
                      </td>

                      {/* Matching evidence */}
                      <td className="py-2.5 px-4 font-sans">
                        {m.match_type === "uncertain" && <span className="text-yellow-400">Uncertain</span>}
                        {m.match_type === "ocr_anchor" && (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                            OCR anchor
                          </span>
                        )}
                        {m.match_type === "container_child_anchors" && (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-500/20 text-blue-300 border border-blue-500/30">
                            Container evidence
                          </span>
                        )}
                        {m.match_type === "evidence_assignment" && (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                            Evidence assignment
                          </span>
                        )}
                      </td>

                      {/* dx, dy */}
                      <td className="py-2.5 px-4 text-right text-slate-400">
                        ({m.delta.dx > 0 ? `+${m.delta.dx}` : m.delta.dx}, {m.delta.dy > 0 ? `+${m.delta.dy}` : m.delta.dy})
                      </td>

                      {/* Shift distance */}
                      <td className="py-2.5 px-4 text-right">
                        <span
                          className={`font-semibold ${
                            isShiftGt8 ? "text-orange-400 font-bold" : "text-slate-400"
                          }`}
                        >
                          {Math.round(shiftDist * 10) / 10} px
                        </span>
                      </td>

                      {/* Relative Size change */}
                      <td className="py-2.5 px-4 text-right">
                        <span
                          className={`font-semibold ${
                            isSizeGt15 ? "text-purple-400 font-bold" : "text-slate-400"
                          }`}
                        >
                          W: {dwPct > 0 ? `+${dwPct}` : dwPct}%, H: {dhPct > 0 ? `+${dhPct}` : dhPct}%
                        </span>
                      </td>

                      {/* IoU */}
                      <td className="py-2.5 px-4 text-right text-emerald-400 font-semibold">
                        {m.matching_features.iou}
                      </td>

                      {/* Match Cost */}
                      <td className="py-2.5 px-4 text-right text-cyan-300">{m.match_cost}</td>

                      {/* Action */}
                      <td className="py-2.5 px-4 text-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectItem(m);
                          }}
                          className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-cyan-300 transition"
                          title="Xem trên ảnh"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        ) : (
          /* Unmatched Items Table */
          <table className="w-full text-left text-xs text-slate-300 border-collapse">
            <thead className="bg-slate-950/80 sticky top-0 z-10 border-b border-slate-800 text-slate-400 font-medium">
              <tr>
                <th className="py-3 px-4 w-12 text-center">STT</th>
                <th className="py-3 px-4">ID phần tử</th>
                <th className="py-3 px-4">Phía</th>
                <th className="py-3 px-4">Nhãn / Class</th>
                <th className="py-3 px-4">Source Detector</th>
                <th className="py-3 px-4 text-right">Tọa độ BBox [x1, y1, x2, y2]</th>
                <th className="py-3 px-4 text-center">Trạng thái</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {(activeTab === "unmatched_before" ? filteredUnmatchedBefore : filteredUnmatchedAfter).length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-500 font-sans">
                    Không có phần tử chưa match trong bộ lọc hiện tại.
                  </td>
                </tr>
              ) : (
                (activeTab === "unmatched_before" ? filteredUnmatchedBefore : filteredUnmatchedAfter).map(
                  (item, idx) => {
                    const isSelected = selectedItem && !('before_id' in selectedItem) && selectedItem.id === item.id;
                    const isHovered = hoveredItemId === item.id;
                    return (
                      <tr
                        key={`row-unmatched-${item.id}`}
                        onMouseEnter={() => onHoverItem(item.id)}
                        onMouseLeave={() => onHoverItem(null)}
                        onClick={() => onSelectItem(item)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-amber-950/60 text-amber-200 font-medium border-l-4 border-l-amber-400"
                            : isHovered
                            ? "bg-slate-800/80 text-slate-100"
                            : "hover:bg-slate-800/40"
                        }`}
                      >
                        <td className="py-2.5 px-4 text-center text-slate-500 font-sans">{idx + 1}</td>
                        <td className="py-2.5 px-4 font-semibold text-slate-200">{item.id}</td>
                        <td className="py-2.5 px-4 font-sans">

                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                              item.side === "before"
                                ? "bg-red-500/20 text-red-300 border border-red-500/30"
                                : "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                            }`}
                          >
                            {item.side}
                          </span>
                        </td>
                        <td className="py-2.5 px-4 font-sans text-slate-300">{item.class_label}</td>
                        <td className="py-2.5 px-4 text-slate-400">{item.source}</td>
                        <td className="py-2.5 px-4 text-right text-slate-400 font-mono">
                          [{item.bbox.join(", ")}]
                        </td>
                        <td className="py-2.5 px-4 text-center font-sans">
                          <span className="text-slate-400 italic">Chưa tìm được tương ứng</span>
                        </td>
                      </tr>
                    );
                  }
                )
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
