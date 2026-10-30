import { useState, useEffect, useMemo } from "react";
import type {
  MatchingResult,
  MatchItem,
  UnmatchedItem,
  ViewMode,
  FilterState,
} from "./types";
import { adaptResult } from "./pipelineAdapter";
import { Header } from "./components/Header";
import { SummaryCards } from "./components/SummaryCards";
import { FilterBar } from "./components/FilterBar";
import { LegendBar } from "./components/LegendBar";
import { SideBySideView } from "./components/SideBySideView";
import { OverlayView } from "./components/OverlayView";
import { ChangeTable } from "./components/ChangeTable";
import { ReportModal } from "./components/ReportModal";
import { AlertCircle, Loader2 } from "lucide-react";

export function App() {
  const [coverageMode, setCoverageMode] = useState("union_support");
  const [pairIds, setPairIds] = useState<string[]>([]);
  const [pairId, setPairId] = useState("000");
  const [includeUncertain, setIncludeUncertain] = useState(false);
  const [evaluation, setEvaluation] = useState<{covered:number;gt_count:number;pass:boolean;predictions:number;protocol:string} | null>(null);
  const [data, setData] = useState<MatchingResult | null>(null);
  const [reportText, setReportText] = useState<string>("");
  const [rawJsonText, setRawJsonText] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // View & Filter states
  const [viewMode, setViewMode] = useState<ViewMode>("side-by-side");
  const [selectedItem, setSelectedItem] = useState<MatchItem | UnmatchedItem | null>(null);
  const [hoveredItemId, setHoveredItemId] = useState<string | null>(null);

  // Modal states
  const [modalType, setModalType] = useState<"none" | "report" | "json">("none");

  // Initial Filter State (Default preset: shift_gt_8)
  const initialFilterState: FilterState = {
    preset: "edges8",
    minShiftPx: 8,
    minSizePct: 15,
    tier: "all",
    searchQuery: "",
    hideMinorChanges: false,
    showUnmatchedBefore: true,
    showUnmatchedAfter: true,
  };

  const [filterState, setFilterState] = useState<FilterState>(initialFilterState);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/data/geometry-v3/manifest.json', {signal:controller.signal})
      .then(r => { if (!r.ok) throw new Error('Không thể tải danh sách geometry'); return r.json(); })
      .then(m => setPairIds(m.pair_ids))
      .catch(e => { if (e.name !== 'AbortError') setError(String(e)); });
    return () => controller.abort();
  }, []);

  // Load selected evidence-pipeline result
  useEffect(() => {
    const controller = new AbortController();
    async function loadSampleData() {
      try {
        setLoading(true);
        setError(null);

        // Fetch matching_result_full.json
        const jsonRes = await fetch(`/data/geometry-v3/${pairId}/matching_result_full.json`, {signal:controller.signal});
        if (!jsonRes.ok) {
          throw new Error(`Không thể tải matching_result_full.json (HTTP ${jsonRes.status})`);
        }
        const raw = await jsonRes.json();
        const jsonData: MatchingResult = adaptResult(raw, includeUncertain);
        setSelectedItem(null);
        setData(jsonData);
        setRawJsonText(JSON.stringify(raw, null, 2));

        const evRes = await fetch(`/data/geometry-v3/${pairId}/eval_${coverageMode}_edges8.json`, {signal:controller.signal});
        setEvaluation(evRes.ok ? await evRes.json() : null);
        // Fetch REPORT.md
        const reportRes = await fetch("/data/geometry-v3/group_REPORT.md", {signal:controller.signal});
        if (reportRes.ok) {
          const reportTxt = await reportRes.text();
          setReportText(reportTxt);
        } else {
          setReportText("Không tìm thấy file report.txt");
        }
      } catch (err: any) {
        if (err.name === "AbortError") return;
        console.error("Lỗi khi tải dữ liệu sample 000:", err);
        setError(err.message || "Đã xảy ra lỗi khi nạp dữ liệu.");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    loadSampleData();
    return () => controller.abort();
  }, [pairId, includeUncertain, coverageMode]);

  // Filter calculation logic
  const filteredMatches = useMemo(() => {
    if (!data) return [];

    return data.matches.filter((match) => {
      // 1. Calculate shift and size metrics
      const { dx, dy, relative_dw: dw, relative_dh: dh } = match.delta;
      const shiftDist = Math.sqrt(dx * dx + dy * dy);
      const dwPct = Math.abs(dw) * 100;
      const dhPct = Math.abs(dh) * 100;
      const maxChangePct = Math.max(dwPct, dhPct);

      if (filterState.preset === "edges8" && !(shiftDist > 8 || maxChangePct > 15 || (match.delta.max_edge_shift ?? 0) > 8)) return false;
      // 2. Hide minor changes check (<0.5px and <1%)
      if (filterState.hideMinorChanges && shiftDist < 0.5 && maxChangePct < 1) {
        return false;
      }

      // 3. Preset filtering
      if (filterState.preset === "shift_gt_8" && shiftDist <= filterState.minShiftPx) {
        return false;
      }
      if (filterState.preset === "size_gt_15" && maxChangePct <= filterState.minSizePct) {
        return false;
      }
      if (filterState.preset === "unmatched") {
        return false; // unmatched tab/view handles this
      }

      // 4. Tier filtering
      if (filterState.tier !== "all" && match.match_type !== filterState.tier) {
        return false;
      }

      // 5. Search query filtering
      if (filterState.searchQuery.trim() !== "") {
        const query = filterState.searchQuery.toLowerCase();
        const bIdMatch = match.before_id.toLowerCase().includes(query);
        const aIdMatch = match.after_id.toLowerCase().includes(query);
        const descMatch = match.description?.some((d) => d.toLowerCase().includes(query));
        const classMatch =
          match.before_class.toLowerCase().includes(query) ||
          match.after_class.toLowerCase().includes(query);

        if (!bIdMatch && !aIdMatch && !descMatch && !classMatch) {
          return false;
        }
      }

      return true;
    });
  }, [data, filterState]);

  // Unmatched Before items filtered
  const filteredUnmatchedBefore = useMemo(() => {
    if (!data || !filterState.showUnmatchedBefore) return [];

    return data.unmatched_before.filter((item) => {
      if (filterState.searchQuery.trim() !== "") {
        const q = filterState.searchQuery.toLowerCase();
        return item.id.toLowerCase().includes(q) || item.class_label.toLowerCase().includes(q);
      }
      return true;
    });
  }, [data, filterState]);

  // Unmatched After items filtered
  const filteredUnmatchedAfter = useMemo(() => {
    if (!data || !filterState.showUnmatchedAfter) return [];
    return data.unmatched_after.filter((item) => {
      if (filterState.searchQuery.trim() !== "") {
        const q = filterState.searchQuery.toLowerCase();
        return item.id.toLowerCase().includes(q) || item.class_label.toLowerCase().includes(q);
      }
      return true;
    });
  }, [data, filterState]);

  const handleResetFilters = () => {
    setFilterState(initialFilterState);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
        <Loader2 className="w-12 h-12 text-cyan-400 animate-spin mb-4" />
        <h2 className="text-xl font-bold text-slate-200">Đang nạp báo cáo mẫu {pairId}...</h2>
        <p className="text-slate-400 text-sm mt-1">Đọc dữ liệu từ pipeline geometry matching</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
        <div className="bg-red-950/80 border border-red-800 p-8 rounded-2xl max-w-lg text-center shadow-2xl">
          <AlertCircle className="w-14 h-14 text-red-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-red-200">Không thể hiển thị báo cáo</h2>
          <p className="text-red-300 text-sm mt-2 font-mono bg-slate-900/80 p-3 rounded-lg border border-red-900/60">
            {error || "Thiếu dữ liệu matching"}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="mt-6 px-6 py-2.5 bg-red-600 hover:bg-red-500 text-white font-medium rounded-xl transition shadow-lg"
          >
            Tải lại trang
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Top Header */}
      <Header
        pairId={data.pair_id || "000"}
        viewMode={viewMode}
        setViewMode={setViewMode}
        onOpenReportModal={() => setModalType("report")}
        onOpenJsonModal={() => setModalType("json")}
      />

      {/* Main Content Workspace */}
      <main className="max-w-[1700px] mx-auto px-4 py-6 flex flex-col gap-6">
        <div className="flex flex-wrap gap-4 items-center">
          <label>Đánh giá <select value={coverageMode} onChange={e => setCoverageMode(e.target.value)}><option value="union_support">Nhóm: hợp diện tích thực</option><option value="single_envelope">Envelope đơn</option><option value="structural_proxy">Nhóm: structural proxy (thử nghiệm)</option></select></label>
          <a href="/group-review/" target="_blank" rel="noreferrer">Review nhóm và thành viên</a>
          <label>Mẫu <select aria-label="Mẫu" value={pairId} onChange={e => setPairId(e.target.value)}>{pairIds.map(id => <option key={id} value={id}>{id}</option>)}</select></label>
          <label><input type="checkbox" checked={includeUncertain} onChange={e => setIncludeUncertain(e.target.checked)} /> Hiển thị thêm uncertain (chưa xác nhận)</label>
          {evaluation && <span aria-live="polite">GT {evaluation.covered}/{evaluation.gt_count} · {evaluation.pass ? 'PASS' : 'FAIL'} · {coverageMode === "single_envelope" ? "Envelope đơn" : "Group coverage"} · matched only · {evaluation.predictions} predictions</span>}
          <span>Geometry v3 · {data.summary.uncertain_pairs ?? 0} uncertain · {includeUncertain ? "Matched + uncertain" : "Matched only"}</span>
        </div>
        {/* 1. Summary Cards */}
        <SummaryCards summary={data.summary} filteredCount={filteredMatches.length} />

        {/* 2. Filter Bar */}
        <FilterBar
          filterState={filterState}
          onFilterChange={setFilterState}
          onResetFilters={handleResetFilters}
          filteredCount={filteredMatches.length}
          totalCount={data.matches.length}
        />

        {/* 3. Visual Color Legend */}
        <LegendBar />

        {/* 4. Main Image Visualizer View (Side-by-Side vs Overlay) */}
        {viewMode === "side-by-side" ? (
          <SideBySideView
            beforeImgUrl={`/data/geometry-v3/${pairId}/before.png`}
            afterImgUrl={`/data/geometry-v3/${pairId}/after.png`}
            matches={data.matches}
            unmatchedBefore={data.unmatched_before}
            unmatchedAfter={data.unmatched_after}
            filteredMatches={filteredMatches}
            filteredUnmatchedBefore={filteredUnmatchedBefore}
            filteredUnmatchedAfter={filteredUnmatchedAfter}
            selectedItem={selectedItem}
            hoveredItemId={hoveredItemId}
            onSelectItem={setSelectedItem}
            onHoverItem={setHoveredItemId}
            filterState={filterState}
          />
        ) : (
          <OverlayView
            beforeImgUrl={`/data/geometry-v3/${pairId}/before.png`}
            afterImgUrl={`/data/geometry-v3/${pairId}/after.png`}
            matches={data.matches}
            unmatchedBefore={data.unmatched_before}
            unmatchedAfter={data.unmatched_after}
            filteredMatches={filteredMatches}
            filteredUnmatchedBefore={filteredUnmatchedBefore}
            filteredUnmatchedAfter={filteredUnmatchedAfter}
            selectedItem={selectedItem}
            hoveredItemId={hoveredItemId}
            onSelectItem={setSelectedItem}
            onHoverItem={setHoveredItemId}
            filterState={filterState}
          />
        )}

        {/* 5. Interactive Change Table */}
        <ChangeTable
          filteredMatches={filteredMatches}
          filteredUnmatchedBefore={filteredUnmatchedBefore}
          filteredUnmatchedAfter={filteredUnmatchedAfter}
          selectedItem={selectedItem}
          hoveredItemId={hoveredItemId}
          onSelectItem={setSelectedItem}
          onHoverItem={setHoveredItemId}
        />
      </main>

      {/* Modals for Diagnostics & Raw Data */}
      <ReportModal
        isOpen={modalType === "report"}
        onClose={() => setModalType("none")}
        title="Báo cáo chẩn đoán pipeline (REPORT.md)"
        type="text"
        content={reportText}
      />

      <ReportModal
        isOpen={modalType === "json"}
        onClose={() => setModalType("none")}
        title="Dữ liệu gốc JSON (matching_result_full.json)"
        type="json"
        content={rawJsonText}
      />
    </div>
  );
}

export default App;
