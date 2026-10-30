import React from "react";
import { X, FileText, Code } from "lucide-react";

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  type: "text" | "json";
  content: string;
}

export const ReportModal: React.FC<ReportModalProps> = ({
  isOpen,
  onClose,
  title,
  type,
  content,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[85vh] shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="p-4 bg-slate-800/80 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-200 font-semibold text-base">
            {type === "text" ? (
              <FileText className="w-5 h-5 text-cyan-400" />
            ) : (
              <Code className="w-5 h-5 text-emerald-400" />
            )}
            <span>{title}</span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto font-mono text-xs text-slate-300 bg-slate-950/90 leading-relaxed scrollbar-thin scrollbar-thumb-slate-800">
          {type === "json" ? (
            <pre className="whitespace-pre-wrap text-emerald-300">{content}</pre>
          ) : (
            <pre className="whitespace-pre-wrap text-slate-200">{content}</pre>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-4 bg-slate-900 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl text-sm font-medium transition"
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
};
