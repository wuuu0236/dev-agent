import React, { useState, useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";

/* ============================================================
 * 数据层：优先调后端 /api（同源部署），后端不可达时回退演示数据
 * 这样单独双击 index.html 也能演示，起服务后自动切真数据
 * ============================================================ */

const apiGet = async (path) => {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
};
const apiPost = async (path, body) => {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${path}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
};
const apiDelete = async (path) => {
  const r = await fetch(path, { method: "DELETE" });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
};

/* ---------- 演示数据（后端不可达时使用） ---------- */
const MOCK_KBS = [
  { id: "af14d071", name: "DataLens 演示", description: "预置演示知识库", doc_count: 1, total_chunks: 14 },
  { id: "16684ef4", name: "RAG 评测语料库", description: "项目内全部 Markdown 文档", doc_count: 14, total_chunks: 127 },
];
const MOCK_DOCS = [
  { id: "m1", name: "rag_agent_guide.md", chunks: 14, status: "ready", date: "2026-08-12 22:11", size: 12242 },
  { id: "m2", name: "hybrid_retriever 设计说明.md", chunks: 18, status: "ready", date: "2026-09-11", size: 8600 },
  { id: "m3", name: "基金定期报告 2026Q2.pdf", chunks: 156, status: "ready", date: "2026-09-08", size: 2510000 },
  { id: "m4", name: "扫描合同_租赁.pdf", chunks: 64, status: "processing", date: "2026-09-14", size: 1800000 },
];
const MOCK_SOURCES = [
  { n: 1, file: "rag_agent_guide.md", page: 0, snippets: ["本手册介绍构建「知识库问答系统」（RAG）与「文件操作 Agent」的核心原理…"] },
];
const MOCK_EVAL = {
  timestamp: "2026-09-12",
  test_size: 26,
  results: [
    { name: "粗排 only", metrics: { context_precision: 84, context_recall: 90, faithfulness: 82, answer_relevancy: 85 } },
    { name: "精排 top5", metrics: { context_precision: 92, context_recall: 96, faithfulness: 89, answer_relevancy: 91 } },
    { name: "精排 + BM25", metrics: { context_precision: 93, context_recall: 96, faithfulness: 90, answer_relevancy: 91 } },
  ],
};
const MOCK_LOG = {
  stats: {
    total: 5, grounded: 3, ungrounded: 2, cache_hits: 1,
    near_miss: [
      { id: 1, question: "RAGAS 的 faithfulness 指标具体怎么算的？", gate_score: 0.26, created_at: "09-14 21:32" },
      { id: 2, question: "怎么配置 Ollama 本地模型离线运行？", gate_score: 0.21, created_at: "09-14 20:11" },
    ],
    no_neighbor: [
      { id: 3, question: "去年股票大盘走势如何？", gate_score: 0.04, created_at: "09-14 18:47" },
      { id: 4, question: "今天天气怎么样？", gate_score: 0.02, created_at: "09-13 19:30" },
    ],
  },
};

const SUGGESTIONS = [
  "混合检索为什么要 BM25 + 向量两路都开？",
  "这个项目包含哪些内容？",
  "引用是怎么做到可核实的？",
];

const fmtSize = (b) => (b > 1024 * 1024 ? (b / 1024 / 1024).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");
const STATUS_MAP = { ready: "ready", processing: "processing", empty: "failed", error: "failed" };

/* ============================================================
 * 通用小组件
 * ============================================================ */
const StatusTag = ({ s }) => (
  <span className={"status " + s}>{s === "ready" ? "已就绪" : s === "processing" ? "解析中" : "异常"}</span>
);

function Radar({ metrics }) {
  const cx = 160, cy = 150, R = 105;
  const angle = (i) => (Math.PI * 2 * i) / 4 - Math.PI / 2;
  const pt = (i, r) => [cx + r * Math.cos(angle(i)), cy + r * Math.sin(angle(i))];
  const poly = (vals, r) => vals.map((v, i) => pt(i, (v / 100) * r).join(",")).join(" ");
  const vals = metrics.map((m) => m.value);
  return (
    <svg width="320" height="290" viewBox="0 0 320 290">
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <polygon key={f} points={poly([100, 100, 100, 100], R * f)} fill="none" stroke="#e7e9f2" strokeWidth="1" />
      ))}
      {[0, 1, 2, 3].map((i) => {
        const [x, y] = pt(i, R);
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#e7e9f2" strokeWidth="1" />;
      })}
      <polygon points={poly(vals, R)} fill="rgba(99,102,241,.18)" stroke="#6366f1" strokeWidth="2.5" strokeLinejoin="round" />
      {vals.map((v, i) => {
        const [x, y] = pt(i, (v / 100) * R);
        return <circle key={i} cx={x} cy={y} r="4.5" fill="#6366f1" stroke="#fff" strokeWidth="2" />;
      })}
      {[0, 1, 2, 3].map((i) => {
        const [x, y] = pt(i, R + 26);
        return (
          <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize="12" fontWeight="700" fill="#5b6072">
            {metrics[i].name.slice(0, 5)}
          </text>
        );
      })}
      {vals.map((v, i) => {
        const [x, y] = pt(i, (v / 100) * R);
        return <text key={i} x={x} y={y - 12} textAnchor="middle" fontSize="10.5" fontWeight="800" fill="#6366f1">{v}</text>;
      })}
    </svg>
  );
}

/* ============================================================
 * 智能问答
 * ============================================================ */
function ChatPage({ apiReady, kbId, kbName }) {
  const hello = apiReady === false
    ? "后端未连接，当前为演示数据。启动 `python src/api/server.py` 后刷新即可接真数据。"
    : apiReady === true
      ? `已连接后端，当前知识库：${kbName}。可以开始提问了。`
      : "正在连接后端…";
  const [msgs, setMsgs] = useState([{ role: "ai", content: hello, citations: [], meta: null }]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [expanded, setExpanded] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [msgs, typing]);

  const send = async (text) => {
    const q = (text || input).trim();
    if (!q || typing) return;
    setInput("");
    setMsgs((m) => [...m, { role: "user", content: q }]);
    setTyping(true);
    const t0 = Date.now();

    if (apiReady !== true) {
      setTimeout(() => {
        setTyping(false);
        setMsgs((m) => [...m, {
          role: "ai",
          content: "（演示数据）混合检索同时开启 BM25 + 稠密向量：向量擅长语义匹配，BM25 擅长专有名词的词面精确匹配，两路 RRF 融合后截断回候选池，再由交叉编码精排取 top5。",
          citations: MOCK_SOURCES.map((s) => ({ n: s.n, file: s.file, page: "P1", score: "0.91", snip: s.snippets[0] })),
          meta: { gate: "通过", topScore: "0.91", rerank: "bge-reranker-v2-m3 · top5", cache: false, latency: "—" },
        }]);
      }, 900);
      return;
    }

    try {
      const history = msgs
        .filter((m) => m.content)
        .slice(-6)
        .map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content }));
      const r = await apiPost("/api/ask", { kb_id: kbId, question: q, history });
      const latency = ((Date.now() - t0) / 1000).toFixed(1) + "s";
      setTyping(false);
      setMsgs((m) => [...m, {
        role: "ai",
        content: r.answer,
        citations: (r.sources || []).map((s) => ({
          n: s.n, file: s.file, page: s.page ? `P${s.page}` : "",
          score: r.gate_score != null ? Number(r.gate_score).toFixed(2) : "—",
          snip: s.snippets && s.snippets[0] ? s.snippets[0] : "",
        })),
        meta: {
          gate: r.grounded ? "通过" : "已拒答",
          topScore: r.gate_score != null ? Number(r.gate_score).toFixed(2) : "—",
          rerank: "bge-reranker-v2-m3 · top5",
          latency,
          rewritten: r.retrieval_query && r.retrieval_query !== q ? r.retrieval_query : null,
        },
      }]);
    } catch (e) {
      setTyping(false);
      setMsgs((m) => [...m, { role: "ai", content: "请求失败：" + e.message, citations: [], meta: null }]);
    }
  };

  return (
    <div className="chat-layout" style={{ height: "100%" }}>
      <div className="card chat-panel chat-col">
        <div className="chat-msgs">
          {msgs.map((m, i) => (
            <div key={i} className={"msg " + (m.role === "user" ? "user" : "ai")}>
              <div className="avatar">{m.role === "user" ? "🧑‍💻" : "🧠"}</div>
              <div>
                <div className="bubble">
                  {m.content.split("\n").map((line, j) => (
                    <div key={j} style={{ whiteSpace: "pre-wrap" }}>{line.replace(/\*\*/g, "")}</div>
                  ))}
                </div>
                {m.meta && (
                  <div className="msg-meta">
                    <span className={"meta-chip " + (m.meta.gate === "通过" ? "hit" : "gate")}>门控：{m.meta.gate}</span>
                    <span className="meta-chip">top1 精排分 {m.meta.topScore}</span>
                    <span className="meta-chip">{m.meta.rerank}</span>
                    <span className="meta-chip">耗时 {m.meta.latency}</span>
                    {m.meta.rewritten && <span className="meta-chip">改写检索：{m.meta.rewritten}</span>}
                  </div>
                )}
                {m.citations && m.citations.length > 0 && (
                  <div className="citations">
                    {m.citations.map((c) => (
                      <div key={c.n} className="cite-item" onClick={() => setExpanded(expanded === i + "-" + c.n ? null : i + "-" + c.n)}>
                        <div className="cite-head">
                          <sup className="cite">[{c.n}]</sup>
                          <span className="file">{c.file}</span>
                          <span className="pg">{c.page}</span>
                          <span className="score">精排 {c.score}</span>
                        </div>
                        {expanded === i + "-" + c.n
                          ? <div className="snip">「{c.snip || "（无原文片段）"}」</div>
                          : <div className="snip" style={{ color: "var(--text-3)" }}>点击展开原文片段…</div>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {typing && (
            <div className="msg ai">
              <div className="avatar">🧠</div>
              <div className="bubble"><span className="typing-dots"><span></span><span></span><span></span></span></div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        <div className="chat-input-wrap">
          <div className="suggest-row">
            {SUGGESTIONS.map((s) => <span key={s} className="suggest" onClick={() => send(s)}>{s}</span>)}
          </div>
          <div className="input-row">
            <input value={input} placeholder="向知识库提问…（追问会自动改写消解）"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()} />
            <button className="send-btn" onClick={() => send()} disabled={typing || !kbId}>发送</button>
          </div>
        </div>
      </div>

      <div className="side-col">
        <div className="card side-card">
          <h3>⚙️ 检索流水线</h3>
          <div className="pipeline">
            {[
              { t: "查询改写", d: "追问自动消解指代，压成自包含查询" },
              { t: "混合召回", d: "BM25 + 向量 → RRF 融合 → 候选池 50" },
              { t: "交叉编码精排", d: "bge-reranker-v2-m3 逐条打分取 top5" },
              { t: "阈值门控", d: "top1 < 0.3 拒答，敢说不知道" },
              { t: "生成 + 引用", d: "序号由代码映射回真实 chunk，防编造" },
            ].map((s, i, arr) => (
              <div className="pipe-step" key={i}>
                <div className="pipe-rail"><div className="pipe-dot"></div>{i < arr.length - 1 && <div className="pipe-line"></div>}</div>
                <div className="pipe-body"><div className="t">{s.t}</div><div className="d">{s.d}</div></div>
              </div>
            ))}
          </div>
        </div>
        <div className="card side-card">
          <h3>🔌 连接状态</h3>
          <div style={{ fontSize: 12.5, lineHeight: 1.8 }}>
            <div>后端：{apiReady === true ? <b style={{ color: "var(--green)" }}>已连接</b>
              : apiReady === false ? <b style={{ color: "var(--orange)" }}>未连接（演示数据）</b>
                : "检测中…"}</div>
            <div>知识库：{kbName || "—"}</div>
            <div style={{ color: "var(--text-3)", marginTop: 6 }}>
              未连接时启动后端即可：<code>python src/api/server.py</code>，前端由后端同源托管在 /app。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
 * 知识库管理
 * ============================================================ */
function KBPage({ apiReady, kbs, kbId, setKbId, docs, stats, refresh, notify }) {
  const kbName = (kbs.find((k) => k.id === kbId) || {}).name || "—";
  const [busy, setBusy] = useState(false);

  const reindex = async () => {
    if (apiReady !== true) return notify("演示模式：重建索引需要连接后端");
    setBusy(true);
    try {
      const r = await apiPost(`/api/kbs/${kbId}/reindex`);
      notify(`重建完成：成功 ${r.rebuilt ?? 0} 个，失败 ${r.failed ?? 0} 个${r.missing ? `，缺原文 ${r.missing.length} 个` : ""}`);
      refresh();
    } catch (e) { notify("重建失败：" + e.message); }
    setBusy(false);
  };

  const remove = async (doc) => {
    if (apiReady !== true) return notify("演示模式：删除需要连接后端");
    if (!window.confirm(`确认删除《${doc.name}》？SQLite 记录与向量都会清除。`)) return;
    try {
      const r = await apiDelete(`/api/docs/${doc.id}`);
      notify(`已删除 ${r.filename}，清除 ${r.deleted_chunks} 个 chunk`);
      refresh();
    } catch (e) { notify("删除失败：" + e.message); }
  };

  return (
    <div>
      <div className="stat-grid">
        {[
          { label: "文档总数", value: (stats && stats.doc_count) ?? docs.length, hint: "当前知识库" },
          { label: "chunk 总数", value: (stats && stats.total_chunks) ?? docs.reduce((a, d) => a + (d.chunks || 0), 0), hint: "滑窗 500 字 / 重叠 50 字" },
          { label: "向量索引", value: "Chroma", hint: "HNSW · 余弦相似度" },
          { label: "BM25 索引", value: "jieba", hint: "惰性构建 · 停用词过滤" },
        ].map((s, i) => (
          <div className="card stat-card" key={i}>
            <div className="label">{s.label}</div>
            <div className="value">{s.value}</div>
            <div className="hint">{s.hint}</div>
          </div>
        ))}
      </div>
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", padding: "16px 18px 0", gap: 10 }}>
          <div className="section-title" style={{ marginBottom: 0 }}>📄 文档列表 · {kbName}</div>
          <select value={kbId || ""} onChange={(e) => setKbId(e.target.value)}
            style={{ border: "1px solid var(--border)", borderRadius: 8, padding: "6px 10px", fontSize: 13 }}>
            {kbs.map((k) => <option key={k.id} value={k.id}>{k.name}</option>)}
          </select>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
            <button className="btn ghost sm" onClick={reindex} disabled={busy}>{busy ? "重建中…" : "🔄 重建索引"}</button>
            <button className="btn primary sm" onClick={() => notify("请到「文档上传」页上传文档")}>＋ 上传文档</button>
          </div>
        </div>
        <table style={{ marginTop: 8 }}>
          <thead>
            <tr><th>文件名</th><th>chunks</th><th>大小</th><th>状态</th><th>上传时间</th><th></th></tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.id}>
                <td style={{ fontWeight: 600 }}>{d.name}</td>
                <td>{d.chunks}</td>
                <td className="muted">{d.size ? fmtSize(d.size) : "—"}</td>
                <td><StatusTag s={d.status} /></td>
                <td className="muted">{d.date}</td>
                <td style={{ textAlign: "right" }}>
                  <button className="icon-btn" title="删除" onClick={() => remove(d)}>🗑</button>
                </td>
              </tr>
            ))}
            {docs.length === 0 && (
              <tr><td colSpan="6" className="empty-tip">该知识库还没有文档</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ============================================================
 * 文档上传
 * ============================================================ */
function UploadPage({ apiReady, kbId, kbs, refresh, notify }) {
  const [items, setItems] = useState([]);
  const [over, setOver] = useState(false);
  const fileRef = useRef(null);

  const uploadOne = (file) => {
    const item = { name: file.name, size: fmtSize(file.size), progress: 0, status: "uploading", note: "" };
    setItems((it) => [item, ...it]);
    const patch = (p) => setItems((it) => it.map((x) => (x === item ? { ...x, ...p } : x)));

    if (apiReady !== true) {
      setTimeout(() => patch({ progress: 100, status: "done", note: "演示模式：未真正入库" }), 800);
      return;
    }
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/kbs/${kbId}/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) patch({ progress: Math.round((e.loaded / e.total) * 100) });
    };
    xhr.onload = () => {
      try {
        const r = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) {
          const ocr = r.image_chunks ? `（含 ${r.image_chunks} 个 OCR 块）` : "";
          patch({ progress: 100, status: r.status === "ready" ? "done" : "failed",
                  note: `${r.paragraphs ?? 0} 段 → ${r.chunks ?? 0} chunks${ocr}` });
          refresh();
        } else {
          patch({ status: "failed", note: String(r.detail || "上传失败").slice(0, 60) });
        }
      } catch { patch({ status: "failed", note: "响应解析失败" }); }
    };
    xhr.onerror = () => patch({ status: "failed", note: "网络错误" });
    xhr.send(form);
  };

  const addFiles = (files) => { [...files].forEach(uploadOne); };

  return (
    <div className="upload-layout">
      <div>
        <div className={"dropzone " + (over ? "over" : "")}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); addFiles(e.dataTransfer.files); }}>
          <div className="big">📤</div>
          <h3>拖拽文件到这里，或点击选择</h3>
          <p>支持 PDF / Word / Markdown / TXT / CSV / 图片 · 扫描版自动 OCR · 单文件 ≤ 20MB</p>
          <input ref={fileRef} type="file" multiple style={{ display: "none" }}
            onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
        </div>
        <div className="card" style={{ marginTop: 16 }}>
          <div style={{ padding: "16px 18px 4px" }} className="section-title">📥 上传队列</div>
          <div style={{ padding: "0 18px 10px" }}>
            {items.map((u, i) => (
              <div className="upload-item" key={i}>
                <div className="file-ico" style={{ background: u.status === "failed" ? "var(--red-soft)" : "var(--green-soft)" }}>
                  {u.status === "failed" ? "⚠️" : "📄"}
                </div>
                <div className="u-info">
                  <div className="u-name">{u.name}</div>
                  <div className="u-sub">{u.size} · {u.note || (u.status === "uploading" ? "上传中…" : "")}</div>
                  {u.status === "uploading" && <div className="progress"><div className="progress-bar" style={{ width: u.progress + "%" }}></div></div>}
                </div>
                {u.status === "done" ? <StatusTag s="ready" /> : u.status === "failed" ? <StatusTag s="failed" /> : <span className="muted" style={{ fontWeight: 700 }}>{u.progress}%</span>}
              </div>
            ))}
            {items.length === 0 && <div className="empty-tip">暂无上传任务</div>}
          </div>
        </div>
      </div>
      <div className="card side-card" style={{ alignSelf: "flex-start", width: "100%" }}>
        <h3>⚙️ 入库流程</h3>
        <div className="pipeline">
          {[
            { t: "保存原文", d: "按知识库分目录留存，是日后重建索引的唯一素材" },
            { t: "解析 + 清洗", d: "页眉页脚去除、硬换行合并、垃圾块过滤；扫描件走 OCR" },
            { t: "分块", d: "滑窗 500 字 / 重叠 50 字，碎块并入相邻块" },
            { t: "嵌入入库", d: "bge-large-zh-v1.5 分批写入 Chroma，同文件重传自动替换" },
          ].map((s, i, arr) => (
            <div className="pipe-step" key={i}>
              <div className="pipe-rail"><div className="pipe-dot"></div>{i < arr.length - 1 && <div className="pipe-line"></div>}</div>
              <div className="pipe-body"><div className="t">{s.t}</div><div className="d">{s.d}</div></div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 14, fontSize: 12, color: "var(--text-3)", lineHeight: 1.7 }}>
          当前知识库：{((kbs.find((k) => k.id === kbId)) || {}).name || "—"}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
 * 评估面板
 * ============================================================ */
function EvalPage({ evalData }) {
  const data = evalData || MOCK_EVAL;
  const results = data.results || [];
  const latest = results[results.length - 1] || { metrics: {} };
  const m = latest.metrics || {};
  const metrics = [
    { key: "cp", name: "上下文精确率", value: Math.round(m.context_precision || 0), desc: "检索到的 chunk 有多少真的有用" },
    { key: "cr", name: "上下文召回率", value: Math.round(m.context_recall || 0), desc: "该找到的参考内容找到了多少" },
    { key: "fa", name: "忠实度", value: Math.round(m.faithfulness || 0), desc: "回答是否严格基于检索内容" },
    { key: "ar", name: "答案相关性", value: Math.round(m.answer_relevancy || 0), desc: "回答是否切题" },
  ];
  const avg = Math.round(metrics.reduce((a, x) => a + x.value, 0) / 4);

  return (
    <div>
      <div className="stat-grid">
        <div className="card stat-card"><div className="label">最新配置综合均分</div><div className="value" style={{ color: "var(--accent)" }}>{avg}</div><div className="hint">{latest.name || "—"}</div></div>
        <div className="card stat-card"><div className="label">测试集规模</div><div className="value">{data.test_size || "—"}</div><div className="hint">RAGAS 四维评估</div></div>
        <div className="card stat-card"><div className="label">配置数量</div><div className="value">{results.length}</div><div className="hint">同测试集对比</div></div>
        <div className="card stat-card"><div className="label">评估时间</div><div className="value" style={{ fontSize: 18 }}>{(data.timestamp || "—").slice(0, 10)}</div><div className="hint">本地历史存档</div></div>
      </div>
      <div className="eval-layout">
        <div className="card" style={{ padding: "18px 20px" }}>
          <div className="section-title">🎯 RAGAS 四维指标 · {latest.name || "—"}</div>
          <div className="radar-wrap"><Radar metrics={metrics} /></div>
          <div className="metric-list">
            {metrics.map((x) => (
              <div key={x.key}>
                <div className="metric-row">
                  <span className="name">{x.name}</span>
                  <div className="metric-bar"><div className="metric-fill" style={{ width: x.value + "%" }}></div></div>
                  <span className="val">{x.value}</span>
                </div>
                <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 3, paddingLeft: 160 }}>{x.desc}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <div style={{ padding: "16px 18px 0" }} className="section-title">🔬 配置对比（同测试集）</div>
          <table className="compare-table" style={{ marginTop: 6 }}>
            <thead><tr><th>配置</th><th>精确率</th><th>召回率</th><th>忠实度</th><th>相关性</th></tr></thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} style={i === results.length - 1 ? { background: "var(--accent-soft)" } : {}}>
                  <td style={{ fontWeight: 600 }}>{r.name}
                    {i === results.length - 1 && <span className="tag" style={{ marginLeft: 6, background: "var(--accent)", color: "#fff" }}>最新</span>}</td>
                  <td>{Math.round((r.metrics && r.metrics.context_precision) || 0)}</td>
                  <td>{Math.round((r.metrics && r.metrics.context_recall) || 0)}</td>
                  <td>{Math.round((r.metrics && r.metrics.faithfulness) || 0)}</td>
                  <td>{Math.round((r.metrics && r.metrics.answer_relevancy) || 0)}</td>
                </tr>
              ))}
              {results.length === 0 && <tr><td colSpan="5" className="empty-tip">暂无评估存档</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
 * 问答日志
 * ============================================================ */
function LogPage({ logData }) {
  const [filter, setFilter] = useState("all");
  const data = logData || MOCK_LOG;
  const s = data.stats || {};
  const near = s.near_miss || [], none = s.no_neighbor || [];
  const shown = filter === "near" ? near : filter === "none" ? none : [...near, ...none];

  return (
    <div>
      <div className="stat-grid">
        <div className="card stat-card"><div className="label">问答总数</div><div className="value">{s.total ?? "—"}</div><div className="hint">线上真实提问</div></div>
        <div className="card stat-card"><div className="label">有知识库支撑</div><div className="value">{s.grounded ?? "—"}</div><div className="hint">通过门控</div></div>
        <div className="card stat-card"><div className="label">未达阈值</div><div className="value">{s.ungrounded ?? "—"}</div><div className="hint">如实说没找到</div></div>
        <div className="card stat-card"><div className="label">缓存命中</div><div className="value">{s.cache_hits ?? "—"}</div><div className="hint">相似问题秒回</div></div>
      </div>
      <div className="filter-row">
        {[["all", "全部"], ["near", `差点过阈 → 调检索（${near.length}）`], ["none", `无语义邻居 → 补文档（${none.length}）`]].map(([k, t]) => (
          <span key={k} className={"filter-chip " + (filter === k ? "on" : "")} onClick={() => setFilter(k)}>{t}</span>
        ))}
      </div>
      <div className="card">
        {shown.map((l) => (
          <div className="log-item" key={l.id}>
            <div className="log-score near">{Number(l.gate_score ?? 0).toFixed(2)}<small>最高精排分</small></div>
            <div className="log-body">
              <div className="log-q">{l.question}</div>
              <div className="log-meta">{l.created_at || ""} · {l.gate_score >= 0.099 ? "接近阈值但被拒答" : "连语义邻居都没有"}</div>
              <span className={"log-action " + (l.gate_score >= 0.099 ? "tune" : "adddoc")}>
                {l.gate_score >= 0.099 ? "🔧 建议调检索（纯计算、可反复试）" : "📚 建议补文档（需要人去找资料）"}
              </span>
            </div>
          </div>
        ))}
        {shown.length === 0 && <div className="empty-tip">暂无记录（多问几个库外问题就会积累）</div>}
      </div>
      <div className="muted" style={{ marginTop: 12, lineHeight: 1.7 }}>
        💡 「检索没命中」和「库里根本没有」在系统内部长得一样（都是 200、都写了缓存）。按最高分把未命中问题分成两类待办，避免去调一个根本没调错的参数。
      </div>
    </div>
  );
}

/* ============================================================
 * App
 * ============================================================ */
const NAV = [
  { key: "chat", ico: "💬", label: "智能问答" },
  { key: "kb", ico: "📚", label: "知识库管理" },
  { key: "upload", ico: "📄", label: "文档上传" },
  { key: "eval", ico: "📊", label: "评估面板" },
  { key: "log", ico: "🩺", label: "问答日志" },
];

function App() {
  const [page, setPage] = useState("chat");
  const [apiReady, setApiReady] = useState(null);
  const [kbs, setKbs] = useState(MOCK_KBS);
  const [kbId, setKbId] = useState(MOCK_KBS[0].id);
  const [docs, setDocs] = useState(MOCK_DOCS);
  const [stats, setStats] = useState(null);
  const [logData, setLogData] = useState(null);
  const [evalData, setEvalData] = useState(null);
  const [notice, setNotice] = useState(null);

  const notify = (msg) => { setNotice(msg); setTimeout(() => setNotice(null), 4000); };

  const loadKbData = async (id) => {
    try {
      const d = await apiGet(`/api/kbs/${id}/docs`);
      setStats(d.stats);
      setDocs(d.docs.map((x) => ({
        id: x.id, name: x.filename, chunks: x.chunk_count, status: STATUS_MAP[x.status] || "failed",
        date: x.created_at, size: x.file_size,
      })));
    } catch { setDocs([]); }
    try { setLogData(await apiGet(`/api/answer_log?kb_id=${id}&limit=50`)); } catch { setLogData(null); }
  };

  const loadEval = async () => {
    try {
      const list = await apiGet("/api/eval/history");
      const pick = list.find((x) => (x.configs || []).length > 0) || list[0];
      if (pick) setEvalData(await apiGet("/api/eval/history/" + pick.file));
    } catch { setEvalData(null); }
  };

  useEffect(() => {
    (async () => {
      try {
        await apiGet("/api/health");
        setApiReady(true);
        const ks = await apiGet("/api/kbs");
        if (ks.length) { setKbs(ks); setKbId(ks[0].id); }
      } catch { setApiReady(false); }
    })();
  }, []);

  useEffect(() => { if (apiReady === true && kbId) { loadKbData(kbId); loadEval(); } }, [apiReady, kbId]);

  const pageMeta = {
    chat: ["智能问答", "混合检索 · 精排 · 门控 · 引用可核实"],
    kb: ["知识库管理", "多库隔离 · 索引可重建"],
    upload: ["文档上传", "多模态解析 · 原文留存"],
    eval: ["评估面板", "RAGAS 四维量化 · 配置对比"],
    log: ["问答日志", "反馈闭环 · 缺口自动分类"],
  }[page];
  const kbName = ((kbs.find((k) => k.id === kbId)) || {}).name;

  return (
    <div className="app">
      <div className="sidebar">
        <div className="logo">
          <div className="logo-icon">🧠</div>
          <div>
            <div className="logo-name">DataLens</div>
            <div className="logo-sub">个人知识库 RAG 问答</div>
          </div>
        </div>
        {NAV.map((n) => (
          <div key={n.key} className={"nav-item " + (page === n.key ? "active" : "")} onClick={() => setPage(n.key)}>
            <span className="ico">{n.ico}</span>{n.label}
          </div>
        ))}
        <div className="sidebar-foot">
          <div className="badge-row">
            <span className={"badge " + (apiReady === true ? "" : "v")}>{apiReady === true ? "🟢 后端已连接" : "演示数据"}</span>
            <span className="badge">🔒 私有化</span>
            <span className="badge v">MCP</span>
          </div>
        </div>
      </div>

      <div className="main">
        <div className="topbar">
          <div>
            <h1>{pageMeta[0]}</h1>
            <div className="desc">{pageMeta[1]}</div>
          </div>
          {(page === "chat" || page === "kb" || page === "upload") && (
            <div className="kb-select">📚 {kbName || "—"}</div>
          )}
        </div>
        {notice && (
          <div style={{ margin: "12px 24px -8px", padding: "10px 14px", borderRadius: 10, background: "var(--accent-soft)", color: "var(--accent)", fontSize: 13, fontWeight: 600 }}>
            {notice}
          </div>
        )}
        <div className="content" style={page === "chat" ? { display: "flex", flexDirection: "column" } : {}}>
          {page === "chat" && <ChatPage apiReady={apiReady} kbId={kbId} kbName={kbName} />}
          {page === "kb" && <KBPage apiReady={apiReady} kbs={kbs} kbId={kbId} setKbId={setKbId} docs={docs} stats={stats}
            refresh={() => loadKbData(kbId)} notify={notify} />}
          {page === "upload" && <UploadPage apiReady={apiReady} kbId={kbId} kbs={kbs}
            refresh={() => loadKbData(kbId)} notify={notify} />}
          {page === "eval" && <EvalPage evalData={evalData} />}
          {page === "log" && <LogPage logData={logData} />}
        </div>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
