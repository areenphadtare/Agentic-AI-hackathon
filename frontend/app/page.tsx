'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import type cytoscape from 'cytoscape'
import { Activity, AlertTriangle, Bot, CheckCircle2, CircleDot, CloudUpload, FileVideo, LoaderCircle, Network, Send, Upload, X, Plus, Minus, Maximize2, Copy, Check } from 'lucide-react'

type GraphNode = { data: { id: string; label: string; type: 'video' | 'frame' | 'attribute' | string; meta?: string; value?: string; key?: string; frame_id?: string; video_id?: string; db_id?: number | string } }
type GraphEdge = { data: { id: string; source: string; target: string; label: string; type?: string } }
type GraphData = { video_id: string; graph_name: string; nodes: GraphNode[]; edges: GraphEdge[] }
type Job = { job_id: string; status: 'queued' | 'processing' | 'completed' | 'failed'; stage?: string; error?: string; filename?: string; video_id?: string; graph?: GraphData; summary?: { frames_processed: number; nodes: number; edges: number; video_name: string; duration_seconds?: number }; stdout?: string; stderr?: string }
type ChatMessage = { role: 'assistant' | 'user'; text: string }

const API_BASE = (process.env.NEXT_PUBLIC_ARGUSGRAPH_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

function renderInline(text: string) {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i} className="font-semibold text-slate-100">{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={i} className="rounded bg-slate-800 px-1 py-0.5 text-cyan-200">{part.slice(1, -1)}</code>
    return <span key={i}>{part}</span>
  })
}

function MarkdownAnswer({ text }: { text: string }) {
  // Some model/tool responses arrive with Markdown punctuation escaped (\\##, \\*, \\_).
  // Normalize those escapes before parsing so headings, lists, and emphasis render as UI.
  const normalized = text
    .replace(/\\([#*_`])/g, '$1')
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
  const lines = normalized.split('\n')
  const blocks: JSX.Element[] = []
  let listItems: { text: string; ordered: boolean }[] = []
  const flushList = () => {
    if (!listItems.length) return
    const ordered = listItems[0].ordered
    const items = listItems
    blocks.push(ordered
      ? <ol key={`list-${blocks.length}`} className="my-2 list-decimal space-y-1 pl-5 text-sm leading-6 text-slate-300">{items.map((item, i) => <li key={i}>{renderInline(item.text)}</li>)}</ol>
      : <ul key={`list-${blocks.length}`} className="my-2 list-disc space-y-1 pl-5 text-sm leading-6 text-slate-300">{items.map((item, i) => <li key={i}>{renderInline(item.text)}</li>)}</ul>)
    listItems = []
  }
  for (const raw of lines) {
    const line = raw.trim()
    if (!line) { flushList(); continue }
    const heading = line.match(/^(#{1,3})\s+(.+)$/)
    if (heading) {
      flushList()
      blocks.push(<h3 key={`h-${blocks.length}`} className="mt-4 mb-2 text-base font-semibold text-violet-200 first:mt-0">{renderInline(heading[2].replace(/\*\*/g, ''))}</h3>)
      continue
    }
    const bullet = line.match(/^[-*]\s+(.+)$/)
    if (bullet) { listItems.push({ text: bullet[1], ordered: false }); continue }
    const numbered = line.match(/^\d+[.)]\s+(.+)$/)
    if (numbered) { if (listItems.length && !listItems[0].ordered) flushList(); listItems.push({ text: numbered[1], ordered: true }); continue }
    flushList()
    if (/^---+$/.test(line)) continue
    blocks.push(<p key={`p-${blocks.length}`} className="my-1.5 text-sm leading-6 text-slate-300">{renderInline(line.replace(/^>\s?/, ''))}</p>)
  }
  flushList()
  return <div className="max-w-none space-y-0 [overflow-wrap:anywhere]">{blocks}</div>
}

export default function Page() {
  const [graph, setGraph] = useState<GraphData | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [highlighted, setHighlighted] = useState<string[]>([])
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [chatError, setChatError] = useState('')
  const [sessionId] = useState(() => (typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `argus-${Date.now()}-${Math.random().toString(36).slice(2)}`))
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'assistant', text: 'Ask a question about the videos and evidence stored in FalkorDB. I will use the graph investigation tools to answer.' },
  ])
  const cyRef = useRef<cytoscape.Core | null>(null)
  const lastFittedGraphRef = useRef<GraphData | null>(null)
  const [copiedNodeValue, setCopiedNodeValue] = useState(false)

  const elements = useMemo(() => {
    if (!graph) return []
    const activeIds = new Set(highlighted)
    const activeEdges = new Set(graph.edges.filter(e => activeIds.has(e.data.id) || activeIds.has(e.data.source) || activeIds.has(e.data.target)).map(e => e.data.id))
    return [
      ...graph.nodes.map(n => ({ ...n, classes: selectedNode === n.data.id ? 'selected' : activeIds.has(n.data.id) ? 'highlighted' : highlighted.length ? 'dimmed' : '' })),
      ...graph.edges.map(e => ({ ...e, classes: activeEdges.has(e.data.id) ? 'highlighted' : highlighted.length ? 'dimmed' : '' })),
    ]
  }, [graph, highlighted, selectedNode])

  const stylesheet = useMemo(() => [
    { selector: 'node', style: { label: 'data(label)', color: '#c4cedd', 'font-size': 10, 'font-family': 'Inter, sans-serif', 'text-wrap': 'ellipsis', 'text-max-width': 130, 'text-valign': 'bottom', 'text-margin-y': 9, 'background-color': '#64748b', width: 20, height: 20, 'border-width': 1, 'border-color': '#94a3b8' } },
    { selector: 'node[type="video"]', style: { 'background-color': '#7c5cff', width: 44, height: 44, 'border-width': 2, 'border-color': '#b4a7ff', color: '#f2efff', 'font-size': 12, 'font-weight': 600, 'text-margin-y': 12, 'text-max-width': 170 } },
    { selector: 'node[type="frame"]', style: { 'background-color': '#22b8d6', width: 27, height: 27, 'border-color': '#8be8f5' } },
    { selector: 'node[type="attribute"]', style: { 'background-color': '#475569', width: 17, height: 17, 'border-color': '#94a3b8', 'font-size': 9 } },
    { selector: 'edge', style: { width: 1, 'line-color': '#475569', 'target-arrow-color': '#64748b', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', label: 'data(label)', color: '#94a3b8', 'font-size': 7, 'text-rotation': 'autorotate', 'text-background-color': '#0b1220', 'text-background-opacity': 0.85, 'text-background-padding': 2 } },
    { selector: '.highlighted', style: { 'border-width': 3, 'border-color': '#ecfeff', 'background-color': '#06b6d4', color: '#ecfeff' } },
    { selector: 'edge.highlighted', style: { width: 3, 'line-color': '#22d3ee', 'target-arrow-color': '#22d3ee', color: '#a5f3fc' } },
    { selector: '.dimmed', style: { opacity: 0.22 } },
    { selector: '.selected', style: { 'border-width': 3, 'border-color': '#ffffff', 'overlay-color': '#ffffff', 'overlay-opacity': 0.12 } },
  ] as any, [])
  const graphLayout = useMemo(() => ({ name: 'cose', animate: false, fit: true, padding: 55, nodeRepulsion: 8500, idealEdgeLength: 105 }), [])

  useEffect(() => {
    if (!job?.job_id || job.status === 'completed' || job.status === 'failed') return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/jobs/${job.job_id}`, { cache: 'no-store' })
        const body = await response.json()
        if (!response.ok) throw new Error(body.detail || 'Unable to check processing status')
        if (cancelled) return
        setJob(body)
        if (body.status === 'completed' && body.graph) {
          setGraph(body.graph)
          setHighlighted([])
          setSelectedNode(null)
          setError('')
        } else if (body.status === 'failed') {
          setError(body.error || 'Video processing failed. Check the backend terminal for details.')
        } else {
          timer = setTimeout(poll, 1800)
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Could not reach the ArgusGraph API')
          timer = setTimeout(poll, 3000)
        }
      }
    }
    timer = setTimeout(poll, 700)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [job?.job_id, job?.status])

  const uploadVideo = async () => {
    if (!file || uploading) return
    setUploading(true); setError(''); setGraph(null); setHighlighted([]); setSelectedNode(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const response = await fetch(`${API_BASE}/api/videos/upload`, { method: 'POST', body: form })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Video upload failed')
      setJob({ job_id: body.job_id, status: body.status, filename: body.filename, stage: 'Upload accepted by backend' })
      setUploadOpen(false); setFile(null)
    } catch (e) {
      setError(`${e instanceof Error ? e.message : 'Upload failed'}. Confirm FastAPI is running at ${API_BASE}.`)
    } finally { setUploading(false) }
  }

  const selectNode = useCallback((id: string) => {
    setSelectedNode(id)
    setHighlighted([id])
  }, [])

  const askAgent = async () => {
    const text = question.trim()
    if (!text || asking) return
    setQuestion('')
    setChatError('')
    setMessages(m => [...m, { role: 'user', text }])
    setAsking(true)
    try {
      const response = await fetch(`${API_BASE}/api/agent/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, session_id: sessionId }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Agent query failed')
      setMessages(m => [...m, { role: 'assistant', text: body.answer || 'The agent returned an empty answer.' }])
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not reach the ArgusGraph agent'
      setChatError(message)
      setMessages(m => [...m, { role: 'assistant', text: `I couldn't complete that investigation: ${message}` }])
    } finally {
      setAsking(false)
    }
  }

  const resetAgentChat = async () => {
    try {
      await fetch(`${API_BASE}/api/agent/reset`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      })
    } catch { /* Local chat can still be cleared if the API is unavailable. */ }
    setMessages([{ role: 'assistant', text: 'Chat context cleared. Ask a new question about the graph.' }])
    setChatError('')
  }

  const selected = graph?.nodes.find(n => n.data.id === selectedNode)?.data
  const selectedValue = selected ? (selected.value ?? JSON.stringify(selected, null, 2)) : ''
  const prettySelectedValue = (() => {
    if (!selectedValue) return 'No value stored on this node.'
    try { return JSON.stringify(JSON.parse(selectedValue), null, 2) } catch { return selectedValue }
  })()
  const zoomIn = () => { const cy = cyRef.current; if (cy) cy.zoom({ level: Math.min(cy.zoom() * 1.25, cy.maxZoom()), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }) }
  const zoomOut = () => { const cy = cyRef.current; if (cy) cy.zoom({ level: Math.max(cy.zoom() / 1.25, cy.minZoom()), renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }) }
  const fitGraph = () => { cyRef.current?.fit(undefined, 55); setHighlighted([]); setSelectedNode(null) }
  const copySelectedValue = async () => {
    try { await navigator.clipboard.writeText(prettySelectedValue); setCopiedNodeValue(true); window.setTimeout(() => setCopiedNodeValue(false), 1400) } catch { setCopiedNodeValue(false) }
  }

  return <main className="min-h-screen bg-[#080d15] text-slate-100">
    <div className="flex min-h-screen">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-slate-800 bg-[#0b111b] p-5 md:flex">
        <div className="mb-10 flex items-center gap-3"><div className="rounded-xl bg-violet-500/20 p-2 text-violet-300"><Network size={20}/></div><div><div className="text-lg font-semibold">Argus<span className="text-violet-300">Graph</span></div><div className="text-[9px] tracking-[.2em] text-slate-500">INTELLIGENCE PLATFORM</div></div></div>
        <div className="mb-3 text-[10px] font-semibold tracking-[.18em] text-slate-500">WORKSPACE</div>
        <div className="flex items-center gap-3 rounded-lg border border-violet-400/20 bg-violet-400/10 px-3 py-3 text-sm text-violet-200"><Network size={16}/> Graph Explorer</div>
        <button onClick={() => setUploadOpen(true)} className="mt-3 flex items-center gap-3 rounded-lg px-3 py-3 text-left text-sm text-slate-300 hover:bg-slate-800"><Upload size={16}/> Upload video</button>
        <div className="mt-auto rounded-xl border border-slate-800 bg-slate-900/70 p-3"><div className="text-xs font-medium">Local development</div><div className="mt-1 text-[11px] text-slate-500">FastAPI · FalkorDB</div><div className="mt-3 flex items-center gap-2 text-[11px]"><span className="h-2 w-2 rounded-full bg-emerald-400"/> UI connected to API endpoint</div></div>
      </aside>
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-5 py-4 md:px-7"><div><div className="text-lg font-semibold">{job?.summary?.video_name || graph?.video_id || 'Video graph workspace'}</div><div className="mt-1 flex items-center gap-2 text-xs text-slate-400"><span className={`h-2 w-2 rounded-full ${job?.status === 'processing' || job?.status === 'queued' ? 'bg-amber-400' : graph ? 'bg-emerald-400' : 'bg-slate-500'}`}/>{job ? `${job.status.toUpperCase()} · ${job.stage || ''}` : graph ? `Connected graph · ${graph.graph_name}` : 'Upload a video to create a graph in FalkorDB'}</div></div><button onClick={() => setUploadOpen(true)} className="flex items-center gap-2 rounded-lg bg-violet-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-400"><CloudUpload size={16}/> Upload video</button></header>
        {error && <div className="mx-5 mt-4 flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200 md:mx-7"><AlertTriangle size={17} className="mt-0.5 shrink-0"/><div className="min-w-0 flex-1">{error}{job?.stderr && <details className="mt-2"><summary className="cursor-pointer text-xs">Backend error output</summary><pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-xs">{job.stderr}</pre></details>}</div><button onClick={() => setError('')} aria-label="Dismiss error"><X size={15}/></button></div>}
        <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="flex min-h-[580px] flex-col border-b border-slate-800 xl:border-b-0 xl:border-r">
            <div className="flex items-center justify-between border-b border-slate-800/80 px-5 py-3"><div className="flex items-center gap-2 text-sm font-medium"><CircleDot size={14} className="text-cyan-300"/> Graph canvas</div><div className="text-xs text-slate-500">{graph ? `${graph.nodes.length} nodes · ${graph.edges.length} edges` : 'Waiting for graph data'}</div></div>
            <div className="relative min-h-[520px] flex-1 bg-[#0a101a]">
              <div className="pointer-events-none absolute inset-0 opacity-30" style={{backgroundImage:'radial-gradient(#475569 0.7px, transparent 0.7px)',backgroundSize:'22px 22px'}}/>
              {graph && graph.nodes.length > 0 ? <CytoscapeComponent elements={elements as any} stylesheet={stylesheet} minZoom={0.08} maxZoom={4} wheelSensitivity={0.18} cy={(cy: cytoscape.Core) => { cyRef.current = cy; cy.off('tap', 'node'); cy.on('tap', 'node', e => selectNode(e.target.id())); if (lastFittedGraphRef.current !== graph) { lastFittedGraphRef.current = graph; cy.fit(undefined, 55) } }} layout={graphLayout as any} style={{width:'100%',height:'100%',position:'absolute',inset:0}}/> : <div className="absolute inset-0 flex flex-col items-center justify-center px-8 text-center"><div className="mb-4 rounded-2xl border border-slate-700 bg-slate-900 p-4 text-slate-500"><Network size={30}/></div><div className="text-sm font-medium text-slate-300">{job?.status === 'processing' || job?.status === 'queued' ? 'Building your video graph' : 'No graph loaded yet'}</div><p className="mt-2 max-w-sm text-xs leading-5 text-slate-500">{job?.status === 'processing' || job?.status === 'queued' ? 'The existing pipeline is sampling frames, running visual models, and writing attributes to FalkorDB. The graph appears when ingestion and verification finish.' : 'Upload a CCTV video. The backend will run the existing pipeline and fetch the persisted graph from FalkorDB.'}</p>{(job?.status === 'processing' || job?.status === 'queued') && <div className="mt-5 flex items-center gap-2 text-xs text-cyan-200"><LoaderCircle className="animate-spin" size={15}/>{job.stage || 'Processing video…'}</div>}</div>}
              {graph && <div className="absolute bottom-4 left-4 rounded-lg border border-slate-700 bg-slate-950/90 p-3 text-[11px] text-slate-300"><div className="mb-2 font-semibold tracking-wider text-slate-500">NODE TYPES</div><div className="flex flex-col gap-2"><span className="flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-violet-400"/>Video</span><span className="flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-cyan-300"/>Frame</span><span className="flex items-center gap-2"><i className="h-3 w-3 rounded-full bg-slate-400"/>Frame attribute</span></div></div>}
              {graph && <div className="absolute right-4 top-4 flex flex-col gap-1 rounded-xl border border-slate-700 bg-slate-950/95 p-1.5 shadow-lg">
                <button onClick={zoomIn} className="rounded-md p-2 text-slate-200 hover:bg-slate-800" title="Zoom in" aria-label="Zoom in"><Plus size={16}/></button>
                <button onClick={zoomOut} className="rounded-md p-2 text-slate-200 hover:bg-slate-800" title="Zoom out" aria-label="Zoom out"><Minus size={16}/></button>
                <button onClick={fitGraph} className="rounded-md p-2 text-slate-200 hover:bg-slate-800" title="Fit graph" aria-label="Fit graph"><Maximize2 size={15}/></button>
              </div>}
            </div>
            {selected && <div className="border-t border-slate-800 bg-[#0b111b] px-4 py-4">
              <div className="flex items-start justify-between gap-3"><div className="min-w-0"><div className="text-[10px] font-semibold tracking-widest text-cyan-300">SELECTED NODE</div><div className="mt-1 break-words text-sm font-semibold text-slate-100">{selected.label}</div><div className="mt-1 text-xs text-slate-400">{selected.meta || selected.type}{selected.key ? ` · ${selected.key}` : ''}</div></div><button onClick={() => void copySelectedValue()} className="flex shrink-0 items-center gap-1 rounded-md border border-slate-700 px-2 py-1.5 text-[11px] text-slate-300 hover:bg-slate-800">{copiedNodeValue ? <Check size={13}/> : <Copy size={13}/>} {copiedNodeValue ? 'Copied' : 'Copy value'}</button></div>
              <div className="mt-3 text-[10px] text-slate-500">NODE ID · <span className="break-all text-slate-400">{selected.id}</span></div>
              <div className="mt-3 max-h-64 overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-3"><div className="mb-2 text-[10px] font-semibold tracking-wider text-slate-500">NODE VALUE / PROPERTIES</div><pre className="whitespace-pre-wrap break-words text-xs leading-5 text-emerald-100">{prettySelectedValue}</pre></div>
              {selected.frame_id && <div className="mt-2 text-[11px] text-slate-500">Frame: <span className="text-slate-300">{selected.frame_id}</span></div>}
              {selected.video_id && <div className="mt-1 text-[11px] text-slate-500">Video: <span className="break-all text-slate-300">{selected.video_id}</span></div>}
            </div>}
          </div>
          <aside className="flex min-h-[500px] flex-col bg-[#0b111b]"><div className="flex items-center gap-3 border-b border-slate-800 px-5 py-4"><div className="rounded-lg bg-violet-400/15 p-2 text-violet-200"><Bot size={17}/></div><div><div className="text-sm font-semibold">ArgusGraph Agent</div><div className="mt-0.5 text-[11px] text-slate-500">Gemini · MCP graph investigation</div></div></div>
            <div className="flex-1 space-y-4 overflow-y-auto p-5">{messages.map((m,i)=><div key={i} className="flex gap-3"><div className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${m.role==='assistant'?'bg-violet-400/15 text-violet-200':'bg-slate-700 text-slate-200'}`}>{m.role==='assistant'?<Activity size={14}/>:<span className="text-[10px]">YOU</span>}</div><div className="min-w-0 flex-1"><div className="mb-1 text-[10px] font-semibold tracking-widest text-slate-500">{m.role==='assistant'?'ARGUS':'YOU'}</div><div className="min-w-0 break-words">{m.role === 'assistant' ? <MarkdownAnswer text={m.text}/> : <div className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-200">{m.text}</div>}</div></div></div>)}
              {asking && <div className="flex items-center gap-2 text-xs text-violet-200"><LoaderCircle size={14} className="animate-spin"/>Investigating graph evidence…</div>}
              {chatError && <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-2 text-[11px] text-rose-200">{chatError}</div>}
              {graph && <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-3"><div className="mb-2 flex items-center gap-2 text-xs font-semibold text-slate-300"><CheckCircle2 size={14} className="text-emerald-300"/> Graph persisted and loaded</div><div className="grid grid-cols-3 gap-2 text-center"><div className="rounded-lg bg-slate-950 p-2"><div className="text-lg font-semibold">{graph.nodes.length}</div><div className="text-[10px] text-slate-500">Nodes</div></div><div className="rounded-lg bg-slate-950 p-2"><div className="text-lg font-semibold">{graph.edges.length}</div><div className="text-[10px] text-slate-500">Edges</div></div><div className="rounded-lg bg-slate-950 p-2"><div className="text-lg font-semibold">{job?.summary?.frames_processed ?? graph.nodes.filter(n=>n.data.type==='frame').length}</div><div className="text-[10px] text-slate-500">Frames</div></div></div><div className="mt-2 text-[10px] text-slate-500">Graph: {graph.graph_name} · Video ID: {graph.video_id}</div></div>}
              {job?.status === 'failed' && <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-200">Processing failed: {job.error}</div>}
            </div>
            <div className="border-t border-slate-800 p-4"><div className="mb-2 flex items-center justify-between text-[10px] text-slate-500"><span>Answers are grounded in registered MCP graph tools.</span><button onClick={resetAgentChat} className="hover:text-slate-300">Clear chat</button></div><div className="rounded-xl border border-slate-700 bg-slate-950 p-2"><textarea value={question} onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();void askAgent()}}} rows={2} placeholder="Ask about your video…" className="w-full resize-none bg-transparent px-2 py-1 text-xs outline-none placeholder:text-slate-600"/><div className="flex items-center justify-between px-1"><span className="text-[10px] text-slate-600">Enter to send</span><button disabled={asking||!question.trim()} onClick={()=>void askAgent()} className="rounded-lg bg-violet-500 p-2 text-white hover:bg-violet-400 disabled:opacity-50" aria-label="Send message">{asking?<LoaderCircle size={14} className="animate-spin"/>:<Send size={14}/>}</button></div></div></div>
          </aside>
        </div>
      </section>
    </div>
    {uploadOpen && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"><div className="w-full max-w-lg rounded-2xl border border-slate-700 bg-[#0d1522] p-6 shadow-2xl"><div className="mb-5 flex items-start justify-between"><div><h2 className="text-lg font-semibold">Upload CCTV video</h2><p className="mt-1 text-xs text-slate-400">The FastAPI backend runs your existing ArgusGraph pipeline.</p></div><button onClick={()=>{setUploadOpen(false);setFile(null)}} aria-label="Close upload" className="rounded-lg p-2 text-slate-400 hover:bg-slate-800"><X size={17}/></button></div><label className="flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-600 bg-slate-900/60 p-5 text-center hover:border-violet-400"><input type="file" accept="video/*,.mkv,.avi" className="sr-only" onChange={e=>setFile(e.target.files?.[0]||null)}/><FileVideo size={28} className="mb-3 text-violet-300"/><div className="text-sm font-medium">{file?.name || 'Choose a video file'}</div><div className="mt-2 text-xs text-slate-500">MP4, MOV, AVI, MKV, WEBM</div>{file && <div className="mt-3 text-xs text-emerald-300">Ready · {(file.size/1024/1024).toFixed(1)} MB</div>}</label><div className="mt-4 flex items-start gap-2 text-[11px] leading-5 text-slate-500"><AlertTriangle size={14} className="mt-0.5 shrink-0"/>Processing can take a while because YOLO, CLIP, BLIP, and Gemini run in the existing pipeline. Keep the API terminal open to see detailed logs.</div><div className="mt-6 flex justify-end gap-3"><button onClick={()=>{setUploadOpen(false);setFile(null)}} className="rounded-lg px-4 py-2 text-sm text-slate-300 hover:bg-slate-800">Cancel</button><button disabled={!file||uploading} onClick={uploadVideo} className="flex items-center gap-2 rounded-lg bg-violet-500 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50">{uploading?<LoaderCircle className="animate-spin" size={15}/>:<Upload size={15}/>} {uploading?'Uploading…':'Upload and build graph'}</button></div></div></div>}
  </main>
}