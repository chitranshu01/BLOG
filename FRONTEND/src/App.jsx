import React from "react";
import { useMemo, useState } from 'react';
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import {
  Document,
  Packer,
  Paragraph,
  HeadingLevel,
  TextRun,
} from "docx";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronDown,
  Clock3,
  Copy,
  FileDown,
  FlaskConical,
  Image as ImageIcon,
  Link2,
  Menu,
  PenLine,
  Search,
  Sparkles,
  Terminal,
  X,
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const GENERATE_ENDPOINT = import.meta.env.VITE_GENERATE_ENDPOINT || '/api/generate';
const MODE_META = {
  auto: {
    label: 'Auto',
    desc: 'Let the router decide whether research is needed.',
  },
  closed_book: {
    label: 'Closed book',
    desc: 'Best for evergreen concepts and stable technical topics.',
  },
  hybrid: {
    label: 'Hybrid',
    desc: 'Blend evergreen knowledge with current sources and examples.',
  },
  open_book: {
    label: 'Open book',
    desc: 'Research recent developments, news, tools, pricing, or policies.',
  },
};

const BLOG_TYPES = [
  { value: 'auto', label: 'Auto' },
  { value: 'explainer', label: 'Explainer' },
  { value: 'tutorial', label: 'Tutorial' },
  { value: 'comparison', label: 'Comparison' },
  { value: 'system_design', label: 'System design' },
  { value: 'news_roundup', label: 'News roundup' },
];

function normalizeExternalUrl(value) {
  const rawUrl = String(value || '').trim();
  if (!/^https?:\/\//i.test(rawUrl)) return '';

  try {
    const url = new URL(rawUrl);
    return url.href;
  } catch {
    return '';
  }
}

function App() {
  const [topic, setTopic] = useState('');
  const [audience, setAudience] = useState('Developers and technical readers');
  const [tone, setTone] = useState('Clear, practical, and authoritative');
  const [mode, setMode] = useState('auto');
  const [blogKind, setBlogKind] = useState('auto');
  const [asOf, setAsOf] = useState(new Date().toISOString().slice(0, 10));
  const [includeCode, setIncludeCode] = useState(true);
  const [includeImages, setIncludeImages] = useState(true);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [showPrototypeNotice, setShowPrototypeNotice] = useState(true);

  const selectedMode = useMemo(() => MODE_META[mode], [mode]);

  async function generateBlog() {
    const trimmedTopic = topic.trim();
    if (!trimmedTopic) {
      setError('Enter a topic before generating your article.');
      return;
    }

    setStatus('loading');
    setError('');
    setResult(null);

    const payload = {
      topic: trimmedTopic,
      audience,
      tone,
      as_of: asOf,
      mode: mode === 'auto' ? null : mode,
      blog_kind: blogKind === 'auto' ? null : blogKind,
      requires_code: includeCode,
      include_images: includeImages,
    };

    try {
      const response = await fetch(`${API_BASE_URL}${GENERATE_ENDPOINT}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.error || `Request failed with ${response.status}`);
      }

      const markdown = (
        data.final ||
        data.markdown ||
        data.content ||
        ''
      )
  // Convert escaped newlines to real newlines
        .replace(/\\+n/g, '\n')
        .replace(/\\+r/g, '')
  // Remove backslashes before Markdown characters
        .replace(/\\+(?=[#*])/g, '')
        .trim();
      if (!markdown) {
        throw new Error('The backend returned no generated Markdown.');
      }

      setResult({
        markdown,
        plan: data.plan || null,
        mode: data.mode || mode,
        evidence: data.evidence || [],
        imageSpecs: data.image_specs || [],
        durationMs: data.duration_ms,
      });
      setStatus('success');
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    }
  }

  async function copyMarkdown() {
    if (!result?.markdown) return;
    await navigator.clipboard.writeText(result.markdown);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  function downloadMarkdown() {
    if (!result?.markdown) return;
    const blob = new Blob([result.markdown], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated-blog.md';
    a.click();
    URL.revokeObjectURL(url);
  }
  async function downloadPDF() {
  const element = document.getElementById("article-content");

  if (!element) {
    setError("Article content not found.");
    return;
  }
  

  try {
    await Promise.all(
      Array.from(element.querySelectorAll("img")).map((image) => {
        image.crossOrigin = "anonymous";
        return image.complete
          ? image.decode
            ? image.decode().catch(() => {})
            : Promise.resolve()
          : new Promise((resolve) => {
              image.addEventListener("load", resolve, { once: true });
              image.addEventListener("error", resolve, { once: true });
            });
      })
    );

    const canvas = await html2canvas(element, {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff",
      onclone: (clonedDocument) => {
        const article = clonedDocument.getElementById("article-content");
        if (!article) return;
        article.style.background = "#ffffff";
        article.style.color = "#172033";
        article.style.border = "0";
        article.querySelectorAll("h1, h2, h3, strong").forEach((node) => {
          node.style.color = "#111827";
        });
        article.querySelectorAll("p, li, blockquote").forEach((node) => {
          node.style.color = "#273449";
        });
        article.querySelectorAll("a").forEach((node) => {
          node.style.color = "#174ea6";
        });
        article.querySelectorAll("pre, th").forEach((node) => {
          node.style.background = "#f3f4f6";
          node.style.color = "#172033";
          node.style.borderColor = "#d1d5db";
        });
        article.querySelectorAll("td").forEach((node) => {
          node.style.borderColor = "#d1d5db";
        });
        article.querySelectorAll("img").forEach((image) => {
          image.style.borderColor = "#d1d5db";
        });
      },
    });

    const imgData = canvas.toDataURL("image/png");

    const pdf = new jsPDF("p", "mm", "a4");

    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();

    const imgWidth = pageWidth;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    let heightLeft = imgHeight;
    let position = 0;

    pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);

    heightLeft -= pageHeight;

    while (heightLeft > 0) {
      position = heightLeft - imgHeight;

      pdf.addPage();
      pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);

      heightLeft -= pageHeight;
    }

    pdf.save("blogforge-article.pdf");
    setExportOpen(false);
  } catch (err) {
    setError("Failed to generate PDF.");
  }
}
  async function downloadDOCX() {
  const element = document.getElementById("article-content");

  if (!element) {
    setError("Article content not found.");
    return;
  }

  try {
    const children = [];

    element.querySelectorAll("h1, h2, h3, p, li, pre").forEach((node) => {
      const text = node.innerText.trim();

      if (!text) return;

      if (node.tagName === "H1") {
        children.push(
          new Paragraph({
            text,
            heading: HeadingLevel.TITLE,
          })
        );
      } else if (node.tagName === "H2") {
        children.push(
          new Paragraph({
            text,
            heading: HeadingLevel.HEADING_1,
          })
        );
      } else if (node.tagName === "H3") {
        children.push(
          new Paragraph({
            text,
            heading: HeadingLevel.HEADING_2,
          })
        );
      } else if (node.tagName === "LI") {
        children.push(
          new Paragraph({
            text,
            bullet: {
              level: 0,
            },
          })
        );
      } else if (node.tagName === "PRE") {
        children.push(
          new Paragraph({
            children: [
              new TextRun({
                text,
                font: "Courier New",
              }),
            ],
          })
        );
      } else {
        children.push(
          new Paragraph({
            text,
          })
        );
      }
    });

    const doc = new Document({
      sections: [
        {
          children,
        },
      ],
    });

    const blob = await Packer.toBlob(doc);

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "blogforge-article.docx";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
    setExportOpen(false);
  } catch (err) {
    setError("Failed to generate Word document.");
  }
}

  return (
    <div className="app-shell">
      {showPrototypeNotice && (
        <div className="prototype-overlay" role="presentation">
          <section className="prototype-notice" role="dialog" aria-modal="true" aria-labelledby="prototype-notice-title">
            <button
              className="prototype-close"
              onClick={() => setShowPrototypeNotice(false)}
              aria-label="Close prototype notice"
            >
              <X size={18} />
            </button>
            <div className="prototype-badge">Prototype</div>
            <h2 id="prototype-notice-title">This is a demo version.</h2>
            <p>
              OpenRouter powers blog creation, while AI Guru Lab generates the images.
              These free models may take 3–4 minutes, and results may vary.
            </p>
            <p>
              With paid, production-grade models, generation will be faster and the responses
              will be significantly better.
            </p>
            <button className="prototype-action" onClick={() => setShowPrototypeNotice(false)}>
              Continue to demo
            </button>
          </section>
        </div>
      )}
      <header className="topbar">
        <div className="brand-block">
          <button className="mobile-menu" onClick={() => setSidebarOpen((v) => !v)} aria-label="Toggle settings">
            <Menu size={19} />
          </button>
          <div className="brand-mark"><Sparkles size={17} /></div>
          <div>
            <div className="brand-name">BlogForge</div>
            <div className="brand-caption">AI technical writing studio</div>
          </div>
        </div>
        <div className="topbar-status"><span className="status-dot" /> API-ready workspace</div>
      </header>

      <div className="workspace">
        <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
          <div className="sidebar-head">
            <div>
              <div className="eyebrow">Workspace</div>
              <h2>Article setup</h2>
            </div>
            <button className="close-sidebar" onClick={() => setSidebarOpen(false)} aria-label="Close settings">
              <X size={18} />
            </button>
          </div>

          <section className="control-group">
            <label htmlFor="topic">Topic</label>
            <textarea
              id="topic"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. How LangGraph handles durable agent workflows"
              rows={5}
            />
          </section>

          <section className="control-group two-col">
            <div>
              <label htmlFor="audience">Audience</label>
              <input id="audience" value={audience} onChange={(e) => setAudience(e.target.value)} />
            </div>
            <div>
              <label htmlFor="date">As of</label>
              <input id="date" type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
            </div>
          </section>

          <section className="control-group">
            <label htmlFor="tone">Tone</label>
            <input id="tone" value={tone} onChange={(e) => setTone(e.target.value)} />
          </section>

          <section className="control-group">
            <div className="label-row">
              <label>Research mode</label>
              <span className="tiny-note"><FlaskConical size={13} /> router-aware</span>
            </div>
            <div className="option-grid">
              {Object.entries(MODE_META).map(([key, meta]) => (
                <button
                  key={key}
                  className={`option-card ${mode === key ? 'active' : ''}`}
                  onClick={() => setMode(key)}
                >
                  <span className="option-title">{meta.label}</span>
                  <span className="option-desc">{meta.desc}</span>
                </button>
              ))}
            </div>
            <div className="selected-hint">{selectedMode.desc}</div>
          </section>

          <section className="control-group">
            <label htmlFor="kind">Blog format</label>
            <div className="select-wrap">
              <select id="kind" value={blogKind} onChange={(e) => setBlogKind(e.target.value)}>
                {BLOG_TYPES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
              </select>
              <ChevronDown size={16} />
            </div>
          </section>

          <section className="control-group toggle-stack">
            <Toggle icon={<Terminal size={15} />} label="Include code examples" checked={includeCode} onChange={setIncludeCode} />
            <Toggle icon={<ImageIcon size={15} />} label="Plan technical images" checked={includeImages} onChange={setIncludeImages} />
          </section>

          <button className="generate-button" onClick={generateBlog} disabled={status === 'loading'}>
            {status === 'loading' ? <span className="spinner" /> : <PenLine size={18} />}
            {status === 'loading' ? 'Generating article…' : 'Generate article'}
            <ArrowUpRight size={17} />
          </button>

          {error && <div className="error-box">{error}</div>}

          <div className="sidebar-footer">
            <div className="api-note"><span className="api-dot" /> Backend endpoint: <code>{GENERATE_ENDPOINT}</code></div>
            <div className="api-note">Keep API keys server-side. The browser should call your backend only.</div>
          </div>
        </aside>

        <main className="main-panel">
          <div className="main-toolbar">
            <div>
              <div className="eyebrow">Studio</div>
              <h1>{result?.plan?.blog_title || 'Draft a publish-ready technical article'}</h1>
            </div>
            <div className="toolbar-actions">
              <button className="toolbar-button" onClick={copyMarkdown} disabled={!result}><Copy size={15} /> {copied ? 'Copied' : 'Copy Markdown'}</button>
              <div className="export-menu">
                <button
                  className="toolbar-button primary-outline"
                  onClick={() => setExportOpen((value) => !value)}
                  disabled={!result}
                >
                  <FileDown size={15} /> Export
                </button>

                {exportOpen && result && (
                  <div className="export-dropdown">
                    <button
                      onClick={() => {
                        downloadMarkdown();
                        setExportOpen(false);
                    }}
                  >
                    Markdown (.md)
                    </button>

                    <button onClick={downloadPDF}>
                      PDF (.pdf)
                    </button>

                    <button onClick={downloadDOCX}>
                      Word (.docx)
                     </button>
                   </div>
                   )}
              </div>
            </div>
          </div>

          {!result ? (
            <EmptyState topic={topic} mode={mode} />
          ) : (
            <ArticleResult result={result} />
          )}
        </main>
      </div>
    </div>
  );
}

function Toggle({ icon, label, checked, onChange }) {
  return (
    <button className="toggle-row" onClick={() => onChange(!checked)} aria-pressed={checked}>
      <span className="toggle-icon">{icon}</span>
      <span>{label}</span>
      <span className={`switch ${checked ? 'on' : ''}`}><span /></span>
    </button>
  );
}

function EmptyState({ topic, mode }) {
  return (
    <div className="empty-state">
      <div className="empty-visual"><BookOpen size={24} /></div>
      <div className="empty-copy">
        <div className="eyebrow">Ready when you are</div>
        <h2>{topic ? 'Your article workspace is configured.' : 'Turn a topic into a technical story.'}</h2>
        <p>
          {topic
            ? `The ${mode === 'auto' ? 'router will choose the right research path' : `${MODE_META[mode].label.toLowerCase()} workflow is selected`} and the backend can orchestrate the article.`
            : 'Add a topic, tune the audience and tone, then generate a clean Markdown article with research and image planning when appropriate.'}
        </p>
        <div className="feature-row">
          <Feature icon={<Search size={15} />} text="Research-aware" />
          <Feature icon={<Link2 size={15} />} text="Evidence links" />
          <Feature icon={<ImageIcon size={15} />} text="Technical visuals" />
          <Feature icon={<Terminal size={15} />} text="Code snippets" />
        </div>
      </div>
    </div>
  );
}

function Feature({ icon, text }) {
  return <div className="feature-chip">{icon}{text}</div>;
}

function ArticleResult({ result }) {
  const title = result.plan?.blog_title || 'Generated article';
  const audience = result.plan?.audience;
  const tone = result.plan?.tone;

  return (
    <div className="result-layout">
      <article className="article-card">
        <div className="article-meta">
          <span className="meta-pill success"><Check size={13} /> Generated</span>
          <span className="meta-pill"><Clock3 size={13} /> {result.durationMs ? `${(result.durationMs / 1000).toFixed(1)}s` : 'Pipeline complete'}</span>
          <span className="meta-pill"><Search size={13} /> {MODE_META[result.mode]?.label || result.mode}</span>
        </div>

        <div className="article-header">
          <div className="article-kicker">{result.plan?.blog_kind?.replace('_', ' ') || 'Technical article'}</div>
          <h2>{title}</h2>
          <div className="article-submeta">
            {audience && <span>Audience: {audience}</span>}
            {tone && <span>· {tone}</span>}
          </div>
        </div>

        <div className="markdown-body" id="article-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children, ...props }) => {
                const externalHref = normalizeExternalUrl(href);
                if (!externalHref) return <span {...props}>{children}</span>;

                return (
                  <a href={externalHref} target="_blank" rel="noreferrer" {...props}>
                    {children}<ArrowUpRight size={12} />
                  </a>
                );
              },
              img: ({ alt, src }) => {
                if (!src) {
                 return null;
                }

                return (
                  <span className="article-image">
                    <img
                      src={src}
                      alt={alt || ''}
                      loading="lazy"
                   />
                   {alt && (
                     <span className="article-image-caption">
                       {alt}
                     </span>
                    )}
                  </span>
                );
              },
              pre: ({ children }) => <pre className="code-block">{children}</pre>,
            }}
          >
            {result.markdown}
          </ReactMarkdown>
        </div>
      </article>

      <aside className="insights-card">
        <div className="insights-head">
          <div>
            <div className="eyebrow">Pipeline</div>
            <h3>Generation details</h3>
          </div>
          <Sparkles size={16} />
        </div>

        <Insight label="Research sources" value={String(result.evidence?.length || 0)} />
        <Insight label="Planned visuals" value={String(result.imageSpecs?.length || 0)} />
        <Insight label="Sections" value={String(result.plan?.tasks?.length || '—')} />

        {!!result.evidence?.length && (
          <div className="sources-block">
            <div className="section-title">Evidence</div>
            {result.evidence.slice(0, 6).map((item) => (
              (() => {
                const sourceUrl = normalizeExternalUrl(item.url);
                if (!sourceUrl) return null;

                return (
                  <a key={sourceUrl} href={sourceUrl} target="_blank" rel="noreferrer" className="source-item">
                    <span className="source-icon"><Link2 size={13} /></span>
                    <span className="source-text">
                      <strong>{item.title || sourceUrl}</strong>
                      <small>{item.source || sourceUrl}</small>
                    </span>
                    <ArrowUpRight size={13} />
                  </a>
                );
              })()
            ))}
          </div>
        )}

        {!!result.imageSpecs?.length && (
          <div className="sources-block">
            <div className="section-title">Image plan</div>
            {result.imageSpecs.map((item) => (
              <div key={item.placeholder} className="image-plan-item">
                <ImageIcon size={14} />
                <div><strong>{item.alt}</strong><small>{item.caption}</small></div>
              </div>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}

function Insight({ label, value }) {
  return <div className="insight-row"><span>{label}</span><strong>{value}</strong></div>;
}

export default App;
