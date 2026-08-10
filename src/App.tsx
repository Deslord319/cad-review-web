"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

type ModelInfo = {
  name: string;
  extension: string;
  size: number;
  modified: string;
  viewable: boolean;
  facets?: number;
  dimensions?: [number, number, number];
  volume?: number;
  watertight?: boolean;
  scope: ModelScope;
};

type ModelScope = "active" | "archive" | "trash";
type ModelAction = "archive" | "trash" | "restore";
type ModelCounts = Record<ModelScope, number>;

type ViewName = "iso" | "front" | "back" | "top" | "right";

type ViewerState = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  model: THREE.Mesh | null;
  frame: number;
  radius: number;
};

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export default function Home() {
  const canvasHost = useRef<HTMLDivElement>(null);
  const viewer = useRef<ViewerState | null>(null);
  const [apiBase, setApiBase] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [scope, setScope] = useState<ModelScope>("active");
  const [counts, setCounts] = useState<ModelCounts>({ active: 0, archive: 0, trash: 0 });
  const [selectedName, setSelectedName] = useState("");
  const [loading, setLoading] = useState(true);
  const [modelLoading, setModelLoading] = useState(false);
  const [error, setError] = useState("");
  const [wireframe, setWireframe] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState("");

  const selected = useMemo(
    () => models.find((model) => model.name === selectedName) ?? null,
    [models, selectedName],
  );

  const refreshModels = useCallback(async (base?: string) => {
    const origin = base || apiBase;
    if (!origin) return;
    try {
      const response = await fetch(`${origin}/api/models?scope=${scope}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as { models: ModelInfo[]; counts: ModelCounts };
      setModels(data.models);
      setCounts(data.counts);
      setSelectedName((current) => {
        if (current && data.models.some((item) => item.name === current)) return current;
        return data.models.find((item) => item.viewable)?.name ?? data.models[0]?.name ?? "";
      });
      setUpdatedAt(new Date());
      setError("");
    } catch {
      setError("无法连接模型服务，请检查后台服务状态。");
    } finally {
      setLoading(false);
    }
  }, [apiBase, scope]);

  useEffect(() => {
    const configuredBase = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "");
    const base = configuredBase || `${window.location.protocol}//${window.location.hostname}:8091`;
    const frame = window.requestAnimationFrame(() => {
      setApiBase(base);
      void refreshModels(base);
    });
    const timer = window.setInterval(() => void refreshModels(base), 15000);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearInterval(timer);
    };
  }, [refreshModels]);

  const modelUrl = useCallback((model: ModelInfo, download = false) => {
    const suffix = download ? "?download=1" : "";
    return `${apiBase}/models/${model.scope}/${encodeURIComponent(model.name)}${suffix}`;
  }, [apiBase]);

  const runModelAction = useCallback(async (action: ModelAction) => {
    if (!selected || actionLoading) return;
    const prompts: Record<ModelAction, string> = {
      archive: `将“${selected.name}”移入归档？`,
      trash: `将“${selected.name}”移入回收站？此操作可以恢复。`,
      restore: `将“${selected.name}”恢复到活跃模型？`,
    };
    if (!window.confirm(prompts[action])) return;

    setActionLoading(true);
    setActionMessage("");
    try {
      const response = await fetch(`${apiBase}/api/models/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, scope: selected.scope, name: selected.name }),
      });
      const result = (await response.json()) as { message?: string };
      if (!response.ok) throw new Error(result.message || `HTTP ${response.status}`);
      const labels: Record<ModelAction, string> = {
        archive: "已归档",
        trash: "已移入回收站",
        restore: "已恢复到活跃模型",
      };
      setActionMessage(labels[action]);
      await refreshModels();
    } catch (actionError) {
      setActionMessage(actionError instanceof Error ? actionError.message : "操作失败");
    } finally {
      setActionLoading(false);
    }
  }, [actionLoading, apiBase, refreshModels, selected]);

  useEffect(() => {
    const host = canvasHost.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101318);
    scene.fog = new THREE.Fog(0x101318, 320, 800);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 2000);
    camera.position.set(120, -150, 105);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.rotateSpeed = 0.65;
    controls.panSpeed = 0.6;

    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(90, -120, 160);
    key.castShadow = true;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x88aaff, 1.6);
    fill.position.set(-140, 60, 80);
    scene.add(fill);
    scene.add(new THREE.HemisphereLight(0xe8f0ff, 0x222832, 1.8));

    const grid = new THREE.GridHelper(360, 36, 0x384150, 0x232932);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -45;
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    gridMaterials.forEach((material) => {
      material.transparent = true;
      material.opacity = 0.6;
    });
    scene.add(grid);

    const state: ViewerState = { scene, camera, renderer, controls, model: null, frame: 0, radius: 80 };
    viewer.current = state;

    const resize = () => {
      const { width, height } = host.getBoundingClientRect();
      renderer.setSize(Math.max(width, 1), Math.max(height, 1), false);
      camera.aspect = Math.max(width, 1) / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const render = () => {
      state.frame = requestAnimationFrame(render);
      controls.update();
      renderer.render(scene, camera);
    };
    render();

    return () => {
      observer.disconnect();
      cancelAnimationFrame(state.frame);
      controls.dispose();
      if (state.model) {
        state.model.geometry.dispose();
        (state.model.material as THREE.Material).dispose();
      }
      renderer.dispose();
      renderer.domElement.remove();
      viewer.current = null;
    };
  }, []);

  const fitView = useCallback((viewName: ViewName = "iso") => {
    const state = viewer.current;
    if (!state) return;
    const distance = Math.max(state.radius * 2.7, 60);
    const positions: Record<ViewName, [number, number, number]> = {
      iso: [distance, -distance, distance * 0.72],
      front: [0, 0, distance],
      back: [0, 0, -distance],
      top: [0, -distance, 0],
      right: [distance, 0, 0],
    };
    state.camera.position.set(...positions[viewName]);
    state.camera.up.set(0, 1, 0);
    state.controls.target.set(0, 0, 0);
    state.controls.update();
  }, []);

  useEffect(() => {
    const state = viewer.current;
    if (!state || !selected?.viewable || !apiBase) return;
    setModelLoading(true);
    const loader = new STLLoader();
    loader.load(
      modelUrl(selected),
      (geometry) => {
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        if (box) {
          const center = new THREE.Vector3();
          const size = new THREE.Vector3();
          box.getCenter(center);
          box.getSize(size);
          geometry.translate(-center.x, -center.y, -center.z);
          state.radius = Math.max(size.length() / 2, 10);
        }
        if (state.model) {
          state.scene.remove(state.model);
          state.model.geometry.dispose();
          (state.model.material as THREE.Material).dispose();
        }
        const material = new THREE.MeshStandardMaterial({
          color: 0xb7c7dc,
          roughness: 0.44,
          metalness: 0.12,
          wireframe,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        state.model = mesh;
        state.scene.add(mesh);
        fitView("iso");
        setModelLoading(false);
      },
      undefined,
      () => {
        setError("模型加载失败，文件可能尚未导出完成。");
        setModelLoading(false);
      },
    );
  }, [apiBase, fitView, modelUrl, selected, wireframe]);

  useEffect(() => {
    const state = viewer.current;
    if (!state) return;
    state.controls.autoRotate = autoRotate;
    state.controls.autoRotateSpeed = 1.2;
  }, [autoRotate]);

  const viewableModels = models.filter((model) => model.viewable);
  const supportingFiles = models.filter((model) => !model.viewable);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark">SC</span>
          <div>
            <strong>Spark CAD Review</strong>
            <span>局域网成品审视器</span>
          </div>
        </div>
        <div className="connection-status">
          <span className={error ? "status-dot error" : "status-dot"} />
          <span>{error ? "服务异常" : "服务在线"}</span>
          {updatedAt && <time>{updatedAt.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}</time>}
          <button type="button" className="quiet-button" onClick={() => void refreshModels()} aria-label="刷新模型列表">
            刷新
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="model-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">OUTPUT LIBRARY</span>
              <h1>模型版本</h1>
            </div>
            <span className="count-pill">{viewableModels.length}</span>
          </div>
          <div className="scope-tabs" role="tablist" aria-label="模型状态">
            {([
              ["active", "活跃"],
              ["archive", "归档"],
              ["trash", "回收站"],
            ] as const).map(([value, label]) => (
              <button
                type="button"
                role="tab"
                aria-selected={scope === value}
                className={scope === value ? "active" : ""}
                key={value}
                onClick={() => {
                  setScope(value);
                  setActionMessage("");
                }}
              >
                {label}<span>{counts[value]}</span>
              </button>
            ))}
          </div>
          <div className="model-list">
            {loading && <p className="empty-message">正在读取 Spark 输出目录…</p>}
            {!loading && viewableModels.length === 0 && (
              <p className="empty-message">
                {scope === "active" ? "尚未发现可审视的 STL。" : scope === "archive" ? "归档区为空。" : "回收站为空。"}
              </p>
            )}
            {viewableModels.map((model) => (
              <button
                type="button"
                key={model.name}
                className={`model-row ${model.name === selectedName ? "active" : ""}`}
                onClick={() => setSelectedName(model.name)}
              >
                <span className="file-badge">STL</span>
                <span className="file-copy">
                  <strong>{model.name.replace(/\.stl$/i, "")}</strong>
                  <span>{formatDate(model.modified)} · {formatBytes(model.size)}</span>
                </span>
                <span className="row-arrow">›</span>
              </button>
            ))}
          </div>
          <div className="support-files">
            <span className="eyebrow">{scope === "active" ? "SOURCE FILES" : scope === "archive" ? "ARCHIVED FILES" : "TRASHED FILES"}</span>
            {supportingFiles.slice(0, 8).map((file) => (
              <a key={file.name} href={modelUrl(file, true)}>
                <span>{file.extension.toUpperCase()}</span>
                <strong>{file.name}</strong>
              </a>
            ))}
          </div>
        </aside>

        <section className="viewer-panel" aria-label="三维模型视口">
          <div ref={canvasHost} className="canvas-host" />
          <div className="viewer-overlay top-left">
            <span className="eyebrow">ACTIVE MODEL</span>
            <strong>{selected?.name.replace(/\.stl$/i, "") || "等待模型"}</strong>
          </div>
          {(modelLoading || error) && (
            <div className={`viewer-message ${error ? "has-error" : ""}`}>
              {error || "正在构建三维视图…"}
            </div>
          )}
          <div className="view-toolbar" role="toolbar" aria-label="视角工具">
            <button type="button" onClick={() => fitView("iso")}>等轴</button>
            <button type="button" onClick={() => fitView("front")}>正面</button>
            <button type="button" onClick={() => fitView("back")}>背面</button>
            <button type="button" onClick={() => fitView("top")}>顶部</button>
            <button type="button" onClick={() => fitView("right")}>右侧</button>
            <span className="toolbar-divider" />
            <button type="button" className={wireframe ? "pressed" : ""} onClick={() => setWireframe((value) => !value)}>线框</button>
            <button type="button" className={autoRotate ? "pressed" : ""} onClick={() => setAutoRotate((value) => !value)}>旋转</button>
          </div>
        </section>

        <aside className="inspector-panel">
          <div className="inspector-section">
            <span className="eyebrow">GEOMETRY</span>
            <h2>模型检查</h2>
            <div className="quality-line">
              <span className={`quality-icon ${selected?.watertight === false ? "bad" : ""}`}>
                {selected?.watertight === false ? "!" : "✓"}
              </span>
              <div>
                <strong>{selected?.watertight === false ? "检测到开放边" : "网格边闭合"}</strong>
                <span>{selected?.watertight === false ? "建议修复后再切片" : "满足基础水密检查"}</span>
              </div>
            </div>
          </div>

          <div className="inspector-section">
            <span className="section-label">包络尺寸</span>
            <div className="dimension-grid">
              {(["X", "Y", "Z"] as const).map((axis, index) => (
                <div key={axis}>
                  <span>{axis}</span>
                  <strong>{selected?.dimensions?.[index]?.toFixed(2) ?? "—"}</strong>
                  <small>mm</small>
                </div>
              ))}
            </div>
          </div>

          <div className="inspector-section metric-list">
            <div><span>三角面</span><strong>{selected?.facets?.toLocaleString() ?? "—"}</strong></div>
            <div><span>估算体积</span><strong>{selected?.volume ? `${selected.volume.toFixed(0)} mm³` : "—"}</strong></div>
            <div><span>文件大小</span><strong>{selected ? formatBytes(selected.size) : "—"}</strong></div>
            <div><span>更新时间</span><strong>{selected ? formatDate(selected.modified) : "—"}</strong></div>
          </div>

          <div className="inspector-actions">
            {selected && (
              <a className="primary-action" href={modelUrl(selected, true)}>
                下载当前 STL
              </a>
            )}
            {selected && scope === "active" && (
              <button type="button" className="secondary-action" disabled={actionLoading} onClick={() => void runModelAction("archive")}>
                归档当前模型
              </button>
            )}
            {selected && scope !== "active" && (
              <button type="button" className="secondary-action" disabled={actionLoading} onClick={() => void runModelAction("restore")}>
                恢复到活跃模型
              </button>
            )}
            {selected && scope !== "trash" && (
              <button type="button" className="danger-action" disabled={actionLoading} onClick={() => void runModelAction("trash")}>
                移入回收站
              </button>
            )}
            {actionMessage && <div className="action-message" role="status">{actionMessage}</div>}
            <p>左键旋转 · 右键平移 · 滚轮缩放</p>
          </div>
        </aside>
      </section>
    </main>
  );
}
