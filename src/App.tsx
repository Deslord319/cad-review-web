"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

type PreviewStatus = "not_applicable" | "pending" | "processing" | "ready" | "failed";
type PreviewCounts = Record<PreviewStatus, number>;

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
  preview_status: PreviewStatus;
  preview_url?: string;
  preview_revision?: string;
  preview_error?: string;
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
  model: THREE.Object3D | null;
  modelKey: string;
  frame: number;
  radius: number;
};

type ViewDepth = {
  near: number;
  far: number;
  fogNear: number;
  fogFar: number;
};

const EMPTY_PREVIEW_COUNTS: PreviewCounts = {
  not_applicable: 0,
  pending: 0,
  processing: 0,
  ready: 0,
  failed: 0,
};

const PREVIEW_STATUSES = new Set<PreviewStatus>([
  "not_applicable",
  "pending",
  "processing",
  "ready",
  "failed",
]);

const SPARSE_POINT_FACE_RATIO = 1.5;
const SPARSE_POINT_MIN_FACES = 5_000;
const SPARSE_POINT_SIZE = 2;

function normalizeModel(model: ModelInfo): ModelInfo {
  const status = PREVIEW_STATUSES.has(model.preview_status)
    ? model.preview_status
    : model.extension.toLowerCase() === "3mf"
      ? "pending"
      : "not_applicable";
  return { ...model, preview_status: status };
}

function countPreviewStates(models: ModelInfo[]): PreviewCounts {
  return models.reduce<PreviewCounts>((result, model) => {
    result[model.preview_status] += 1;
    return result;
  }, { ...EMPTY_PREVIEW_COUNTS });
}

function previewStatusLabel(status: PreviewStatus) {
  return {
    not_applicable: "无需预览",
    pending: "等待生成",
    processing: "生成中",
    ready: "已就绪",
    failed: "生成失败",
  }[status];
}

function resolveApiUrl(value: string, apiBase: string) {
  try {
    return new URL(value, `${apiBase.replace(/\/$/, "")}/`).toString();
  } catch {
    return "";
  }
}

function disposeModel(model: THREE.Object3D) {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  model.traverse((child) => {
    const renderable = child as THREE.Object3D & {
      geometry?: THREE.BufferGeometry;
      material?: THREE.Material | THREE.Material[];
    };
    if (renderable.geometry instanceof THREE.BufferGeometry) {
      geometries.add(renderable.geometry);
    }
    if (renderable.material) {
      const childMaterials = Array.isArray(renderable.material)
        ? renderable.material
        : [renderable.material];
      childMaterials.forEach((material) => materials.add(material));
    }
  });
  materials.forEach((material) => material.dispose());
  geometries.forEach((geometry) => geometry.dispose());
}

function addSparsePreviewPointOverlays(model: THREE.Object3D) {
  const meshes: THREE.Mesh[] = [];
  model.traverse((child) => {
    if (child instanceof THREE.Mesh) meshes.push(child);
  });

  let material: THREE.PointsMaterial | null = null;
  let overlays = 0;
  let vertices = 0;
  meshes.forEach((mesh) => {
    const position = mesh.geometry.getAttribute("position");
    const faceCount = Math.floor(
      (mesh.geometry.index?.count ?? position?.count ?? 0) / 3,
    );
    if (!position || faceCount < SPARSE_POINT_MIN_FACES) return;
    if (position.count / faceCount <= SPARSE_POINT_FACE_RATIO) return;

    material ??= new THREE.PointsMaterial({
      color: 0xb9d6ff,
      size: SPARSE_POINT_SIZE,
      sizeAttenuation: false,
      fog: false,
    });
    const points = new THREE.Points(mesh.geometry, material);
    points.name = `${mesh.name || "mesh"}__cad_preview_points`;
    points.userData.cadPreviewPointOverlay = true;
    points.frustumCulled = mesh.frustumCulled;
    mesh.add(points);
    overlays += 1;
    vertices += position.count;
  });
  return { overlays, vertices };
}

function setModelWireframe(model: THREE.Object3D, wireframe: boolean) {
  model.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => {
      if ("wireframe" in material) {
        (material as THREE.MeshBasicMaterial).wireframe = wireframe;
        material.needsUpdate = true;
      }
    });
  });
}

// Keep the complete bounding sphere inside the frustum while retaining a
// compact far/near ratio. A fixed far plane breaks as soon as a large model's
// fitted camera is more than that distance from the origin.
function calculateViewDepth(radius: number, cameraDistance: number): ViewDepth {
  const safeRadius = Number.isFinite(radius) ? Math.max(radius, 1) : 1;
  const safeDistance = Number.isFinite(cameraDistance)
    ? Math.max(cameraDistance, safeRadius)
    : safeRadius;
  const near = Math.max(0.01, safeDistance - safeRadius * 1.5);
  const far = Math.max(near + safeRadius * 3, safeDistance + safeRadius * 1.5);
  return {
    near,
    far,
    fogNear: safeDistance + safeRadius * 1.5,
    fogFar: safeDistance + safeRadius * 6,
  };
}

function updateViewDepth(state: ViewerState) {
  if (!state.model) return;
  const cameraDistance = state.camera.position.length();
  const depth = calculateViewDepth(state.radius, cameraDistance);
  const cameraChanged = Math.abs(state.camera.near - depth.near) > 0.001
    || Math.abs(state.camera.far - depth.far) > 0.001;
  if (cameraChanged) {
    state.camera.near = depth.near;
    state.camera.far = depth.far;
    state.camera.updateProjectionMatrix();
    state.renderer.domElement.dataset.cameraNear = depth.near.toFixed(3);
    state.renderer.domElement.dataset.cameraFar = depth.far.toFixed(3);
    state.renderer.domElement.dataset.cameraDistance = cameraDistance.toFixed(3);
    state.renderer.domElement.dataset.modelRadius = state.radius.toFixed(3);
  }
  if (state.scene.fog instanceof THREE.Fog) {
    state.scene.fog.near = depth.fogNear;
    state.scene.fog.far = depth.fogFar;
  }
}

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
  const loadGeneration = useRef(0);
  const wireframeRef = useRef(false);
  const [apiBase, setApiBase] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [scope, setScope] = useState<ModelScope>("active");
  const [counts, setCounts] = useState<ModelCounts>({ active: 0, archive: 0, trash: 0 });
  const [previewCounts, setPreviewCounts] = useState<PreviewCounts>(EMPTY_PREVIEW_COUNTS);
  const [selectedName, setSelectedName] = useState("");
  const [loading, setLoading] = useState(true);
  const [modelLoading, setModelLoading] = useState(false);
  const [loadedModelKey, setLoadedModelKey] = useState("");
  const [serviceError, setServiceError] = useState("");
  const [modelError, setModelError] = useState("");
  const [wireframe, setWireframe] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const [previewRetrying, setPreviewRetrying] = useState(false);
  const [urlReady, setUrlReady] = useState(false);

  const selected = useMemo(
    () => models.find((model) => model.name === selectedName) ?? null,
    [models, selectedName],
  );

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const parameters = new URLSearchParams(window.location.search);
      const requestedScope = parameters.get("scope");
      if (requestedScope === "archive" || requestedScope === "trash") setScope(requestedScope);
      setSelectedName(parameters.get("file") || "");
      setUrlReady(true);
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  const refreshModels = useCallback(async (base?: string) => {
    const origin = base || apiBase;
    if (!origin) return;
    try {
      const response = await fetch(`${origin}/api/models?scope=${scope}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as {
        models: ModelInfo[];
        counts: ModelCounts;
        preview_counts?: Partial<PreviewCounts>;
      };
      const nextModels = data.models.map(normalizeModel);
      setModels(nextModels);
      setCounts(data.counts);
      setPreviewCounts(data.preview_counts
        ? { ...EMPTY_PREVIEW_COUNTS, ...data.preview_counts }
        : countPreviewStates(nextModels));
      setSelectedName((current) => {
        if (current && nextModels.some((item) => item.name === current)) return current;
        return nextModels.find((item) => item.viewable)?.name ?? nextModels[0]?.name ?? "";
      });
      setUpdatedAt(new Date());
      setServiceError("");
    } catch {
      setServiceError("无法连接模型服务，请检查后台服务状态。");
    } finally {
      setLoading(false);
    }
  }, [apiBase, scope]);

  useEffect(() => {
    if (!urlReady) return;
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
  }, [refreshModels, urlReady]);

  const modelUrl = useCallback((model: ModelInfo, download = false) => {
    const suffix = download ? "?download=1" : "";
    return `${apiBase}/models/${model.scope}/${encodeURIComponent(model.name)}${suffix}`;
  }, [apiBase]);

  const selectedScope = selected?.scope ?? "";
  const selectedExtension = selected?.extension.toLowerCase() ?? "";
  const selectedPreviewStatus = selected?.preview_status ?? "not_applicable";
  const selectedPreviewUrl = selected?.preview_url ?? "";
  const selectedPreviewRevision = selected?.preview_revision ?? "";
  const selectedModified = selected?.modified ?? "";
  const selectedViewable = selected?.viewable ?? false;
  const selectedItemKey = selected ? `${selectedScope}\u0000${selected.name}` : "";
  const selectedAssetUrl = useMemo(() => {
    if (!selected || !apiBase || !selectedViewable) return "";
    if (selectedExtension === "3mf") {
      if (selectedPreviewStatus !== "ready" || !selectedPreviewUrl) return "";
      const resolved = resolveApiUrl(selectedPreviewUrl, apiBase);
      if (!resolved) return "";
      const url = new URL(resolved);
      if (selectedPreviewRevision && !url.searchParams.has("v")) {
        url.searchParams.set("v", selectedPreviewRevision);
      }
      return url.toString();
    }
    return `${apiBase}/models/${selectedScope}/${encodeURIComponent(selected.name)}`;
  }, [
    apiBase,
    selected,
    selectedExtension,
    selectedPreviewRevision,
    selectedPreviewStatus,
    selectedPreviewUrl,
    selectedScope,
    selectedViewable,
  ]);
  const loadKey = selectedAssetUrl
    ? `${selectedItemKey}\u0000${selectedExtension}\u0000${selectedExtension === "3mf" ? selectedPreviewRevision : selectedModified}`
    : "";

  useEffect(() => {
    if (
      selectedExtension !== "3mf"
      || !selectedItemKey
      || (selectedPreviewStatus !== "pending" && selectedPreviewStatus !== "processing")
    ) return;
    const timer = window.setInterval(() => void refreshModels(), 2000);
    return () => window.clearInterval(timer);
  }, [refreshModels, selectedExtension, selectedItemKey, selectedPreviewStatus]);

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

  const retryPreview = useCallback(async () => {
    if (!selected || selected.extension.toLowerCase() !== "3mf" || previewRetrying) return;
    setPreviewRetrying(true);
    setActionMessage("");
    setModelError("");
    try {
      const response = await fetch(`${apiBase}/api/previews/retry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scope: selected.scope, name: selected.name }),
      });
      const result = (await response.json().catch(() => ({}))) as { message?: string };
      if (!response.ok) throw new Error(result.message || `HTTP ${response.status}`);
      setActionMessage("已重新加入预览队列");
      await refreshModels();
    } catch (retryError) {
      setModelError(retryError instanceof Error ? retryError.message : "无法重新生成预览");
    } finally {
      setPreviewRetrying(false);
    }
  }, [apiBase, previewRetrying, refreshModels, selected]);

  useEffect(() => {
    const host = canvasHost.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101318);
    scene.fog = new THREE.Fog(0x101318, 320, 800);

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 2000);
    camera.position.set(120, -150, 105);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
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

    const state: ViewerState = { scene, camera, renderer, controls, model: null, modelKey: "", frame: 0, radius: 80 };
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
      updateViewDepth(state);
      renderer.render(scene, camera);
    };
    render();

    return () => {
      observer.disconnect();
      cancelAnimationFrame(state.frame);
      controls.dispose();
      if (state.model) {
        disposeModel(state.model);
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
    updateViewDepth(state);
  }, []);

  useEffect(() => {
    const state = viewer.current;
    if (!state) return;

    const generation = ++loadGeneration.current;
    const controller = new AbortController();
    const clearFrame = window.requestAnimationFrame(() => {
      setModelLoading(false);
      setLoadedModelKey("");
      setModelError("");
    });

    if (state.model) {
      state.scene.remove(state.model);
      disposeModel(state.model);
      state.model = null;
      state.modelKey = "";
    }
    state.renderer.domElement.dataset.previewPointOverlays = "0";
    state.renderer.domElement.dataset.previewPointVertices = "0";

    if (!selectedItemKey || !selectedViewable || !apiBase) {
      return () => {
        window.cancelAnimationFrame(clearFrame);
        controller.abort();
        if (loadGeneration.current === generation) loadGeneration.current += 1;
      };
    }

    if (selectedExtension === "3mf" && selectedPreviewStatus !== "ready") {
      return () => {
        window.cancelAnimationFrame(clearFrame);
        controller.abort();
        if (loadGeneration.current === generation) loadGeneration.current += 1;
      };
    }

    if (!selectedAssetUrl || !loadKey) {
      const errorFrame = window.requestAnimationFrame(() => {
        setModelError(selectedExtension === "3mf" ? "快速预览地址缺失，请重新生成预览。" : "模型地址无效。");
      });
      return () => {
        window.cancelAnimationFrame(clearFrame);
        window.cancelAnimationFrame(errorFrame);
        controller.abort();
        if (loadGeneration.current === generation) loadGeneration.current += 1;
      };
    }

    const loadingFrame = window.requestAnimationFrame(() => {
      setModelLoading(true);
      setLoadedModelKey("");
      setModelError("");
    });

    const isCurrent = () => loadGeneration.current === generation && !controller.signal.aborted;
    const showModel = (model: THREE.Object3D) => {
      if (!isCurrent()) {
        disposeModel(model);
        return;
      }
      window.cancelAnimationFrame(clearFrame);
      window.cancelAnimationFrame(loadingFrame);
      const pointOverlay = selectedExtension === "3mf"
        ? addSparsePreviewPointOverlays(model)
        : { overlays: 0, vertices: 0 };
      state.renderer.domElement.dataset.previewPointOverlays = String(pointOverlay.overlays);
      state.renderer.domElement.dataset.previewPointVertices = String(pointOverlay.vertices);
      model.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(model);
      if (!box.isEmpty()) {
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        model.position.sub(center);
        state.radius = Math.max(size.length() / 2, 10);
      }
      const enableShadows = selectedExtension !== "3mf";
      model.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        child.castShadow = enableShadows;
        child.receiveShadow = enableShadows;
      });
      setModelWireframe(model, wireframeRef.current);
      state.model = model;
      state.modelKey = loadKey;
      state.scene.add(model);
      fitView("iso");
      setLoadedModelKey(loadKey);
      setModelLoading(false);
    };

    void (async () => {
      try {
        const response = await fetch(selectedAssetUrl, { signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const buffer = await response.arrayBuffer();
        if (!isCurrent()) return;

        if (selectedExtension === "3mf") {
          const model = await new Promise<THREE.Object3D>((resolve, reject) => {
            new GLTFLoader().parse(
              buffer,
              new URL(".", selectedAssetUrl).toString(),
              (result) => resolve(result.scene),
              reject,
            );
          });
          showModel(model);
          return;
        }

        const geometry = new STLLoader().parse(buffer);
        geometry.computeVertexNormals();
        showModel(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
          color: 0xb7c7dc,
          roughness: 0.44,
          metalness: 0.12,
        })));
      } catch (loadError) {
        if (!isCurrent()) return;
        window.cancelAnimationFrame(clearFrame);
        window.cancelAnimationFrame(loadingFrame);
        setModelError(loadError instanceof Error
          ? `模型加载失败：${loadError.message}`
          : "模型加载失败，预览格式可能不受支持。");
        setModelLoading(false);
      }
    })();

    return () => {
      window.cancelAnimationFrame(clearFrame);
      window.cancelAnimationFrame(loadingFrame);
      controller.abort();
      if (loadGeneration.current === generation) loadGeneration.current += 1;
    };
  }, [
    apiBase,
    fitView,
    loadKey,
    selectedAssetUrl,
    selectedExtension,
    selectedItemKey,
    selectedPreviewStatus,
    selectedViewable,
  ]);

  useEffect(() => {
    wireframeRef.current = wireframe;
    const model = viewer.current?.model;
    if (model) setModelWireframe(model, wireframe);
  }, [wireframe]);

  useEffect(() => {
    const state = viewer.current;
    if (!state) return;
    state.controls.autoRotate = autoRotate;
    state.controls.autoRotateSpeed = 1.2;
  }, [autoRotate]);

  const viewableModels = models.filter((model) => model.viewable);
  const supportingFiles = models.filter((model) => !model.viewable);
  const hasMeshInspection = selected?.extension.toLowerCase() === "stl";
  const isThreeMf = selectedExtension === "3mf";
  const modelIsLoaded = Boolean(loadKey && loadedModelKey === loadKey);
  const canRetryPreview = isThreeMf && (selectedPreviewStatus === "failed" || Boolean(modelError));
  const previewProgressMessage = isThreeMf
    ? {
        not_applicable: "此文件不需要生成快速预览。",
        pending: "快速预览已排队，生成完成后会自动加载。",
        processing: "正在后台生成快速预览，生成完成后会自动加载。",
        ready: modelLoading ? "正在加载已生成的快速预览…" : "",
        failed: selected?.preview_error || "快速预览生成失败，请重新尝试。",
      }[selectedPreviewStatus]
    : "";
  const viewerMessage = modelError || previewProgressMessage || (modelLoading ? "正在加载三维视图…" : "");
  const previewTitle = modelError
    ? "预览加载失败"
    : selectedPreviewStatus === "failed"
      ? "快速预览生成失败"
      : selectedPreviewStatus === "pending"
        ? "快速预览等待生成"
        : selectedPreviewStatus === "processing"
          ? "快速预览生成中"
          : modelIsLoaded
            ? "3MF 快速预览已加载"
            : "3MF 快速预览已就绪";
  const previewDescription = modelError
    ? modelError
    : selectedPreviewStatus === "ready"
      ? "浏览器只加载服务端预生成的 GLB，原始 3MF 仍可下载。"
      : selected?.preview_error || "服务端会在后台生成可快速审视的 GLB。";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark">SC</span>
          <div>
            <strong>CAD Review Web</strong>
            <span>局域网成品审视器</span>
          </div>
        </div>
        <div className="connection-status">
          <span className={serviceError ? "status-dot error" : "status-dot"} />
          <span>{serviceError ? "服务异常" : "服务在线"}</span>
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
            <span className="count-pill">{counts[scope]}</span>
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
          <div className="preview-summary" aria-label="快速预览状态汇总">
            <span className="ready">就绪 {previewCounts.ready}</span>
            <span className="pending">等待 {previewCounts.pending}</span>
            <span className="processing">生成中 {previewCounts.processing}</span>
            <span className="failed">失败 {previewCounts.failed}</span>
          </div>
          <div className="model-list">
            {loading && <p className="empty-message">正在读取模型目录…</p>}
            {!loading && viewableModels.length === 0 && (
              <p className="empty-message">
                {scope === "active" ? "尚未发现可审视的 STL 或 3MF。" : scope === "archive" ? "归档区为空。" : "回收站为空。"}
              </p>
            )}
            {viewableModels.map((model) => (
              <button
                type="button"
                key={model.name}
                className={`model-row ${model.name === selectedName ? "active" : ""}`}
                onClick={() => setSelectedName(model.name)}
              >
                <span className="file-badge">{model.extension.toUpperCase()}</span>
                <span className="file-copy">
                  <strong>{model.name.replace(/\.[^.]+$/i, "")}</strong>
                  <span>
                    {formatDate(model.modified)} · {formatBytes(model.size)}
                    {model.extension.toLowerCase() === "3mf" && (
                      <em className={`preview-state ${model.preview_status}`}>
                        {previewStatusLabel(model.preview_status)}
                      </em>
                    )}
                  </span>
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
            <strong>{selected?.name.replace(/\.[^.]+$/i, "") || "等待模型"}</strong>
          </div>
          {viewerMessage && (
            <div className={`viewer-message ${modelError || selectedPreviewStatus === "failed" ? "has-error" : ""}`}>
              <span>{viewerMessage}</span>
              {canRetryPreview && (
                <button type="button" disabled={previewRetrying} onClick={() => void retryPreview()}>
                  {previewRetrying ? "正在提交…" : "重新生成预览"}
                </button>
              )}
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
              <span className={`quality-icon ${hasMeshInspection && selected?.watertight === false ? "bad" : ""}`}>
                {!hasMeshInspection ? "3D" : selected?.watertight === false ? "!" : "✓"}
              </span>
              <div>
                <strong>{isThreeMf ? previewTitle : selected?.watertight === false ? "检测到开放边" : "网格边闭合"}</strong>
                <span>{isThreeMf ? previewDescription : selected?.watertight === false ? "建议修复后再切片" : "满足基础水密检查"}</span>
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
                下载当前 {selected.extension.toUpperCase()}
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
