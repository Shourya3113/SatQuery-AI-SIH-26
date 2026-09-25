import { useEffect, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import axios from 'axios';
import { 
  Globe, 
  Layers, 
  MapPin, 
  Satellite, 
  Waves, 
  Building2, 
  Compass, 
  Info
} from 'lucide-react';
import { API_BASE } from '../config';

export default function Globe3D() {
  const cesiumContainer = useRef(null);
  const viewerRef = useRef(null);
  const location = useLocation();

  const [mouseCoords, setMouseCoords] = useState({ lat: null, lon: null, height: null });
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [activePreset, setActivePreset] = useState('india');
  const [activeBasemap, setActiveBasemap] = useState('satellite');
  const [layers, setLayers] = useState({
    flood: true,
    footprints: true,
    urban: true,
    atmosphere: true
  });

  // 100% Pan-India Remote Sensing Observation Scenarios
  const PRESETS = {
    india: {
      name: 'India Overview',
      lon: 78.9629,
      lat: 21.5,
      range: 4800000,
      heading: 0,
      pitch: -89.9,
      description: 'National Synoptic Satellite Earth Observation Coverage (ISRO / SAC)'
    },
    isro: {
      name: 'ISRO SAC Ahmedabad (Gujarat)',
      lon: 72.502,
      lat: 23.033,
      range: 3500,
      heading: 0,
      pitch: -45,
      description: 'Space Applications Centre (SAC), ISRO, Ahmedabad — Problem Statement SIH26167 Host & Lead Remote Sensing Centre'
    },
    flood: {
      name: 'Brahmaputra Flood Inundation (Assam)',
      lon: 93.150,
      lat: 26.600,
      range: 6500,
      heading: 0,
      pitch: -45,
      description: 'Brahmaputra River Basin, Assam — 15.2% detected monsoon flood inundation across 6.25 hectares (CDVQA Benchmark F1-Score: 1.0000)'
    },
    bigearthnet: {
      name: 'Sundarbans Optical+SAR Fusion (West Bengal)',
      lon: 88.850,
      lat: 21.950,
      range: 6500,
      heading: 0,
      pitch: -45,
      description: 'Sundarbans Delta, West Bengal — Sentinel-2 Optical (4-Band) & Sentinel-1 SAR (C-Band microwave radar penetrating tropical cloud cover)'
    },
    vrsbench: {
      name: 'New Delhi Urban Grounding',
      lon: 77.220,
      lat: 28.615,
      range: 3000,
      heading: 0,
      pitch: -40,
      description: 'New Delhi Central Complex — 0.5m High-Resolution spatial grounding target building complex (VRSBench Benchmark mIoU: 1.0000)'
    }
  };

  useEffect(() => {
    let isCancelled = false;

    const initCesium = async () => {
      // 1. Single-instance ref guard
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        return;
      }

      // Check optional token from backend
      try {
        const { data } = await axios.get(`${API_BASE}/api/settings`);
        if (data.cesium_ion_token && !data.cesium_ion_token.includes('dummy')) {
          Cesium.Ion.defaultAccessToken = data.cesium_ion_token;
        }
      } catch (e) {
        console.warn('Cesium settings query fallback:', e);
      }

      if (isCancelled || !cesiumContainer.current) return;

      // Clean container DOM to guarantee zero zombie canvases
      cesiumContainer.current.innerHTML = '';

      // 2. High-Resolution Satellite Basemap
      const satelliteProvider = new Cesium.UrlTemplateImageryProvider({
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        maximumLevel: 19,
        credit: 'Esri World Imagery'
      });
      const baseLayer = new Cesium.ImageryLayer(satelliteProvider);

      // 3. Initialize Cesium Viewer with requestRenderMode
      const viewer = new Cesium.Viewer(cesiumContainer.current, {
        animation: false,
        baseLayer: baseLayer,
        baseLayerPicker: false,
        fullscreenButton: false,
        geocoder: false,
        homeButton: false,
        infoBox: false,
        sceneModePicker: false,
        selectionIndicator: false,
        timeline: false,
        navigationHelpButton: false,
        terrainProvider: new Cesium.EllipsoidTerrainProvider(),
        requestRenderMode: true,
        maximumRenderTimeChange: Infinity,
        msaaSamples: 4
      });

      viewerRef.current = viewer;

      // Daylight on all sides (no night darkness)
      viewer.scene.globe.enableLighting = false;
      viewer.scene.globe.showGroundAtmosphere = true;
      viewer.scene.skyAtmosphere.show = true;

      // 4. Create Jitter-Free Indian Geospatial Vector Overlays
      const pinBuilder = new Cesium.PinBuilder();

      // Layer A: ISRO SAC Ahmedabad Headquarters Marker (Gujarat)
      viewer.entities.add({
        id: 'isro-sac',
        name: 'ISRO Space Applications Centre (SAC)',
        position: Cesium.Cartesian3.fromDegrees(72.502, 23.033, 0),
        billboard: {
          image: pinBuilder.fromColor(Cesium.Color.fromCssColorString('#2563EB'), 42).toDataURL(),
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY
        },
        label: {
          text: 'ISRO SAC Ahmedabad (SIH26167)',
          font: 'bold 12px Inter, sans-serif',
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 3,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -48),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, 3000000.0)
        },
        description: 'Space Applications Centre (SAC), ISRO, Ahmedabad — Problem Statement SIH26167 Lead Authority for Satellite Earth Observation.'
      });

      // Layer B: Brahmaputra River Basin Flood Inundation Polygon (Assam, India)
      const floodCoords = [
        93.120, 26.580,
        93.180, 26.580,
        93.180, 26.620,
        93.120, 26.620
      ];
      viewer.entities.add({
        id: 'cdvqa-flood-layer',
        name: 'Brahmaputra Flood Inundation (Assam)',
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(floodCoords),
          material: Cesium.Color.fromCssColorString('#EF4444').withAlpha(0.65),
          outline: true,
          outlineColor: Cesium.Color.WHITE,
          classificationType: Cesium.ClassificationType.BOTH
        },
        description: 'Brahmaputra River Basin (Assam) — 15.2% detected monsoon flood inundation across 6.25 hectares (CDVQA Benchmark Test Pair).'
      });

      // Layer C: Sundarbans Coastal Mangrove Optical+SAR Multimodal Footprint (West Bengal, India)
      const benCoords = [
        88.810, 21.920,
        88.890, 21.920,
        88.890, 21.980,
        88.810, 21.980
      ];
      viewer.entities.add({
        id: 'bigearthnet-footprint',
        name: 'Sundarbans Optical+SAR Fusion Footprint (West Bengal)',
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(benCoords),
          material: Cesium.Color.fromCssColorString('#8B5CF6').withAlpha(0.55),
          outline: true,
          outlineColor: Cesium.Color.WHITE,
          classificationType: Cesium.ClassificationType.BOTH
        },
        description: 'Sundarbans Delta (West Bengal) — Co-registered Sentinel-2 (Optical 4-band) & Sentinel-1 SAR (C-Band microwave radar penetrating tropical cloud cover).'
      });

      // Layer D: New Delhi Urban Infrastructure Complex (Delhi, India)
      const vrsCoords = [
        77.208, 28.605,
        77.232, 28.605,
        77.232, 28.625,
        77.208, 28.625
      ];
      viewer.entities.add({
        id: 'vrsbench-target',
        name: 'New Delhi Urban Grounding Target (Delhi)',
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(vrsCoords),
          material: Cesium.Color.fromCssColorString('#10B981').withAlpha(0.65),
          outline: true,
          outlineColor: Cesium.Color.WHITE,
          classificationType: Cesium.ClassificationType.BOTH
        },
        description: 'New Delhi Central Complex — 0.5m High-Resolution spatial grounding target building complex (Text-Guided Grounding Target).'
      });

      // 5. Handle incoming dynamic vector layers from Analysis page
      if (location.state?.vector_layers && location.state.vector_layers.length > 0) {
        location.state.vector_layers.forEach((layer) => {
          if (layer.geojson) {
            Cesium.GeoJsonDataSource.load(layer.geojson, {
              stroke: Cesium.Color.fromCssColorString('#3B82F6'),
              fill: Cesium.Color.fromCssColorString('#3B82F6').withAlpha(0.5),
              strokeWidth: 3,
              clampToGround: true
            }).then((dataSource) => {
              viewer.dataSources.add(dataSource);
              viewer.zoomTo(dataSource);
              viewer.scene.requestRender();
            }).catch(console.error);
          }
        });
      } else {
        // Center initial camera directly over India in dead center
        const targetCenter = Cesium.Cartesian3.fromDegrees(78.9629, 21.5, 0);
        const bs = new Cesium.BoundingSphere(targetCenter, 1000);
        viewer.camera.flyToBoundingSphere(bs, {
          offset: new Cesium.HeadingPitchRange(0.0, Cesium.Math.toRadians(-89.9), 4800000),
          duration: 0.0
        });
      }

      // Initial render pass
      viewer.scene.requestRender();

      // 6. Setup Mouse Coordinate Tracker & Entity Selection
      const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);

      handler.setInputAction((movement) => {
        const ray = viewer.camera.getPickRay(movement.endPosition);
        const position = viewer.scene.globe.pick(ray, viewer.scene);
        if (position) {
          const cartographic = Cesium.Cartographic.fromCartesian(position);
          setMouseCoords({
            lon: Cesium.Math.toDegrees(cartographic.longitude).toFixed(4),
            lat: Cesium.Math.toDegrees(cartographic.latitude).toFixed(4),
            height: Math.round(cartographic.height)
          });
        }
      }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

      handler.setInputAction((click) => {
        const pickedObject = viewer.scene.pick(click.position);
        if (Cesium.defined(pickedObject) && pickedObject.id) {
          const entity = pickedObject.id;
          setSelectedFeature({
            name: entity.name || entity.id,
            description: entity.description ? entity.description.getValue() : 'Geospatial Satellite Feature',
            id: entity.id
          });
        } else {
          setSelectedFeature(null);
        }
        viewer.scene.requestRender();
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
    };

    initCesium();

    return () => {
      isCancelled = true;
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
      if (cesiumContainer.current) {
        cesiumContainer.current.innerHTML = '';
      }
    };
  }, [location.state]);

  const switchBasemap = (type) => {
    if (!viewerRef.current) return;
    setActiveBasemap(type);
    viewerRef.current.imageryLayers.removeAll();

    let provider;
    if (type === 'satellite') {
      provider = new Cesium.UrlTemplateImageryProvider({
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        maximumLevel: 19,
        credit: 'Esri World Imagery'
      });
    } else {
      provider = new Cesium.UrlTemplateImageryProvider({
        url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        maximumLevel: 19,
        credit: 'OpenStreetMap'
      });
    }

    viewerRef.current.imageryLayers.addImageryProvider(provider);
    viewerRef.current.scene.requestRender();
  };

  const flyToPreset = (key) => {
    const p = PRESETS[key];
    if (!p || !viewerRef.current) return;
    setActivePreset(key);

    const targetCenter = Cesium.Cartesian3.fromDegrees(p.lon, p.lat, 0);
    const boundingSphere = new Cesium.BoundingSphere(targetCenter, 100);

    viewerRef.current.camera.flyToBoundingSphere(boundingSphere, {
      offset: new Cesium.HeadingPitchRange(
        Cesium.Math.toRadians(p.heading),
        Cesium.Math.toRadians(p.pitch),
        p.range
      ),
      duration: 2.0,
      complete: () => {
        viewerRef.current?.scene?.requestRender();
      }
    });

    if (p.description) {
      setSelectedFeature({
        name: p.name,
        description: p.description,
        id: key
      });
    }
  };

  const toggleLayer = (layerKey) => {
    const updated = !layers[layerKey];
    setLayers((prev) => ({ ...prev, [layerKey]: updated }));

    if (!viewerRef.current) return;
    const entityMap = {
      flood: 'cdvqa-flood-layer',
      footprints: 'bigearthnet-footprint',
      urban: 'vrsbench-target'
    };

    const entityId = entityMap[layerKey];
    if (entityId) {
      const entity = viewerRef.current.entities.getById(entityId);
      if (entity) {
        entity.show = updated;
      }
    }

    if (layerKey === 'atmosphere') {
      viewerRef.current.scene.globe.showAtmosphere = updated;
      viewerRef.current.scene.skyAtmosphere.show = updated;
    }

    viewerRef.current.scene.requestRender();
  };

  return (
    <div className="relative w-full h-[calc(100vh-8.5rem)] rounded-2xl overflow-hidden shadow-xl border border-border bg-slate-950 animate-in fade-in duration-300">
      {/* 3D WebGL Canvas Container */}
      <div ref={cesiumContainer} className="w-full h-full" />

      {/* Top Left: Title & Navigation HUD */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 max-w-sm">
        <div className="bg-slate-900/90 backdrop-blur-md p-4 rounded-xl border border-slate-700/60 shadow-lg text-white">
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-blue-600/30 text-blue-400 border border-blue-500/30">
                <Globe className="w-4 h-4" />
              </div>
              <h1 className="font-bold text-sm tracking-tight">Cesium 3D Digital Globe</h1>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              India 3D Earth
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Pan-India satellite earth observation scenarios: ISRO SAC Ahmedabad, Brahmaputra floodplains, Sundarbans delta, and Delhi infrastructure.
          </p>

          {/* Basemap Switcher */}
          <div className="mt-3 pt-2.5 border-t border-slate-700/50 flex items-center gap-2">
            <span className="text-[10px] uppercase font-semibold text-slate-400 mr-1">Basemap:</span>
            <button
              onClick={() => switchBasemap('satellite')}
              className={`text-xs px-2.5 py-1 rounded-md border font-medium transition-all cursor-pointer ${
                activeBasemap === 'satellite'
                  ? 'bg-blue-600 text-white border-blue-500'
                  : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
              }`}
            >
              Satellite
            </button>
            <button
              onClick={() => switchBasemap('osm')}
              className={`text-xs px-2.5 py-1 rounded-md border font-medium transition-all cursor-pointer ${
                activeBasemap === 'osm'
                  ? 'bg-blue-600 text-white border-blue-500'
                  : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
              }`}
            >
              Street / Topo
            </button>
          </div>
        </div>

        {/* Preset Locations Quick-Bar (100% Pan-India) */}
        <div className="bg-slate-900/90 backdrop-blur-md p-3 rounded-xl border border-slate-700/60 shadow-lg text-white">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
            Pan-India Fly-To Presets
          </span>
          <div className="grid grid-cols-2 gap-1.5">
            <button
              onClick={() => flyToPreset('isro')}
              className={`text-left text-xs p-2 rounded-lg border transition-all flex items-center gap-1.5 cursor-pointer ${
                activePreset === 'isro'
                  ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
                  : 'bg-slate-800/80 hover:bg-slate-700/90 text-slate-200 border-slate-700'
              }`}
            >
              <MapPin className="w-3.5 h-3.5 text-blue-400 shrink-0" />
              <span className="truncate">ISRO SAC (Ahmedabad)</span>
            </button>

            <button
              onClick={() => flyToPreset('flood')}
              className={`text-left text-xs p-2 rounded-lg border transition-all flex items-center gap-1.5 cursor-pointer ${
                activePreset === 'flood'
                  ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
                  : 'bg-slate-800/80 hover:bg-slate-700/90 text-slate-200 border-slate-700'
              }`}
            >
              <Waves className="w-3.5 h-3.5 text-red-400 shrink-0" />
              <span className="truncate">Brahmaputra (Assam)</span>
            </button>

            <button
              onClick={() => flyToPreset('bigearthnet')}
              className={`text-left text-xs p-2 rounded-lg border transition-all flex items-center gap-1.5 cursor-pointer ${
                activePreset === 'bigearthnet'
                  ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
                  : 'bg-slate-800/80 hover:bg-slate-700/90 text-slate-200 border-slate-700'
              }`}
            >
              <Satellite className="w-3.5 h-3.5 text-purple-400 shrink-0" />
              <span className="truncate">Sundarbans (WB)</span>
            </button>

            <button
              onClick={() => flyToPreset('vrsbench')}
              className={`text-left text-xs p-2 rounded-lg border transition-all flex items-center gap-1.5 cursor-pointer ${
                activePreset === 'vrsbench'
                  ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
                  : 'bg-slate-800/80 hover:bg-slate-700/90 text-slate-200 border-slate-700'
              }`}
            >
              <Building2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span className="truncate">New Delhi Urban</span>
            </button>

            <button
              onClick={() => flyToPreset('india')}
              className="col-span-2 text-center text-xs p-1.5 rounded-lg border bg-slate-800/60 hover:bg-slate-700 text-slate-300 border-slate-700 transition-colors cursor-pointer"
            >
              Reset to India Overview
            </button>
          </div>
        </div>
      </div>

      {/* Top Right: Layer Visibility Toggles */}
      <div className="absolute top-4 right-4 z-10 bg-slate-900/90 backdrop-blur-md p-3.5 rounded-xl border border-slate-700/60 shadow-lg text-white max-w-xs">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-2 flex items-center gap-1">
          <Layers className="w-3.5 h-3.5" /> 3D Layer Overlays
        </span>
        <div className="space-y-1.5 text-xs">
          <label className="flex items-center justify-between gap-3 p-1.5 rounded hover:bg-slate-800/60 cursor-pointer">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
              Brahmaputra Flood (Assam)
            </span>
            <input 
              type="checkbox" 
              checked={layers.flood} 
              onChange={() => toggleLayer('flood')} 
              className="accent-blue-600 cursor-pointer" 
            />
          </label>

          <label className="flex items-center justify-between gap-3 p-1.5 rounded hover:bg-slate-800/60 cursor-pointer">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
              Sundarbans Optical+SAR (WB)
            </span>
            <input 
              type="checkbox" 
              checked={layers.footprints} 
              onChange={() => toggleLayer('footprints')} 
              className="accent-blue-600 cursor-pointer" 
            />
          </label>

          <label className="flex items-center justify-between gap-3 p-1.5 rounded hover:bg-slate-800/60 cursor-pointer">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              New Delhi Urban Grounding
            </span>
            <input 
              type="checkbox" 
              checked={layers.urban} 
              onChange={() => toggleLayer('urban')} 
              className="accent-blue-600 cursor-pointer" 
            />
          </label>

          <label className="flex items-center justify-between gap-3 p-1.5 rounded hover:bg-slate-800/60 cursor-pointer">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
              Atmosphere
            </span>
            <input 
              type="checkbox" 
              checked={layers.atmosphere} 
              onChange={() => toggleLayer('atmosphere')} 
              className="accent-blue-600 cursor-pointer" 
            />
          </label>
        </div>
      </div>

      {/* Bottom Center: Live Coordinates HUD */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 bg-slate-900/90 backdrop-blur-md px-4 py-2 rounded-full border border-slate-700/60 shadow-lg text-white flex items-center gap-4 text-xs font-mono">
        <div className="flex items-center gap-1.5 text-slate-400">
          <Compass className="w-3.5 h-3.5 text-blue-400" />
          <span>Lat: <strong className="text-white">{mouseCoords.lat || '21.5000'}° N</strong></span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <span>Lon: <strong className="text-white">{mouseCoords.lon || '78.9629'}° E</strong></span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <span>Elevation: <strong className="text-white">{mouseCoords.height || 0} m</strong></span>
        </div>
      </div>

      {/* Bottom Left: Feature Inspection Card */}
      {selectedFeature && (
        <div className="absolute bottom-4 left-4 z-10 max-w-sm bg-slate-900/95 backdrop-blur-md p-4 rounded-xl border border-slate-700/60 shadow-2xl text-white animate-in fade-in duration-200">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] uppercase font-semibold text-blue-400 flex items-center gap-1">
              <Info className="w-3.5 h-3.5" /> Selected Indian Scenario
            </span>
            <button 
              onClick={() => setSelectedFeature(null)} 
              className="text-slate-400 hover:text-white text-xs px-1"
            >
              ✕
            </button>
          </div>
          <h3 className="font-bold text-sm text-white">{selectedFeature.name}</h3>
          <p className="text-xs text-slate-300 mt-1 leading-relaxed">
            {selectedFeature.description}
          </p>
        </div>
      )}
    </div>
  );
}
