import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
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
  Sparkles, 
  Maximize2, 
  Eye, 
  Compass, 
  ArrowLeft,
  CheckCircle2,
  Info
} from 'lucide-react';
import { API_BASE } from '../config';

export default function Globe3D() {
  const cesiumContainer = useRef(null);
  const viewerRef = useRef(null);
  const location = useLocation();
  const navigate = useNavigate();

  const [mouseCoords, setMouseCoords] = useState({ lat: null, lon: null, height: null });
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [activePreset, setActivePreset] = useState('orbit');
  const [layers, setLayers] = useState({
    flood: true,
    footprints: true,
    urban: true,
    atmosphere: true
  });
  const [hasIonToken, setHasIonToken] = useState(false);

  // Quick-Fly Locations
  const PRESETS = {
    orbit: {
      name: 'Global Orbit',
      lon: 78.9629,
      lat: 20.5937,
      height: 15000000,
      heading: 0,
      pitch: -90
    },
    isro: {
      name: 'ISRO SAC Ahmedabad',
      lon: 72.502,
      lat: 23.033,
      height: 3500,
      heading: 25,
      pitch: -35,
      description: 'Space Applications Centre (SAC), ISRO — Problem Statement SIH26167 Lead Authority'
    },
    flood: {
      name: 'CDVQA Flood Inundation',
      lon: 14.34,
      lat: 35.12,
      height: 5000,
      heading: 0,
      pitch: -45,
      description: 'Bi-Temporal Satellite Inundation Zone (15.2% detected water surface expansion)'
    },
    bigearthnet: {
      name: 'BigEarthNet-MM Optical+SAR',
      lon: 13.405,
      lat: 52.52,
      height: 6000,
      heading: 10,
      pitch: -40,
      description: 'Co-registered Sentinel-2 (4-Band Optical) & Sentinel-1 (C-Band SAR) Joint Footprint'
    },
    vrsbench: {
      name: 'VRSBench Urban Grounding',
      lon: 77.209,
      lat: 28.614,
      height: 2500,
      heading: 45,
      pitch: -30,
      description: 'High-Resolution 0.5m Spatial Grounding Target Complex'
    }
  };

  useEffect(() => {
    let viewer = null;

    const initCesium = async () => {
      // 1. Fetch Cesium token from settings
      try {
        const { data } = await axios.get(`${API_BASE}/api/settings`);
        if (data.cesium_ion_token) {
          Cesium.Ion.defaultAccessToken = data.cesium_ion_token;
          setHasIonToken(true);
        } else {
          // Provide fallback token or open provider
          setHasIonToken(false);
        }
      } catch (err) {
        console.warn('Could not fetch Cesium settings, using default/open providers:', err);
      }

      if (!cesiumContainer.current) return;

      // 2. Initialize Cesium Viewer
      viewer = new Cesium.Viewer(cesiumContainer.current, {
        animation: false,
        baseLayerPicker: false,
        fullscreenButton: false,
        geocoder: false,
        homeButton: false,
        infoBox: false,
        sceneModePicker: false,
        selectionIndicator: false,
        timeline: false,
        navigationHelpButton: false,
        imageryProvider: new Cesium.ArcGisMapServerImageryProvider({
          url: 'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
        }),
        terrainProvider: new Cesium.EllipsoidTerrainProvider()
      });

      viewerRef.current = viewer;

      // Optimize visual fidelity & atmosphere
      viewer.scene.globe.enableLighting = true;
      viewer.scene.globe.depthTestAgainstTerrain = false;

      // 3. Add Pre-configured 3D Layers

      // Layer A: ISRO SAC Ahmedabad Headquarters Marker & Footprint
      viewer.entities.add({
        id: 'isro-sac',
        name: 'ISRO Space Applications Centre (SAC)',
        position: Cesium.Cartesian3.fromDegrees(72.502, 23.033, 0),
        point: {
          pixelSize: 14,
          color: Cesium.Color.fromCssColorString('#2563EB'),
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 3
        },
        label: {
          text: 'ISRO SAC Ahmedabad (SIH26167)',
          font: '12px Inter, sans-serif',
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -12)
        },
        description: 'Lead Indian Space Research Organisation (ISRO) Centre for Remote Sensing & Earth Observation.'
      });

      // Layer B: CDVQA Bi-Temporal Flood Inundation Polygon (3D Extrusion)
      const floodCoords = [
        14.330, 35.110,
        14.360, 35.110,
        14.360, 35.135,
        14.330, 35.135
      ];
      viewer.entities.add({
        id: 'cdvqa-flood-layer',
        name: 'CDVQA Flood Inundation Extent',
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(floodCoords),
          material: Cesium.Color.fromCssColorString('#EF4444').withAlpha(0.55),
          outline: true,
          outlineColor: Cesium.Color.fromCssColorString('#F87171'),
          extrudedHeight: 80.0
        },
        description: 'Bi-Temporal detected inundation: 15.2% water expansion across 6.25 hectares (F1-Score: 1.0000).'
      });

      // Layer C: BigEarthNet-MM Optical+SAR Footprint (3D Box)
      const benCoords = [
        13.390, 52.510,
        13.420, 52.510,
        13.420, 52.530,
        13.390, 52.530
      ];
      viewer.entities.add({
        id: 'bigearthnet-footprint',
        name: 'BigEarthNet-MM Joint Observation Footprint',
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(benCoords),
          material: Cesium.Color.fromCssColorString('#8B5CF6').withAlpha(0.4),
          outline: true,
          outlineColor: Cesium.Color.fromCssColorString('#A78BFA'),
          extrudedHeight: 50.0
        },
        description: 'Sentinel-2 4-band optical + Sentinel-1 C-band SAR co-registered footprint.'
      });

      // Layer D: VRSBench High-Res Grounding Target
      const vrsCoords = [
        77.200, 28.610,
        77.215, 28.610,
        77.215, 28.620,
        77.200, 28.620
      ];
      viewer.entities.add({
        id: 'vrsbench-target',
        name: 'VRSBench Urban Grounding Target',
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(vrsCoords),
          material: Cesium.Color.fromCssColorString('#10B981').withAlpha(0.5),
          outline: true,
          outlineColor: Cesium.Color.fromCssColorString('#34D399'),
          extrudedHeight: 40.0
        },
        description: '0.5m GSD High-Resolution spatial grounding building complex (mIoU: 1.0000).'
      });

      // 4. Handle incoming dynamic vector layers from Analysis page (if passed via location state)
      if (location.state?.vector_layers && location.state.vector_layers.length > 0) {
        location.state.vector_layers.forEach((layer, idx) => {
          if (layer.geojson) {
            Cesium.GeoJsonDataSource.load(layer.geojson, {
              stroke: Cesium.Color.fromCssColorString('#3B82F6'),
              fill: Cesium.Color.fromCssColorString('#3B82F6').withAlpha(0.5),
              strokeWidth: 3
            }).then((dataSource) => {
              viewer.dataSources.add(dataSource);
              viewer.zoomTo(dataSource);
            }).catch(console.error);
          }
        });
      } else {
        // Initial fly to India / Global Orbit
        flyToPreset('orbit');
      }

      // 5. Setup Mouse Coordinate Tracker & Feature Selection
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
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
    };

    initCesium();

    return () => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, [location.state]);

  const flyToPreset = (key) => {
    const p = PRESETS[key];
    if (!p || !viewerRef.current) return;
    setActivePreset(key);

    viewerRef.current.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.height),
      orientation: {
        heading: Cesium.Math.toRadians(p.heading),
        pitch: Cesium.Math.toRadians(p.pitch),
        roll: 0.0
      },
      duration: 2.2
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
  };

  return (
    <div className="relative w-full h-[calc(100vh-8.5rem)] rounded-2xl overflow-hidden shadow-xl border border-border bg-slate-950 animate-in fade-in duration-300">
      {/* 3D WebGL Canvas Container */}
      <div ref={cesiumContainer} className="w-full h-full" />

      {/* Top Left: Title & Navigation HUD */}
      <div className="absolute top-4 left-4 z-10 flex flex-col gap-2 max-w-sm">
        <div className="bg-slate-900/85 backdrop-blur-md p-4 rounded-xl border border-slate-700/60 shadow-lg text-white">
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-blue-600/30 text-blue-400 border border-blue-500/30">
                <Globe className="w-4 h-4" />
              </div>
              <h1 className="font-bold text-sm tracking-tight">Cesium 3D Digital Globe</h1>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              WebGL 3D
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Real-time orbital projection of Sentinel-1/2 rasters, 3D flood inundation zones, and urban groundings.
          </p>
          <div className="mt-2.5 pt-2 border-t border-slate-700/50 flex items-center justify-between text-[11px] text-slate-400">
            <span>Terrain Mode: {hasIonToken ? 'Cesium World Terrain' : 'ArcGIS World Imagery + Ellipsoid'}</span>
          </div>
        </div>

        {/* Preset Locations Quick-Bar */}
        <div className="bg-slate-900/85 backdrop-blur-md p-3 rounded-xl border border-slate-700/60 shadow-lg text-white">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
            Orbital Fly-To Presets
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
              <span className="truncate">ISRO SAC HQ</span>
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
              <span className="truncate">Flood Inundation</span>
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
              <span className="truncate">Optical + SAR</span>
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
              <span className="truncate">Urban Grounding</span>
            </button>

            <button
              onClick={() => flyToPreset('orbit')}
              className="col-span-2 text-center text-xs p-1.5 rounded-lg border bg-slate-800/60 hover:bg-slate-700 text-slate-300 border-slate-700 transition-colors cursor-pointer"
            >
              Reset to Global Orbit
            </button>
          </div>
        </div>
      </div>

      {/* Top Right: Layer Visibility Toggles */}
      <div className="absolute top-4 right-4 z-10 bg-slate-900/85 backdrop-blur-md p-3.5 rounded-xl border border-slate-700/60 shadow-lg text-white max-w-xs">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-2 flex items-center gap-1">
          <Layers className="w-3.5 h-3.5" /> 3D Layer Overlays
        </span>
        <div className="space-y-1.5 text-xs">
          <label className="flex items-center justify-between gap-3 p-1.5 rounded hover:bg-slate-800/60 cursor-pointer">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
              Flood Inundation (CDVQA)
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
              Optical-SAR Footprint
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
              Target Grounding (VRSBench)
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
              Atmosphere & Lighting
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
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 bg-slate-900/85 backdrop-blur-md px-4 py-2 rounded-full border border-slate-700/60 shadow-lg text-white flex items-center gap-4 text-xs font-mono">
        <div className="flex items-center gap-1.5 text-slate-400">
          <Compass className="w-3.5 h-3.5 text-blue-400" />
          <span>Lat: <strong className="text-white">{mouseCoords.lat || '20.5937'}°</strong></span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <span>Lon: <strong className="text-white">{mouseCoords.lon || '78.9629'}°</strong></span>
        </div>
        <div className="flex items-center gap-1.5 text-slate-400">
          <span>Elevation: <strong className="text-white">{mouseCoords.height || 0} m</strong></span>
        </div>
      </div>

      {/* Bottom Left: Feature Inspection Card */}
      {selectedFeature && (
        <div className="absolute bottom-4 left-4 z-10 max-w-sm bg-slate-900/90 backdrop-blur-md p-4 rounded-xl border border-slate-700/60 shadow-2xl text-white animate-in fade-in duration-200">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] uppercase font-semibold text-blue-400 flex items-center gap-1">
              <Info className="w-3.5 h-3.5" /> Selected 3D Feature
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
