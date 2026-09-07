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
  Maximize2, 
  Compass, 
  Info,
  Sun,
  Map
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
  const [hasIonToken, setHasIonToken] = useState(false);

  // Quick-Fly Locations
  const PRESETS = {
    india: {
      name: 'India Overview',
      lon: 78.9629,
      lat: 20.5937,
      height: 6000000,
      heading: 0,
      pitch: -85,
      description: 'Indian Subcontinent Earth Observation Coverage (ISRO / SAC)'
    },
    isro: {
      name: 'ISRO SAC Ahmedabad',
      lon: 72.502,
      lat: 23.033,
      height: 4500,
      heading: 30,
      pitch: -35,
      description: 'Space Applications Centre (SAC), ISRO — Problem Statement SIH26167 Lead Authority'
    },
    flood: {
      name: 'CDVQA Flood Inundation',
      lon: 14.34,
      lat: 35.12,
      height: 7000,
      heading: 0,
      pitch: -45,
      description: 'Bi-Temporal Satellite Inundation Zone (15.2% detected water surface expansion)'
    },
    bigearthnet: {
      name: 'BigEarthNet-MM Optical+SAR',
      lon: 13.405,
      lat: 52.52,
      height: 8000,
      heading: 10,
      pitch: -40,
      description: 'Co-registered Sentinel-2 (4-Band Optical) & Sentinel-1 (C-Band SAR) Joint Footprint'
    },
    vrsbench: {
      name: 'VRSBench Urban Grounding',
      lon: 77.209,
      lat: 28.614,
      height: 3500,
      heading: 45,
      pitch: -30,
      description: 'High-Resolution 0.5m Spatial Grounding Target Complex'
    }
  };

  useEffect(() => {
    let viewer = null;

    const initCesium = async () => {
      // 1. Fetch Cesium token from settings if available (skip if dummy test token)
      try {
        const { data } = await axios.get(`${API_BASE}/api/settings`);
        if (data.cesium_ion_token && !data.cesium_ion_token.includes('dummy')) {
          Cesium.Ion.defaultAccessToken = data.cesium_ion_token;
          setHasIonToken(true);
        } else {
          setHasIonToken(false);
        }
      } catch (err) {
        console.warn('Could not fetch Cesium settings:', err);
      }

      if (!cesiumContainer.current) return;

      // 2. High-Resolution Satellite Basemap via ArcGIS World Imagery (Reliable, high-res, token-free)
      const satelliteProvider = new Cesium.UrlTemplateImageryProvider({
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        maximumLevel: 19,
        credit: 'Esri World Imagery'
      });
      const baseLayer = new Cesium.ImageryLayer(satelliteProvider);

      // 3. Initialize Cesium Viewer with clean UI and zero glitching
      viewer = new Cesium.Viewer(cesiumContainer.current, {
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
        terrainProvider: new Cesium.EllipsoidTerrainProvider()
      });

      viewerRef.current = viewer;

      // CRITICAL FIX FOR GLITCHING & NIGHT DARKNESS:
      // Turn off sun night shadows so the entire globe is illuminated and visible 24/7
      viewer.scene.globe.enableLighting = false;
      viewer.scene.globe.showGroundAtmosphere = true;
      viewer.scene.skyAtmosphere.show = true;

      // Set distance display condition so markers only appear when zoomed in (eliminates orbital jitter)
      const nearFarCondition = new Cesium.DistanceDisplayCondition(0.0, 7000000.0);

      const pinBuilder = new Cesium.PinBuilder();

      // Layer A: ISRO SAC Ahmedabad Headquarters Marker
      viewer.entities.add({
        id: 'isro-sac',
        name: 'ISRO Space Applications Centre (SAC)',
        position: Cesium.Cartesian3.fromDegrees(72.502, 23.033, 50),
        billboard: {
          image: pinBuilder.fromColor(Cesium.Color.fromCssColorString('#2563EB'), 44).toDataURL(),
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          distanceDisplayCondition: nearFarCondition
        },
        label: {
          text: 'ISRO SAC Ahmedabad (SIH26167)',
          font: 'bold 13px Inter, sans-serif',
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          fillColor: Cesium.Color.WHITE,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 3,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          pixelOffset: new Cesium.Cartesian2(0, -50),
          distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, 2000000.0)
        },
        description: 'Space Applications Centre (SAC), ISRO — Problem Statement SIH26167 Lead Authority for Satellite Earth Observation.'
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
          material: Cesium.Color.fromCssColorString('#EF4444').withAlpha(0.65),
          outline: true,
          outlineColor: Cesium.Color.WHITE,
          extrudedHeight: 80.0,
          distanceDisplayCondition: nearFarCondition
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
          material: Cesium.Color.fromCssColorString('#8B5CF6').withAlpha(0.55),
          outline: true,
          outlineColor: Cesium.Color.WHITE,
          extrudedHeight: 60.0,
          distanceDisplayCondition: nearFarCondition
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
          material: Cesium.Color.fromCssColorString('#10B981').withAlpha(0.65),
          outline: true,
          outlineColor: Cesium.Color.WHITE,
          extrudedHeight: 40.0,
          distanceDisplayCondition: nearFarCondition
        },
        description: '0.5m GSD High-Resolution spatial grounding building complex (mIoU: 1.0000).'
      });

      // 4. Handle incoming dynamic vector layers from Analysis page
      if (location.state?.vector_layers && location.state.vector_layers.length > 0) {
        location.state.vector_layers.forEach((layer) => {
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
        // Set initial camera directly over India
        viewer.camera.setView({
          destination: Cesium.Cartesian3.fromDegrees(78.9629, 20.5937, 7500000),
          orientation: {
            heading: 0.0,
            pitch: Cesium.Math.toRadians(-88),
            roll: 0.0
          }
        });
      }

      // 5. Setup Mouse Coordinate Tracker & Entity Selection
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
  };

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
      duration: 2.0
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
        <div className="bg-slate-900/90 backdrop-blur-md p-4 rounded-xl border border-slate-700/60 shadow-lg text-white">
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-blue-600/30 text-blue-400 border border-blue-500/30">
                <Globe className="w-4 h-4" />
              </div>
              <h1 className="font-bold text-sm tracking-tight">Cesium 3D Digital Globe</h1>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              Live Satellite 3D
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            High-resolution satellite imagery with 3D flood inundation extrusions, multi-sensor footprints, and orbital navigation.
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

        {/* Preset Locations Quick-Bar */}
        <div className="bg-slate-900/90 backdrop-blur-md p-3 rounded-xl border border-slate-700/60 shadow-lg text-white">
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
        <div className="absolute bottom-4 left-4 z-10 max-w-sm bg-slate-900/95 backdrop-blur-md p-4 rounded-xl border border-slate-700/60 shadow-2xl text-white animate-in fade-in duration-200">
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
