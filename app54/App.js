import { useEffect, useRef, useState } from "react";
import {
  Animated,
  Easing,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
  Platform,
  KeyboardAvoidingView,
} from "react-native";

import * as Speech from "expo-speech";
import { Audio } from "expo-av";
import { DeviceMotion } from "expo-sensors";
import { LinearGradient } from "expo-linear-gradient";
import Svg, { Circle, Line, Polygon, Polyline, Rect } from "react-native-svg";
import { WebView } from "react-native-webview";

const RECORDING_OPTIONS = {
  android: {
    extension: ".m4a",
    outputFormat: Audio.AndroidOutputFormat.MPEG_4,
    audioEncoder: Audio.AndroidAudioEncoder.AAC,
    sampleRate: 44100,
    numberOfChannels: 1,
    bitRate: 128000,
  },
  ios: {
    extension: ".caf",
    audioQuality: Audio.IOSAudioQuality.HIGH,
    sampleRate: 44100,
    numberOfChannels: 1,
    bitRate: 128000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
};

const OBJECT_CLASSIFICATION_COLORS = {
  static: "#38bdf8",
  dynamic: "#f97316",
  landmark: "#facc15",
  exit: "#22f59c",
};

const PROMPT_RESPONSE_TIMEOUT_MS = 10000;
const MAP_SAVE_STABLE_OBS_MS = 300;
const POSE_POLL_INTERVAL_MS = 100;

const clampMapCoord = (value) => Math.max(0, Math.min(Number(value) || 0, 99));

const normalizeDegrees = (value) => ((Number(value) % 360) + 360) % 360;

const angleDelta = (fromDeg, toDeg) => (
  (normalizeDegrees(toDeg) - normalizeDegrees(fromDeg) + 540) % 360
) - 180;

const getMapFacingLabel = (yaw) => {
  const normalized = normalizeDegrees(yaw);
  if (normalized >= 45 && normalized < 135) return "RIGHT";
  if (normalized >= 135 && normalized < 225) return "UP";
  if (normalized >= 225 && normalized < 315) return "LEFT";
  return "DOWN";
};

const getObjectMobility = (object) => {
  const value = String(object?.mobility || object?.classification || "").toLowerCase();
  return value === "static" ? "static" : "dynamic";
};

const getObjectColor = (object) => {
  const label = String(object?.label || object?.detected_label || object?.object || object?.semantic_label || "").toLowerCase();
  if (label.includes("exit")) return OBJECT_CLASSIFICATION_COLORS.exit;
  if (label.includes("door")) return OBJECT_CLASSIFICATION_COLORS.landmark;
  return object?.color || OBJECT_CLASSIFICATION_COLORS[getObjectMobility(object)] || OBJECT_CLASSIFICATION_COLORS.dynamic;
};

export default function App() {
  const [serverUrl, setServerUrl] = useState("http://192.168.45.85:8000");
  const [data, setData] = useState(null);
  const [livePoseData, setLivePoseData] = useState(null);
  const [command, setCommand] = useState("");
  const [recording, setRecording] = useState(null);
  const [lastTranscript, setLastTranscript] = useState("");
  const [lastResponse, setLastResponse] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [guidanceEnabled, setGuidanceEnabled] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");
  const [history, setHistory] = useState([]);
  const [savedMaps, setSavedMaps] = useState([]);
  const [currentMap, setCurrentMap] = useState(null);
  const [mapActionStatus, setMapActionStatus] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState(null);
  const [roomSavePromptVisible, setRoomSavePromptVisible] = useState(false);
  const [roomNameDraft, setRoomNameDraft] = useState("");

  const lastSpoken = useRef("");
  const lastSpeakTime = useRef(0);
  const lastCommandTime = useRef(0);
  const guidanceEnabledRef = useRef(true);
  const pendingPromptRef = useRef(null);
  const promptTimeoutRef = useRef(null);

  const speakText = async (text, options = {}) => {
    if (!text) return;
    try {
      const speaking = Speech.isSpeakingAsync
        ? await Speech.isSpeakingAsync()
        : false;
      if (speaking && !options.interrupt) return;
      if (options.interrupt) {
        await Speech.stop();
      }
      Speech.speak(text, {
        language: "en-US",
        rate: 0.9,
        pitch: 1.0,
        volume: 1.0,
      });
    } catch (error) {
      console.log("Speech error:", error);
    }
  };

  const clearPromptTimeout = () => {
    if (promptTimeoutRef.current) {
      clearTimeout(promptTimeoutRef.current);
      promptTimeoutRef.current = null;
    }
  };

  const dismissPendingPrompt = () => {
    clearPromptTimeout();
    pendingPromptRef.current = null;
    setPendingPrompt(null);
  };

  const expirePendingPrompt = async (promptId) => {
    const activePrompt = pendingPromptRef.current;
    if (!activePrompt || activePrompt.id !== promptId) return;

    clearPromptTimeout();
    pendingPromptRef.current = null;
    setPendingPrompt(null);
    lastSpoken.current = "";
    lastSpeakTime.current = 0;

    try {
      await fetch(`${serverUrl}/answer-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_id: promptId, answer: "no" }),
      });
    } catch (error) {
      console.log("Prompt timeout resolve error:", error);
    }
  };

  const updatePendingPrompt = (json) => {
    if (!json) return;

    if (json.requires_answer && json.answer_type === "yes_no") {
      const promptId = json.prompt_id || json.target || "prompt";
      const guidance = json.guidance || json.response || "Please confirm.";
      const existingPrompt = pendingPromptRef.current;

      if (existingPrompt?.id === promptId && existingPrompt?.guidance === guidance) {
        return;
      }

      clearPromptTimeout();
      const prompt = {
        id: promptId,
        guidance,
        expiresAt: Date.now() + PROMPT_RESPONSE_TIMEOUT_MS,
      };
      pendingPromptRef.current = prompt;
      setPendingPrompt(prompt);
      lastCommandTime.current = Date.now();
      promptTimeoutRef.current = setTimeout(() => {
        expirePendingPrompt(promptId);
      }, PROMPT_RESPONSE_TIMEOUT_MS);
      return;
    }

    if (json.prompt_resolved) {
      dismissPendingPrompt();
    }
  };

  useEffect(() => () => clearPromptTimeout(), []);

  useEffect(() => {
  if (!serverUrl) return;

  const fetchAutopilot = async () => {
    try {
      const response = await fetch(`${serverUrl}/autopilot-guidance`);
      const json = await response.json();
      const isAnswerPrompt = json.requires_answer && json.answer_type === "yes_no";

      updatePendingPrompt(json);
      if (pendingPromptRef.current && !isAnswerPrompt) return;

      if (!json.active || !json.guidance) return;
      if (json.guidance === lastSpoken.current) return;

      lastSpoken.current = json.guidance;

      await speakText(json.guidance);
    } catch (error) {
      console.log("Autopilot guidance error:", error);
    }
  };

  const interval = setInterval(fetchAutopilot, 1200);

  return () => clearInterval(interval);
}, [serverUrl]);

  const pulseAnim = useRef(new Animated.Value(1)).current;
  const scanAnim = useRef(new Animated.Value(0)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;

  const wsRef = useRef(null);

  useEffect(() => {
    let ip = "127.0.0.1";
    try {
      const match = serverUrl.match(/\/\/([^:]+)/);
      if (match && match[1]) {
        ip = match[1];
      }
    } catch (err) {
      console.log("Failed to parse IP from serverUrl:", err);
    }

    const wsUrl = `ws://${ip}:8005`;
    console.log("[IMU WS] Connecting to:", wsUrl);

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[IMU WS] Connected successfully to edge node.");
    };

    ws.onerror = (e) => {
      console.log("[IMU WS] Error:", e.message);
    };

    ws.onclose = () => {
      console.log("[IMU WS] Closed.");
    };

    let subscription = null;
    const startSensors = async () => {
      try {
        const isAvailable = await DeviceMotion.isAvailableAsync();
        if (!isAvailable) {
          console.log("[Sensors] DeviceMotion is not available on this device");
          return;
        }

        if (Platform.OS === 'ios') {
          const { status } = await DeviceMotion.requestPermissionsAsync();
          if (status !== 'granted') {
            console.log("[Sensors] DeviceMotion permission not granted on iOS");
            return;
          }
        }

        DeviceMotion.setUpdateInterval(20);

        subscription = DeviceMotion.addListener((motionData) => {
          if (!motionData) return;

          const accel = motionData.acceleration || { x: 0, y: 0, z: 0 };
          const rot = motionData.rotation || { alpha: 0, beta: 0, gamma: 0 };

          const payload = {
            packet_type: "SMARTPHONE_IMU_STREAM",
            timestamp: Date.now() / 1000,
            linear_accel: { x: accel.x, y: accel.y, z: accel.z },
            rotation_rpy: { roll: rot.alpha, pitch: rot.beta, yaw: rot.gamma }
          };

          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(payload));
          }
        });
      } catch (err) {
        console.log("[Sensors] Error:", err);
      }
    };

    startSensors();

    return () => {
      if (subscription) {
        subscription.remove();
      }
      if (ws) {
        ws.close();
      }
    };
  }, [serverUrl]);

  useEffect(() => {
    guidanceEnabledRef.current = guidanceEnabled;
  }, [guidanceEnabled]);

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.14,
          duration: 850,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 850,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ])
    ).start();

    Animated.loop(
      Animated.timing(scanAnim, {
        toValue: 1,
        duration: 2200,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    ).start();

    Animated.loop(
      Animated.timing(rotateAnim, {
        toValue: 1,
        duration: 9000,
        easing: Easing.linear,
        useNativeDriver: true,
      })
    ).start();
  }, []);

  const addHistory = (question, answer) => {
    setHistory((prev) => [
      { question: question || "-", answer: answer || "-" },
      ...prev.slice(0, 3),
    ]);
  };

  const getAssistantStatus = () => {
    if (recording) return "LISTENING";
    if (isProcessing) return "PROCESSING";
    return "READY";
  };

  const getGuidanceColor = () => {
    if (data?.guidance === "STOP! DANGER") return "#ff3b5c";
    if (data?.guidance === "GO FORWARD") return "#22f59c";
    return "#ffd166";
  };

  const getDirectionArrow = () => {
    if (data?.guidance === "TURN LEFT") return "L";
    if (data?.guidance === "TURN RIGHT") return "R";
    if (data?.guidance === "GO FORWARD") return "^";
    if (data?.guidance === "STOP! DANGER") return "!";
    return ".";
  };

  const getDistanceLabel = (value) => {
    if (value < 700) return "DANGER";
    if (value < 1200) return "CAUTION";
    return "CLEAR";
  };

  const startRecording = async () => {
    try {
      await Speech.stop();
      lastSpoken.current = "";
      lastCommandTime.current = Date.now();

      const permission = await Audio.requestPermissionsAsync();
      if (!permission.granted) return;

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        interruptionModeIOS: 1,
        shouldDuckAndroid: false,
        playThroughEarpieceAndroid: false,
      });

      const { recording } = await Audio.Recording.createAsync(RECORDING_OPTIONS);
      setRecording(recording);
    } catch (error) {
      console.log("Start recording error:", error);
    }
  };

  const stopRecording = async () => {
    if (!recording) return;
    setIsProcessing(true);

    try {
      await recording.stopAndUnloadAsync();
      await new Promise((resolve) => setTimeout(resolve, 1000));

      const uri = recording.getURI();
      setRecording(null);

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        interruptionModeIOS: 1,
        shouldDuckAndroid: false,
        playThroughEarpieceAndroid: false,
      });

      const formData = new FormData();
      formData.append("file", {
        uri,
        name: "command.caf",
        type: "audio/x-caf",
      });

      const response = await fetch(`${serverUrl}/voice-command`, {
        method: "POST",
        body: formData,
      });

      const text = await response.text();
      const json = JSON.parse(text);

      const transcript = json.transcript || "";
      const answer = json.response || "";

      setLastTranscript(transcript);
      setLastResponse(answer);
      addHistory(transcript, answer);
      updatePendingPrompt(json);

      lastCommandTime.current = Date.now();

      await new Promise((resolve) => setTimeout(resolve, 600));
      await speakText(answer, { interrupt: true });
    } catch (error) {
      console.log("Stop recording error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const speakGuidance = (json) => {
    if (!guidanceEnabledRef.current || !json) return;
    if (pendingPromptRef.current) return;
    if (Date.now() - lastCommandTime.current < 5000) return;

    let message = "";

    if (json.guidance === "STOP! DANGER") message = "Stop. Danger ahead.";
    else if (json.guidance === "TURN LEFT") message = "Turn left.";
    else if (json.guidance === "TURN RIGHT") message = "Turn right.";
    else if (json.guidance === "GO FORWARD") message = "Path clear. Go forward.";

    const firstObject = json.detections?.[0];
    if (firstObject) {
      message += ` ${firstObject.object} detected ${firstObject.distance} at ${firstObject.position}.`;
    }

    if (!message || message === lastSpoken.current) return;
    if (Date.now() - lastSpeakTime.current < 4000) return;

    lastSpoken.current = message;
    lastSpeakTime.current = Date.now();

    speakText(message);
  }

  const sendCommand = async () => {
    const userCommand = command.trim();
    if (!userCommand || isProcessing) return;

    setIsProcessing(true);
    setCommand("");

    try {
      const response = await fetch(`${serverUrl}/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: userCommand }),
      });

      const json = await response.json();
      const answer = json.response || "";

      setLastTranscript(userCommand);
      setLastResponse(answer);
      addHistory(userCommand, answer);
      updatePendingPrompt(json);

      lastCommandTime.current = Date.now();

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: false,
        playThroughEarpieceAndroid: false,
      });

      await new Promise((resolve) => setTimeout(resolve, 600));
      await speakText(answer, { interrupt: true });
    } catch (error) {
      console.log("Command error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const runPromptAnswer = async (answerText) => {
    try {
      const activePrompt = pendingPromptRef.current;
      if (activePrompt?.expiresAt && Date.now() > activePrompt.expiresAt) {
        await expirePendingPrompt(activePrompt.id);
        return;
      }
      dismissPendingPrompt();
      setIsProcessing(true);
      setMapActionStatus(`${answerText.toUpperCase()} selected...`);

      const response = await fetch(`${serverUrl}/answer-prompt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt_id: activePrompt?.id,
          answer: answerText,
        }),
      });

      const json = await response.json();
      const answer = json.response || "";

      setLastTranscript(answerText);
      setLastResponse(answer);
      addHistory(answerText, answer);
      updatePendingPrompt(json);

      lastCommandTime.current = Date.now();

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        interruptionModeIOS: 1,
        shouldDuckAndroid: false,
        playThroughEarpieceAndroid: false,
      });

      await speakText(answer, { interrupt: true });
      await refreshMapState();
    } catch (error) {
      console.log("Prompt answer error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const postJson = async (path, body = {}) => {
    const response = await fetch(`${serverUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const json = await response.json();
    if (!response.ok) {
      throw new Error(json.detail || json.message || `Request failed: ${response.status}`);
    }
    return json;
  };

  const refreshMapState = async () => {
    if (!serverUrl) return;
    try {
      const [mapsResponse, currentResponse] = await Promise.all([
        fetch(`${serverUrl}/maps`),
        fetch(`${serverUrl}/current-map`),
      ]);
      const mapsJson = await mapsResponse.json();
      const currentJson = await currentResponse.json();
      setSavedMaps(mapsJson.maps || []);
      setCurrentMap(currentJson.active ? currentJson.map : null);
    } catch (error) {
      console.log("Map state error:", error);
    }
  };

  const runMapAction = async (path, body = {}) => {
    try {
      const label = path.replace("/", "").replace("-", " ");
      setMapActionStatus(`${label}...`);
      const json = await postJson(path, body);
      const answer = path === "/save-map" && json.map_id
        ? "Room saved."
        : json.message || json.status || json.response || "Done.";
      setMapActionStatus(answer);
      setLastResponse(answer);
      addHistory(path, answer);
      await speakText(answer, { interrupt: true });
      await refreshMapState();
    } catch (error) {
      console.log("Map action error:", error);
      const message = error?.message || "Map action failed.";
      setMapActionStatus(message);
      setLastResponse(message);
      await speakText(message, { interrupt: true });
    }
  };

  const askForRoomSaveName = async () => {
    const message = "What would you like to name this room before I save it?";
    const existingName = currentMap?.map_name && currentMap.map_name !== "Room"
      ? currentMap.map_name
      : "";
    setRoomNameDraft(existingName);
    setRoomSavePromptVisible(true);
    setMapActionStatus(message);
    setLastResponse(message);
    lastCommandTime.current = Date.now();
    await speakText(message, { interrupt: true });
  };

  const submitRoomSaveName = async () => {
    const mapName = roomNameDraft.trim();
    if (!mapName) {
      const message = "Please enter a room name first.";
      setMapActionStatus(message);
      setLastResponse(message);
      await speakText(message, { interrupt: true });
      return;
    }
    setRoomSavePromptVisible(false);
    await runMapAction("/save-map", {
      map_name: mapName,
      min_observed_ms: MAP_SAVE_STABLE_OBS_MS,
    });
  };

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${serverUrl}/status`);
        const json = await response.json();

        setData(json);
        setIsConnected(true);
        setLastUpdated(new Date().toLocaleTimeString());
        speakGuidance(json);
      } catch {
        setIsConnected(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 300);
    return () => clearInterval(interval);
  }, [serverUrl]);

  useEffect(() => {
    let isMounted = true;
    let inFlight = false;

    const fetchLivePose = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const response = await fetch(`${serverUrl}/pose`);
        const json = await response.json();
        if (isMounted) {
          setLivePoseData(json);
        }
      } catch {
        // Status polling still handles connection state; pose polling is only for fast facing.
      } finally {
        inFlight = false;
      }
    };

    fetchLivePose();
    const interval = setInterval(fetchLivePose, POSE_POLL_INTERVAL_MS);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [serverUrl]);

  useEffect(() => {
    refreshMapState();
    const interval = setInterval(refreshMapState, 5000);
    return () => clearInterval(interval);
  }, [serverUrl]);

  const left = data?.left_distance ?? 0;
  const center = data?.center_distance ?? 0;
  const right = data?.right_distance ?? 0;
  const guidanceColor = getGuidanceColor();
  const spatialMemory = data?.spatial_memory || {};
  const liveObjects = data?.objects || [];
  const liveStaticObjects = liveObjects.filter((object) => getObjectMobility(object) === "static");
  const liveDynamicObjects = liveObjects.filter((object) => getObjectMobility(object) === "dynamic");
  const mapGrid = spatialMemory.mode === "mapping"
    ? data?.map || currentMap?.static_grid || []
    : currentMap?.static_grid || data?.map || [];
  const mapLandmarks = currentMap?.landmarks || [];
  const mapStaticObjects = currentMap?.static_objects || [];
  const userGrid = livePoseData?.user_grid || data?.user_grid || { x: 50, z: 50 };
  const mapPath = data?.path || [];
  const pose = livePoseData?.pose || data?.pose || {};
  const userMapX = clampMapCoord(userGrid.x ?? 50);
  const userMapZ = clampMapCoord(userGrid.z ?? 50);
  const userYaw = normalizeDegrees(pose.raw_yaw ?? pose.yaw ?? 0);
  const userFacingLabel = getMapFacingLabel(userYaw);
  const poseX = ((Number(pose.x) || 0) / 1000).toFixed(2);
  const poseY = ((Number(pose.y) || 0) / 1000).toFixed(2);
  const poseZ = ((Number(pose.z) || 0) / 1000).toFixed(2);
  const poseRoll = (Number(pose.roll) || 0).toFixed(1);
  const posePitch = (Number(pose.pitch) || 0).toFixed(1);
  const poseYaw = userYaw.toFixed(1);
  const trackingState = livePoseData?.tracking_state || spatialMemory.tracking_state || pose.tracking_state || "UNKNOWN";
  const performance = data?.performance || {};
  const perfFps = (Number(performance.fps) || 0).toFixed(1);
  const perfFrameMs = (Number(performance.frame_ms) || 0).toFixed(0);
  const perfDepthMs = (Number(performance.depth_ms) || 0).toFixed(0);
  const perfYoloMs = (Number(performance.yolo_ms) || 0).toFixed(0);
  const userYawRad = (userYaw * Math.PI) / 180;
  const headingTipX = clampMapCoord(userMapX + 2.2 * Math.sin(userYawRad));
  const headingTipZ = clampMapCoord(userMapZ + 2.2 * Math.cos(userYawRad));
  const headingBackX = userMapX - 1.1 * Math.sin(userYawRad);
  const headingBackZ = userMapZ - 1.1 * Math.cos(userYawRad);
  const headingSideX = 1.05 * Math.cos(userYawRad);
  const headingSideZ = -1.05 * Math.sin(userYawRad);
  const headingMarkerPoints = [
    `${headingTipX},${headingTipZ}`,
    `${clampMapCoord(headingBackX + headingSideX)},${clampMapCoord(headingBackZ + headingSideZ)}`,
    `${clampMapCoord(headingBackX - headingSideX)},${clampMapCoord(headingBackZ - headingSideZ)}`,
  ].join(" ");
  const nextRoutePoint = Array.isArray(mapPath)
    ? mapPath.find(([z, x]) => Math.hypot(Number(x) - userMapX, Number(z) - userMapZ) > 2)
    : null;
  const routeTargetX = nextRoutePoint ? clampMapCoord(nextRoutePoint[1]) : null;
  const routeTargetZ = nextRoutePoint ? clampMapCoord(nextRoutePoint[0]) : null;
  const hasRouteTarget = routeTargetX !== null && routeTargetZ !== null;
  const routeBearing = hasRouteTarget
    ? normalizeDegrees((Math.atan2(routeTargetX - userMapX, routeTargetZ - userMapZ) * 180) / Math.PI)
    : null;
  const routeDelta = hasRouteTarget ? angleDelta(userYaw, routeBearing) : 0;
  const routeTurnLabel = !hasRouteTarget
    ? "NO ROUTE"
    : Math.abs(routeDelta) <= 20
      ? "ON ROUTE"
      : routeDelta > 0
        ? "TURN RIGHT"
        : "TURN LEFT";
  const mapSampleSize = 50;
  const mapCellSize = 100 / mapSampleSize;
  const mapCells = [];

  if (Array.isArray(mapGrid) && mapGrid.length) {
    const rowStep = Math.max(1, Math.floor(mapGrid.length / mapSampleSize));
    const colStep = Math.max(1, Math.floor((mapGrid[0]?.length || 100) / mapSampleSize));
    for (let row = 0; row < mapSampleSize; row += 1) {
      for (let col = 0; col < mapSampleSize; col += 1) {
        const rowStart = row * rowStep;
        const rowEnd = Math.min(rowStart + rowStep, mapGrid.length);
        let hasWall = false;
        let hasUnknown = false;
        let hasSpecial = false;

        for (let sourceRow = rowStart; sourceRow < rowEnd; sourceRow += 1) {
          const rowData = mapGrid[sourceRow] || [];
          const colStart = col * colStep;
          const colEnd = Math.min(colStart + colStep, rowData.length || 100);
          for (let sourceCol = colStart; sourceCol < colEnd; sourceCol += 1) {
            const cellValue = Number(rowData[sourceCol] || 0);
            if (cellValue === 1) hasWall = true;
            else if (cellValue === 2) hasUnknown = true;
            else if (cellValue > 0) hasSpecial = true;
          }
        }

        const value = hasWall ? 1 : hasSpecial ? 3 : hasUnknown ? 2 : 0;
        mapCells.push({ row, col, value });
      }
    }
  }

  const scanTranslate = scanAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [-120, 260],
  });

  const spin = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ["0deg", "360deg"],
  });

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" />
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={styles.container}
          keyboardShouldPersistTaps="handled"
        >
        <LinearGradient
          colors={["#172554", "#0b1026", "#050816"]}
          style={styles.hero}
        >
          <View style={styles.heroGlow} />

          <Text style={styles.brand}>VICKY</Text>
          <Text style={styles.subtitle}>AI VISION NAVIGATION SYSTEM</Text>

          <View style={styles.heroBottom}>
            <View style={styles.connectionPill}>
              <View style={[styles.dot, isConnected ? styles.greenDot : styles.redDot]} />
              <Text style={styles.connectionText}>
                {isConnected ? "ONLINE" : "OFFLINE"}
              </Text>
            </View>

            <Text style={styles.systemText}>{getAssistantStatus()}</Text>
          </View>
        </LinearGradient>

        <View style={styles.settingsCard}>
          <Text style={styles.panelLabel}>AI SERVER SETTINGS</Text>
          <TextInput
            style={styles.settingsInput}
            value={serverUrl}
            onChangeText={setServerUrl}
            placeholder="http://192.168.56.1:8000"
            placeholderTextColor="#64748b"
            autoCapitalize="none"
            autoCorrect={false}
          />
        </View>

        <LinearGradient
          colors={["#101827", "#080d1d"]}
          style={[styles.guidancePanel, { borderColor: guidanceColor }]}
        >
          <Text style={styles.panelLabel}>NAVIGATION COMMAND</Text>

          <View style={styles.commandRow}>
            <Text style={[styles.arrow, { color: guidanceColor }]}>
              {getDirectionArrow()}
            </Text>

            <Text style={[styles.guidance, { color: guidanceColor }]}>
              {data?.guidance ?? "CONNECTING"}
            </Text>
          </View>

          <Text style={styles.updateText}>LAST SYNC {lastUpdated || "--:--:--"}</Text>

          <View style={styles.scannerClip}>
            <Animated.View
              style={[
                styles.scannerLine,
                {
                  backgroundColor: guidanceColor,
                  transform: [{ translateX: scanTranslate }],
                },
              ]}
            />
          </View>
        </LinearGradient>

        {pendingPrompt && (
          <View style={styles.answerPromptCard}>
            <Text style={styles.panelLabel}>CONFIRMATION NEEDED</Text>
            <Text style={styles.answerPromptText}>{pendingPrompt.guidance}</Text>
            <View style={styles.answerButtonRow}>
              <Pressable
                style={[styles.answerButton, styles.yesButton]}
                onPress={() => runPromptAnswer("yes")}
              >
                <Text style={styles.answerButtonText}>YES</Text>
              </Pressable>
              <Pressable
                style={[styles.answerButton, styles.noButton]}
                onPress={() => runPromptAnswer("no")}
              >
                <Text style={styles.answerButtonText}>NO</Text>
              </Pressable>
            </View>
          </View>
        )}

        <View style={styles.radarPanel}>
          <Text style={styles.panelLabel}>SPATIAL RADAR & PATH FINDING</Text>

          <View style={styles.radarWrap}>
            <Animated.View style={{ transform: [{ rotate: spin }] }}>
              <Svg width={220} height={220} viewBox="0 0 220 220">
                <Circle cx="110" cy="110" r="92" stroke="#1e3a8a" strokeWidth="2" fill="none" />
                <Circle cx="110" cy="110" r="62" stroke="#164e63" strokeWidth="1.5" fill="none" />
                <Circle cx="110" cy="110" r="32" stroke="#155e75" strokeWidth="1.2" fill="none" />
                <Line x1="110" y1="18" x2="110" y2="202" stroke="#1e3a8a" strokeWidth="1" />
                <Line x1="18" y1="110" x2="202" y2="110" stroke="#1e3a8a" strokeWidth="1" />
                <Polygon points="110,18 118,110 110,110" fill="rgba(56,189,248,0.30)" />
              </Svg>
            </Animated.View>

            <View style={styles.robotDot} />

            <View style={[styles.radarPoint, styles.leftPoint, left < 1200 && styles.hotPoint]} />
            <View style={[styles.radarPoint, styles.centerPoint, center < 1200 && styles.hotPoint]} />
            <View style={[styles.radarPoint, styles.rightPoint, right < 1200 && styles.hotPoint]} />
          </View>

        </View>

        <View style={styles.mapCard}>
          <View style={styles.mapHeader}>
            <View>
              <Text style={styles.panelLabel}>ROOM MAP</Text>
              <Text style={styles.mapTitle}>
                {spatialMemory.current_map_name || currentMap?.map_name || "No active map"}
              </Text>
            </View>
            <View style={styles.modePill}>
              <Text style={styles.modeText}>{(spatialMemory.mode || "idle").toUpperCase()}</Text>
            </View>
          </View>

          <View style={styles.mapStatsRow}>
            <View style={styles.mapStat}>
              <Text style={styles.mapStatValue}>{spatialMemory.landmark_count ?? mapLandmarks.length}</Text>
              <Text style={styles.mapStatLabel}>LANDMARKS</Text>
            </View>
            <View style={styles.mapStat}>
              <Text style={styles.mapStatValue}>{(spatialMemory.static_object_count ?? mapStaticObjects.length) || liveStaticObjects.length}</Text>
              <Text style={styles.mapStatLabel}>STATIC OBJ</Text>
            </View>
            <View style={styles.mapStat}>
              <Text style={styles.mapStatValue}>{spatialMemory.live_dynamic_count ?? liveDynamicObjects.length}</Text>
              <Text style={styles.mapStatLabel}>DYNAMIC</Text>
            </View>
            <View style={styles.mapStat}>
              <Text style={styles.mapStatValue}>{isConnected ? "ON" : "OFF"}</Text>
              <Text style={styles.mapStatLabel}>SERVER</Text>
            </View>
          </View>

          <View style={styles.poseHudStrip}>
            <Text style={styles.poseHudText}>
              POS X {poseX}m  Y {poseY}m  Z {poseZ}m
            </Text>
            <Text style={styles.poseHudText}>
              ROT R {poseRoll}°  P {posePitch}°  Y {poseYaw}°  | {trackingState}
            </Text>
            <Text style={styles.performanceHudText}>
              PERF FPS {perfFps}  FRAME {perfFrameMs}ms
            </Text>
            <Text style={styles.performanceHudText}>
              DEPTH {perfDepthMs}ms  YOLO {perfYoloMs}ms
            </Text>
          </View>

          <View style={styles.nativeMapFrame}>
            <Svg width="100%" height="100%" viewBox="0 0 100 100">
              <Rect x="0" y="0" width="100" height="100" fill="#020617" />
              {mapCells.map((cell, index) => {
                const fill =
                  cell.value === 1 ? "#64748b" :
                  cell.value === 2 ? "#111827" :
                  cell.value > 0 ? "#facc15" :
                  "#0f172a";
                return (
                  <Rect
                    key={`cell-${index}`}
                    x={cell.col * mapCellSize}
                    y={cell.row * mapCellSize}
                    width={mapCellSize * 0.92}
                    height={mapCellSize * 0.92}
                    fill={fill}
                    opacity={cell.value === 0 ? 0.55 : 0.9}
                  />
                );
              })}
              {mapLandmarks.map((landmark, index) => (
                <Circle
                  key={`landmark-${landmark.id || index}`}
                  cx={Number(landmark.grid_x ?? 50)}
                  cy={Number(landmark.grid_z ?? 50)}
                  r="2.4"
                  fill={String(landmark.type || "").includes("exit") ? "#22f59c" : "#facc15"}
                  stroke="#f8fafc"
                  strokeWidth="0.5"
                />
              ))}
              {mapPath.length > 1 && (
                <Polyline
                  points={mapPath.map(([z, x]) => `${Number(x)},${Number(z)}`).join(" ")}
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth="1.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.95"
                />
              )}
              {mapStaticObjects.map((object, index) => (
                <Circle
                  key={`static-object-${object.id || index}`}
                  cx={Number(object.grid_x ?? object.x ?? 50)}
                  cy={Number(object.grid_z ?? object.z ?? 50)}
                  r="2"
                  fill={getObjectColor({ ...object, mobility: "static" })}
                  stroke="#e0f2fe"
                  strokeWidth="0.35"
                  opacity="0.9"
                />
              ))}
              {liveObjects.map((object, index) => {
                const mobility = getObjectMobility(object);
                const fill = getObjectColor(object);
                return (
                  <Circle
                    key={`live-object-${index}`}
                    cx={Number(object.x ?? 50)}
                    cy={Number(object.z ?? 50)}
                    r={mobility === "static" ? "2.2" : "2.7"}
                    fill={fill}
                    stroke="#f8fafc"
                    strokeWidth="0.45"
                  />
                );
              })}
              {hasRouteTarget && (
                <Line
                  x1={userMapX}
                  y1={userMapZ}
                  x2={routeTargetX}
                  y2={routeTargetZ}
                  stroke="#22c55e"
                  strokeWidth="0.7"
                  strokeDasharray="2 1"
                  strokeLinecap="round"
                  opacity="0.9"
                />
              )}
              <Polygon
                points={headingMarkerPoints}
                fill="#38bdf8"
                stroke="#e0f2fe"
                strokeWidth="0.35"
                opacity="0.96"
              />
            </Svg>
          </View>

          <View style={styles.mapHeadingRow}>
            <Text style={styles.mapHeadingText}>Facing {userYaw.toFixed(0)}° {userFacingLabel}</Text>
            <Text
              style={[
                styles.mapHeadingBadge,
                routeTurnLabel === "ON ROUTE" && styles.onRouteBadge,
                routeTurnLabel === "TURN LEFT" && styles.turnLeftBadge,
                routeTurnLabel === "TURN RIGHT" && styles.turnRightBadge,
              ]}
            >
              {routeTurnLabel}
            </Text>
          </View>

          <View style={styles.legendRow}>
            <Text style={styles.legendText}>Blue triangle: user facing</Text>
            <Text style={styles.legendText}>Gray: wall/block</Text>
            <Text style={styles.legendText}>Dark: unknown</Text>
            <Text style={styles.legendText}>Green/yellow: exit or door</Text>
            <Text style={styles.legendText}>Cyan: static object</Text>
            <Text style={styles.legendText}>Orange: dynamic object</Text>
          </View>

          <View style={styles.mapActions}>
            <Pressable style={styles.mapActionButton} onPress={() => runMapAction("/start-mapping", { map_name: "Room", awaiting_name: true })}>
              <Text style={styles.mapActionText}>START MAP</Text>
            </Pressable>
            <Pressable style={styles.mapActionButton} onPress={askForRoomSaveName}>
              <Text style={styles.mapActionText}>SAVE</Text>
            </Pressable>
            <Pressable style={styles.mapActionButton} onPress={() => runMapAction("/start-navigation", { goal_type: "exit" })}>
              <Text style={styles.mapActionText}>FIND EXIT</Text>
            </Pressable>
            <Pressable style={[styles.mapActionButton, styles.unloadButton]} onPress={() => runMapAction("/unload-map")}>
              <Text style={styles.mapActionText}>UNLOAD</Text>
            </Pressable>
          </View>

          {!!mapActionStatus && (
            <Text style={styles.mapActionStatus}>{mapActionStatus}</Text>
          )}

          {roomSavePromptVisible && (
            <View style={styles.roomNamePromptCard}>
              <Text style={styles.panelLabel}>ROOM NAME</Text>
              <Text style={styles.answerPromptText}>Name this saved map.</Text>
              <TextInput
                style={styles.roomNameInput}
                value={roomNameDraft}
                onChangeText={setRoomNameDraft}
                placeholder="bedroom1"
                placeholderTextColor="#64748b"
                autoCapitalize="none"
                autoCorrect={false}
                onSubmitEditing={submitRoomSaveName}
                returnKeyType="done"
              />
              <View style={styles.answerButtonRow}>
                <Pressable
                  style={[styles.answerButton, styles.yesButton]}
                  onPress={submitRoomSaveName}
                >
                  <Text style={styles.answerButtonText}>SAVE</Text>
                </Pressable>
                <Pressable
                  style={[styles.answerButton, styles.noButton]}
                  onPress={() => setRoomSavePromptVisible(false)}
                >
                  <Text style={styles.answerButtonText}>CANCEL</Text>
                </Pressable>
              </View>
            </View>
          )}

          {savedMaps.slice(0, 3).map((item) => (
            <View key={item.map_id} style={styles.savedMapRow}>
              <View>
                <Text style={styles.savedMapName}>{item.map_name}</Text>
                <Text style={styles.savedMapMeta}>
                  {item.file_name || `${item.map_id}.json`}
                </Text>
                <Text style={styles.savedMapMeta}>
                  {item.landmark_count} landmarks | {item.static_object_count ?? 0} static | {Number(item.coverage_percent || 0).toFixed(1)}% coverage
                </Text>
              </View>
              <Pressable style={styles.loadButton} onPress={() => runMapAction("/load-map", { map_id: item.map_id })}>
                <Text style={styles.loadText}>LOAD</Text>
              </Pressable>
            </View>
          ))}

          <Text style={styles.panelLabel}>DEBUG WEB MAP</Text>
          <View style={styles.webViewContainer}>
            <WebView
              source={{ uri: `${serverUrl}/map?embed=true` }}
              style={styles.webView}
              javaScriptEnabled={true}
              domStorageEnabled={true}
            />
          </View>
        </View>

        <Animated.View style={{ transform: [{ scale: recording ? pulseAnim : 1 }] }}>
          <Pressable
            style={[
              styles.voiceButton,
              recording ? styles.stopButton : styles.askButton,
            ]}
            onPress={recording ? stopRecording : startRecording}
          >
            <View style={styles.micOrb}>
              <Text style={styles.micIcon}>{recording ? "●" : "◉"}</Text>
            </View>

            <Text style={styles.voiceSmall}>
              {recording ? "MICROPHONE ACTIVE" : "VOICE INTERFACE"}
            </Text>

            <Text style={styles.voiceMain}>
              {recording ? "STOP LISTENING" : "ASK VICKY"}
            </Text>
          </Pressable>
        </Animated.View>

        {isProcessing && (
          <LinearGradient colors={["#422006", "#1c1204"]} style={styles.processingBox}>
            <Text style={styles.processingText}>VICKY IS THINKING...</Text>
          </LinearGradient>
        )}

        <View style={styles.switchPanel}>
          <Text style={styles.switchText}>AUTONOMOUS AUDIO GUIDANCE</Text>
          <Switch value={guidanceEnabled} onValueChange={setGuidanceEnabled} />
        </View>

        <View style={styles.consoleCard}>
          <Text style={styles.panelLabel}>LAST INTERACTION</Text>
          <Text style={styles.userText}>USER · {lastTranscript || "-"}</Text>
          <Text style={styles.aiText}>VICKY · {lastResponse || "-"}</Text>
        </View>

        <View style={styles.distanceGrid}>
          <View style={styles.distanceCard}>
            <Text style={styles.distanceLabel}>LEFT</Text>
            <Text style={styles.distanceValue}>{left.toFixed(0)}</Text>
            <Text style={styles.distanceStatus}>{getDistanceLabel(left)}</Text>
          </View>

          <View style={styles.distanceCardCenter}>
            <Text style={styles.distanceLabel}>CENTER</Text>
            <Text style={styles.distanceValue}>{center.toFixed(0)}</Text>
            <Text style={styles.distanceStatus}>{getDistanceLabel(center)}</Text>
          </View>

          <View style={styles.distanceCard}>
            <Text style={styles.distanceLabel}>RIGHT</Text>
            <Text style={styles.distanceValue}>{right.toFixed(0)}</Text>
            <Text style={styles.distanceStatus}>{getDistanceLabel(right)}</Text>
          </View>
        </View>

        <View style={styles.commandBox}>
          <TextInput
            style={styles.input}
            placeholder="Type manual command..."
            placeholderTextColor="#64748b"
            value={command}
            onChangeText={setCommand}
            onSubmitEditing={sendCommand}
            returnKeyType="send"
            editable={!isProcessing}
          />

          <Pressable 
            style={[styles.sendButton, isProcessing && { opacity: 0.5 }]} 
            onPress={sendCommand}
            disabled={isProcessing}
          >
            <Text style={styles.sendText}>
              {isProcessing ? "SENDING..." : "SEND COMMAND"}
            </Text>
          </Pressable>
        </View>

        <View style={styles.promptCard}>
          <Text style={styles.panelLabel}>VOICE PROMPTS</Text>
          <Text style={styles.prompt}>What is in front of me?</Text>
          <Text style={styles.prompt}>Is the path safe?</Text>
          <Text style={styles.prompt}>Where should I go?</Text>
          <Text style={styles.prompt}>Can I turn left?</Text>
        </View>

        <Text style={styles.sectionTitle}>LIVE DETECTIONS</Text>

        {data?.detections?.map((item, index) => {
          const mobility = getObjectMobility(item);
          const color = getObjectColor(item);
          return (
          <View key={index} style={[styles.objectCard, { borderLeftColor: color }]}>
            <View style={{ flex: 1 }}>
              <View style={styles.objectTitleRow}>
                <Text style={styles.objectName}>{item.object}</Text>
                <View style={[styles.classificationBadge, { backgroundColor: color }]}>
                  <Text style={styles.classificationText}>{mobility.toUpperCase()}</Text>
                </View>
              </View>
              <Text style={styles.objectMeta}>
                {item.position} · {item.distance}
              </Text>
            </View>

            <Text style={styles.objectDepth}>
              {item.depth_meters ? `${item.depth_meters.toFixed(2)}m` : "N/A"}
            </Text>
          </View>
          );
        })}

        <Text style={styles.sectionTitle}>COMMAND HISTORY</Text>

        {history.map((item, index) => (
          <View key={index} style={styles.historyCard}>
            <Text style={styles.historyUser}>USER: {item.question}</Text>
            <Text style={styles.historyAI}>VICKY: {item.answer}</Text>
          </View>
        ))}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#050816",
  },
  container: {
    paddingHorizontal: 20,
    paddingTop: 18,
    paddingBottom: 42,
    backgroundColor: "#050816",
  },
  hero: {
    borderRadius: 34,
    padding: 28,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#1f2a44",
    overflow: "hidden",
  },
  heroGlow: {
    position: "absolute",
    right: -40,
    top: -40,
    width: 160,
    height: 160,
    borderRadius: 999,
    backgroundColor: "rgba(56,189,248,0.16)",
  },
  brand: {
    fontSize: 56,
    fontWeight: "900",
    color: "#f8fafc",
    letterSpacing: 7,
  },
  subtitle: {
    color: "#38bdf8",
    marginTop: 6,
    fontSize: 12,
    letterSpacing: 2.1,
    fontWeight: "800",
  },
  heroBottom: {
    marginTop: 24,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  connectionPill: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(15,23,42,0.85)",
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#334155",
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 99,
    marginRight: 8,
  },
  greenDot: {
    backgroundColor: "#22f59c",
  },
  redDot: {
    backgroundColor: "#ff3b5c",
  },
  connectionText: {
    color: "#e2e8f0",
    fontWeight: "900",
    letterSpacing: 1,
    fontSize: 12,
  },
  systemText: {
    color: "#38bdf8",
    fontWeight: "900",
    letterSpacing: 2,
    fontSize: 13,
  },
  guidancePanel: {
    borderRadius: 30,
    padding: 24,
    marginBottom: 18,
    borderWidth: 2,
    overflow: "hidden",
  },
  panelLabel: {
    color: "#64748b",
    fontSize: 12,
    letterSpacing: 2,
    fontWeight: "900",
    marginBottom: 10,
  },
  commandRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  arrow: {
    fontSize: 44,
    fontWeight: "900",
    marginRight: 14,
  },
  guidance: {
    flex: 1,
    fontSize: 34,
    fontWeight: "900",
    letterSpacing: 1,
  },
  updateText: {
    color: "#64748b",
    marginTop: 12,
    fontSize: 12,
    letterSpacing: 1,
  },
  scannerClip: {
    marginTop: 18,
    height: 5,
    borderRadius: 99,
    backgroundColor: "#0f172a",
    overflow: "hidden",
  },
  scannerLine: {
    width: 120,
    height: 5,
    borderRadius: 99,
  },
  radarPanel: {
    backgroundColor: "#070b1f",
    borderRadius: 30,
    padding: 20,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#1e3a8a",
    alignItems: "center",
  },
  radarWrap: {
    width: 220,
    height: 220,
    justifyContent: "center",
    alignItems: "center",
  },
  robotDot: {
    position: "absolute",
    width: 18,
    height: 18,
    borderRadius: 99,
    backgroundColor: "#38bdf8",
    borderWidth: 3,
    borderColor: "#e0f2fe",
  },
  radarPoint: {
    position: "absolute",
    width: 14,
    height: 14,
    borderRadius: 99,
    backgroundColor: "#22f59c",
  },
  hotPoint: {
    backgroundColor: "#ff3b5c",
  },
  leftPoint: {
    left: 48,
    top: 88,
  },
  centerPoint: {
    left: 103,
    top: 38,
  },
  rightPoint: {
    right: 48,
    top: 88,
  },
  voiceButton: {
    padding: 28,
    borderRadius: 34,
    alignItems: "center",
    marginBottom: 16,
  },
  askButton: {
    backgroundColor: "#2563eb",
  },
  stopButton: {
    backgroundColor: "#e11d48",
  },
  micOrb: {
    width: 54,
    height: 54,
    borderRadius: 999,
    backgroundColor: "rgba(255,255,255,0.14)",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.28)",
  },
  micIcon: {
    color: "white",
    fontSize: 25,
    fontWeight: "900",
  },
  voiceSmall: {
    color: "#bfdbfe",
    fontSize: 12,
    letterSpacing: 2,
    fontWeight: "900",
  },
  voiceMain: {
    color: "white",
    fontSize: 26,
    fontWeight: "900",
    marginTop: 6,
  },
  processingBox: {
    borderWidth: 1,
    borderColor: "#facc15",
    borderRadius: 18,
    padding: 14,
    marginBottom: 16,
  },
  processingText: {
    color: "#facc15",
    textAlign: "center",
    fontWeight: "900",
    letterSpacing: 2,
  },
  answerPromptCard: {
    backgroundColor: "rgba(8,47,73,0.94)",
    borderRadius: 24,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#38bdf8",
  },
  roomNamePromptCard: {
    backgroundColor: "rgba(8,47,73,0.94)",
    borderRadius: 18,
    padding: 14,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: "#38bdf8",
  },
  answerPromptText: {
    color: "#f8fafc",
    fontSize: 17,
    lineHeight: 25,
    fontWeight: "800",
    marginBottom: 16,
  },
  answerButtonRow: {
    flexDirection: "row",
    gap: 12,
  },
  answerButton: {
    flex: 1,
    minHeight: 54,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
  },
  yesButton: {
    backgroundColor: "#16a34a",
    borderColor: "#86efac",
  },
  noButton: {
    backgroundColor: "#991b1b",
    borderColor: "#fca5a5",
  },
  answerButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "900",
    letterSpacing: 1,
  },
  switchPanel: {
    backgroundColor: "rgba(15,23,42,0.84)",
    borderRadius: 24,
    padding: 18,
    marginBottom: 16,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  switchText: {
    color: "#f8fafc",
    fontWeight: "900",
    letterSpacing: 1,
  },
  consoleCard: {
    backgroundColor: "rgba(2,6,23,0.86)",
    borderRadius: 24,
    padding: 20,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#334155",
  },
  userText: {
    color: "#7dd3fc",
    fontSize: 15,
    marginBottom: 10,
    lineHeight: 22,
  },
  aiText: {
    color: "#f8fafc",
    fontSize: 16,
    lineHeight: 24,
  },
  distanceGrid: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 18,
  },
  distanceCard: {
    flex: 1,
    backgroundColor: "#0f172a",
    borderRadius: 24,
    padding: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  distanceCardCenter: {
    flex: 1.15,
    backgroundColor: "#111827",
    borderRadius: 24,
    padding: 16,
    borderWidth: 1,
    borderColor: "#38bdf8",
  },
  distanceLabel: {
    color: "#94a3b8",
    fontSize: 11,
    letterSpacing: 2,
    fontWeight: "900",
  },
  distanceValue: {
    color: "white",
    fontSize: 27,
    fontWeight: "900",
    marginTop: 8,
  },
  distanceStatus: {
    color: "#38bdf8",
    fontWeight: "900",
    marginTop: 6,
    fontSize: 12,
  },
  commandBox: {
    backgroundColor: "#0f172a",
    borderRadius: 24,
    padding: 18,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  input: {
    backgroundColor: "#e2e8f0",
    borderRadius: 18,
    padding: 16,
    color: "#020617",
    fontSize: 16,
    marginBottom: 12,
  },
  sendButton: {
    backgroundColor: "#334155",
    borderRadius: 18,
    padding: 16,
    alignItems: "center",
  },
  sendText: {
    color: "white",
    fontWeight: "900",
    letterSpacing: 1,
  },
  promptCard: {
    backgroundColor: "#0b1026",
    borderRadius: 24,
    padding: 18,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#1f2a44",
  },
  prompt: {
    color: "#cbd5e1",
    fontSize: 15,
    marginBottom: 6,
  },
  sectionTitle: {
    color: "#e2e8f0",
    fontSize: 18,
    fontWeight: "900",
    letterSpacing: 2,
    marginBottom: 12,
    marginTop: 10,
  },
  objectCard: {
    backgroundColor: "#e2e8f0",
    borderRadius: 22,
    padding: 18,
    marginBottom: 12,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    borderLeftWidth: 6,
    gap: 10,
  },
  objectTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 8,
  },
  objectName: {
    color: "#020617",
    fontSize: 18,
    fontWeight: "900",
    textTransform: "capitalize",
  },
  objectMeta: {
    color: "#475569",
    marginTop: 4,
    fontSize: 13,
  },
  objectDepth: {
    color: "#0f172a",
    fontWeight: "900",
    fontSize: 15,
  },
  classificationBadge: {
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  classificationText: {
    color: "#020617",
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 0.5,
  },
  historyCard: {
    backgroundColor: "#0f172a",
    borderRadius: 22,
    padding: 18,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  historyUser: {
    color: "#7dd3fc",
    marginBottom: 8,
    fontSize: 14,
  },
  historyAI: {
    color: "#f8fafc",
    fontSize: 14,
    lineHeight: 20,
  },
  settingsCard: {
    backgroundColor: "rgba(15,23,42,0.84)",
    borderRadius: 24,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  settingsInput: {
    backgroundColor: "#1e293b",
    borderRadius: 12,
    padding: 10,
    color: "#f8fafc",
    fontSize: 14,
    borderWidth: 1,
    borderColor: "#334155",
  },
  roomNameInput: {
    backgroundColor: "#1e293b",
    borderRadius: 12,
    padding: 12,
    color: "#f8fafc",
    fontSize: 15,
    borderWidth: 1,
    borderColor: "#334155",
    marginBottom: 12,
  },
  mapHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
  },
  mapTitle: {
    color: "#f8fafc",
    fontSize: 20,
    fontWeight: "900",
  },
  modePill: {
    backgroundColor: "#1e293b",
    borderRadius: 14,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderWidth: 1,
    borderColor: "#38bdf8",
  },
  modeText: {
    color: "#7dd3fc",
    fontSize: 11,
    fontWeight: "900",
    letterSpacing: 1,
  },
  mapStatsRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 16,
    marginBottom: 14,
  },
  mapStat: {
    flex: 1,
    backgroundColor: "#0f172a",
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  mapStatValue: {
    color: "#f8fafc",
    fontSize: 18,
    fontWeight: "900",
  },
  mapStatLabel: {
    color: "#64748b",
    fontSize: 9,
    fontWeight: "900",
    letterSpacing: 1,
    marginTop: 4,
  },
  poseHudStrip: {
    backgroundColor: "#0f172a",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#1e293b",
    paddingHorizontal: 12,
    paddingVertical: 9,
    marginBottom: 12,
    gap: 3,
  },
  poseHudText: {
    color: "#93c5fd",
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.2,
  },
  performanceHudText: {
    color: "#c4b5fd",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.2,
  },
  nativeMapFrame: {
    height: 260,
    borderRadius: 18,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "#334155",
    backgroundColor: "#020617",
  },
  mapHeadingRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 10,
    marginTop: 10,
  },
  mapHeadingText: {
    color: "#bae6fd",
    fontSize: 12,
    fontWeight: "900",
  },
  mapHeadingBadge: {
    color: "#e2e8f0",
    backgroundColor: "#334155",
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 11,
    fontWeight: "900",
    overflow: "hidden",
  },
  onRouteBadge: {
    backgroundColor: "#166534",
    color: "#dcfce7",
  },
  turnLeftBadge: {
    backgroundColor: "#1d4ed8",
    color: "#dbeafe",
  },
  turnRightBadge: {
    backgroundColor: "#7c2d12",
    color: "#ffedd5",
  },
  legendRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 10,
    marginBottom: 14,
  },
  legendText: {
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: "700",
  },
  mapActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginBottom: 12,
  },
  mapActionButton: {
    flex: 1,
    minWidth: "46%",
    backgroundColor: "#1d4ed8",
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: "center",
  },
  unloadButton: {
    backgroundColor: "#475569",
  },
  mapActionText: {
    color: "#f8fafc",
    fontSize: 11,
    fontWeight: "900",
    letterSpacing: 0.5,
  },
  mapActionStatus: {
    color: "#7dd3fc",
    fontSize: 13,
    fontWeight: "800",
    marginBottom: 12,
    lineHeight: 18,
  },
  savedMapRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: "#0f172a",
    borderRadius: 14,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: "#1e293b",
  },
  savedMapName: {
    color: "#e2e8f0",
    fontWeight: "900",
    fontSize: 14,
  },
  savedMapMeta: {
    color: "#94a3b8",
    fontSize: 12,
    marginTop: 3,
  },
  loadButton: {
    backgroundColor: "#334155",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  loadText: {
    color: "#f8fafc",
    fontSize: 11,
    fontWeight: "900",
  },
  webViewContainer: {
    alignSelf: "stretch",
    height: 350,
    marginTop: 15,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#1e3a8a",
    overflow: "hidden",
  },
  webView: {
    flex: 1,
  },
  mapCard: {
    backgroundColor: "#070b1f",
    borderRadius: 30,
    padding: 20,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: "#1e3a8a",
  },
});
