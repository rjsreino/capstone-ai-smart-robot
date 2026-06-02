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
} from "react-native";

import * as Speech from "expo-speech";
import { Audio } from "expo-av";
import { DeviceMotion } from "expo-sensors";
import { LinearGradient } from "expo-linear-gradient";
import Svg, { Circle, Line, Polygon } from "react-native-svg";
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

export default function App() {
  const [serverUrl, setServerUrl] = useState("http://192.168.45.5:8000");
  const [data, setData] = useState(null);
  const [command, setCommand] = useState("");
  const [recording, setRecording] = useState(null);
  const [lastTranscript, setLastTranscript] = useState("");
  const [lastResponse, setLastResponse] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [guidanceEnabled, setGuidanceEnabled] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState("");
  const [history, setHistory] = useState([]);

  const lastSpoken = useRef("");
  const lastSpeakTime = useRef(0);
  const lastCommandTime = useRef(0);
  const guidanceEnabledRef = useRef(true);

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
        const { status } = await DeviceMotion.requestPermissionsAsync();
        if (status !== 'granted') {
          console.log("[Sensors] DeviceMotion permission not granted");
          return;
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
    if (data?.guidance === "TURN LEFT") return "←";
    if (data?.guidance === "TURN RIGHT") return "→";
    if (data?.guidance === "GO FORWARD") return "↑";
    if (data?.guidance === "STOP! DANGER") return "!";
    return "•";
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

      lastCommandTime.current = Date.now();

      await Speech.stop();
      await new Promise((resolve) => setTimeout(resolve, 1600));

      Speech.speak(answer, {
        language: "en-US",
        rate: 0.9,
        pitch: 1.0,
        volume: 1.0,
      });
    } catch (error) {
      console.log("Stop recording error:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const speakGuidance = (json) => {
    if (!guidanceEnabledRef.current || !json) return;
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

    Speech.isSpeakingAsync().then((speaking) => {
      if (!speaking) {
        Speech.speak(message, {
          language: "en-US",
          rate: 0.9,
          pitch: 1.0,
          volume: 1.0,
        });
      }
    });
  };

  const sendCommand = async () => {
    if (!command.trim()) return;

    try {
      const userCommand = command;

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

      lastCommandTime.current = Date.now();

      await Speech.stop();

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: false,
        playThroughEarpieceAndroid: false,
      });

      await new Promise((resolve) => setTimeout(resolve, 2000));

      Speech.speak(answer, {
        language: "en-US",
        rate: 0.9,
        pitch: 1.0,
        volume: 1.0,
      });

      setCommand("");
    } catch (error) {
      console.log("Command error:", error);
    }
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
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, [serverUrl]);

  const left = data?.left_distance ?? 0;
  const center = data?.center_distance ?? 0;
  const right = data?.right_distance ?? 0;
  const guidanceColor = getGuidanceColor();

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
      <ScrollView contentContainerStyle={styles.container}>
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

          {/* Embedded SLAM grid & A* visualizer */}
          <WebView
            source={{ uri: `${serverUrl}/map` }}
            style={styles.webView}
            javaScriptEnabled={true}
            domStorageEnabled={true}
          />
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
          />

          <Pressable style={styles.sendButton} onPress={sendCommand}>
            <Text style={styles.sendText}>SEND COMMAND</Text>
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

        {data?.detections?.map((item, index) => (
          <View key={index} style={styles.objectCard}>
            <View>
              <Text style={styles.objectName}>{item.object}</Text>
              <Text style={styles.objectMeta}>
                {item.position} · {item.distance}
              </Text>
            </View>

            <Text style={styles.objectDepth}>
              {item.depth_meters ? `${item.depth_meters.toFixed(2)}m` : "N/A"}
            </Text>
          </View>
        ))}

        <Text style={styles.sectionTitle}>COMMAND HISTORY</Text>

        {history.map((item, index) => (
          <View key={index} style={styles.historyCard}>
            <Text style={styles.historyUser}>USER: {item.question}</Text>
            <Text style={styles.historyAI}>VICKY: {item.answer}</Text>
          </View>
        ))}
      </ScrollView>
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
  webView: {
    width: "100%",
    height: 350,
    marginTop: 15,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#1e3a8a",
    overflow: "hidden",
  },
});