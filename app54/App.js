import { useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TextInput,
  Button,
} from "react-native";
import * as Speech from "expo-speech";
import { Audio } from "expo-av";

const BASE_URL = "http://192.168.45.6:8000";

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
  web: {
    mimeType: "audio/webm",
    bitsPerSecond: 128000,
  },
};

export default function App() {
  const [data, setData] = useState(null);
  const [command, setCommand] = useState("");
  const [recording, setRecording] = useState(null);

  const lastSpoken = useRef("");
  const lastSpeakTime = useRef(0);
  const lastCommandTime = useRef(0);

  const startRecording = async () => {
    try {
      const permission = await Audio.requestPermissionsAsync();

      if (!permission.granted) {
        console.log("Microphone permission denied");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(RECORDING_OPTIONS);

      setRecording(recording);
      console.log("Recording started");
    } catch (error) {
      console.log("Start recording error:", error);
    }
  };

  const stopRecording = async () => {
    try {
      if (!recording) return;

      await recording.stopAndUnloadAsync();

      await new Promise((resolve) => setTimeout(resolve, 1000));

      const uri = recording.getURI();
      setRecording(null);

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: false,
        playsInSilentModeIOS: true,
      });

      console.log("Audio URI:", uri);
      console.log("Sending voice command...");

      const formData = new FormData();

      formData.append("file", {
        uri,
        name: "command.caf",
        type: "audio/x-caf",
      });

      const response = await fetch(`${BASE_URL}/voice-command`, {
        method: "POST",
        body: formData,
      });

      const text = await response.text();
      console.log("Raw voice response:", text);

      let json;
      try {
        json = JSON.parse(text);
      } catch (e) {
        console.log("Backend returned non-JSON:", text);
        return;
      }

      console.log("Voice response:", json);

      lastCommandTime.current = Date.now();

      await Audio.setAudioModeAsync({
  allowsRecordingIOS: false,
  playsInSilentModeIOS: true,
  shouldDuckAndroid: true,
  playThroughEarpieceAndroid: false,
});

lastCommandTime.current = Date.now();

await Speech.stop();

setTimeout(async () => {
  console.log("Speaking response:", json.response);

  const available = await Speech.isSpeakingAsync();
  console.log("Speech already speaking:", available);

  Speech.speak(json.response, {
    language: "en-US",
    rate: 0.85,
    pitch: 1.0,
  });
}, 1000);
    } catch (error) {
      console.log("Stop recording error:", error);
    }
  };

  const speakGuidance = (json) => {
    if (!json) return;

    if (Date.now() - lastCommandTime.current < 5000) {
      return;
    }

    let message = "";

    if (json.guidance === "STOP! DANGER") {
      message = "Stop. Danger ahead.";
    } else if (json.guidance === "TURN LEFT") {
      message = "Turn left.";
    } else if (json.guidance === "TURN RIGHT") {
      message = "Turn right.";
    } else if (json.guidance === "GO FORWARD") {
      message = "Path clear. Go forward.";
    }

    const firstObject = json.detections?.[0];

    if (firstObject) {
      message += ` ${firstObject.object} detected ${firstObject.distance} at ${firstObject.position}.`;
    }

    if (!message || message === lastSpoken.current) return;

    const now = Date.now();

    if (now - lastSpeakTime.current < 4000) return;

    lastSpoken.current = message;
    lastSpeakTime.current = now;

    Speech.isSpeakingAsync().then((speaking) => {
      if (!speaking) {
        Speech.speak(message, {
          language: "en-US",
          rate: 0.9,
        });
      }
    });
  };

  const sendCommand = async () => {
    if (!command.trim()) return;

    try {
      const response = await fetch(`${BASE_URL}/command`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          command: command,
        }),
      });

      const json = await response.json();

      lastCommandTime.current = Date.now();

      await Speech.stop();

      Speech.speak(json.response, {
        language: "en-US",
        rate: 0.9,
      });

      setCommand("");
    } catch (error) {
      console.log("Command error:", error);
    }
  };

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${BASE_URL}/status`);
        const json = await response.json();
        setData(json);
        speakGuidance(json);
      } catch (error) {
        console.log("Fetch error:", error);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Vision Assistant</Text>

      <TextInput
        style={styles.input}
        placeholder="Ask: what is in front of me?"
        placeholderTextColor="#94a3b8"
        value={command}
        onChangeText={setCommand}
      />

      <Button title="Send Command" onPress={sendCommand} />

      <Button
  title="Test Speak Response"
  onPress={() => Speech.speak("Voice output test", { language: "en-US" })}
/>

      <View style={styles.buttonGap} />

      <Button
        title={recording ? "Stop Recording" : "Start Voice Command"}
        onPress={recording ? stopRecording : startRecording}
      />

      <View style={styles.statusBox}>
        <Text style={styles.guidance}>{data?.guidance ?? "Connecting..."}</Text>
      </View>

      <Text style={styles.text}>
        Left: {data?.left_distance?.toFixed?.(0) ?? 0} mm
      </Text>

      <Text style={styles.text}>
        Center: {data?.center_distance?.toFixed?.(0) ?? 0} mm
      </Text>

      <Text style={styles.text}>
        Right: {data?.right_distance?.toFixed?.(0) ?? 0} mm
      </Text>

      <Text style={styles.subtitle}>Detections</Text>

      {data?.detections?.map((item, index) => (
        <View key={index} style={styles.card}>
          <Text>Object: {item.object}</Text>
          <Text>Position: {item.position}</Text>
          <Text>Distance: {item.distance}</Text>
          <Text>
            Depth: {item.depth_meters ? item.depth_meters.toFixed(2) : "N/A"} m
          </Text>
          <Text>Confidence: {item.confidence}</Text>
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 32,
    backgroundColor: "#0f172a",
  },
  title: {
    fontSize: 30,
    fontWeight: "bold",
    color: "white",
    marginBottom: 20,
  },
  input: {
    backgroundColor: "white",
    padding: 14,
    borderRadius: 10,
    marginBottom: 12,
    fontSize: 16,
  },
  buttonGap: {
    height: 10,
  },
  statusBox: {
    backgroundColor: "#1e293b",
    padding: 20,
    borderRadius: 16,
    marginTop: 20,
    marginBottom: 20,
  },
  guidance: {
    fontSize: 30,
    fontWeight: "bold",
    color: "#f87171",
  },
  text: {
    fontSize: 18,
    color: "white",
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 22,
    color: "white",
    fontWeight: "bold",
    marginTop: 20,
    marginBottom: 12,
  },
  card: {
    backgroundColor: "#e2e8f0",
    padding: 16,
    borderRadius: 12,
    marginBottom: 10,
  },
});