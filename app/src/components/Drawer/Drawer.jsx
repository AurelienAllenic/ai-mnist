import React, { useRef, useEffect, useState } from "react";
import { InferenceSession, Tensor } from "onnxjs";
import Btn from "../Btn/Btn";
import "./drawer.css";

const Drawer = () => {
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const sessionRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const isDrawingRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    contextRef.current = ctx;

    const loadModel = async () => {
      try {
        const session = new InferenceSession();
        await session.loadModel("/mnist_cnn.onnx");
        sessionRef.current = session;
        console.log("Modèle ONNX chargé avec succès");
      } catch (err) {
        console.error("Erreur lors du chargement du modèle :", err);
        alert("Impossible de charger le modèle ONNX.");
      }
    };

    loadModel();
  }, []);

  const startDrawing = (e) => {
    isDrawingRef.current = true;
    draw(e);
  };

  const stopDrawing = () => {
    isDrawingRef.current = false;
    contextRef.current.beginPath();
  };

  const draw = (e) => {
    if (!isDrawingRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ctx = contextRef.current;
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(null);
  };

  const findBoundingBox = (data, width, height) => {
    let top = height,
      bottom = 0,
      left = width,
      right = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const brightness = data[i]; // R = G = B pour du noir
        if (brightness < 240) {
          if (x < left) left = x;
          if (x > right) right = x;
          if (y < top) top = y;
          if (y > bottom) bottom = y;
        }
      }
    }

    return { top, bottom, left, right };
  };

  const preprocessImage = (canvas) => {
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    const imgData = ctx.getImageData(0, 0, width, height);
    const { top, bottom, left, right } = findBoundingBox(
      imgData.data,
      width,
      height
    );

    if (left > right || top > bottom) {
      throw new Error("Aucun chiffre détecté");
    }

    const padding = 10;
    const croppedLeft = Math.max(0, left - padding);
    const croppedTop = Math.max(0, top - padding);
    const croppedW = Math.min(width, right + padding) - croppedLeft;
    const croppedH = Math.min(height, bottom + padding) - croppedTop;

    const croppedCanvas = document.createElement("canvas");
    croppedCanvas.width = croppedW;
    croppedCanvas.height = croppedH;
    const croppedCtx = croppedCanvas.getContext("2d");
    croppedCtx.drawImage(
      canvas,
      croppedLeft,
      croppedTop,
      croppedW,
      croppedH,
      0,
      0,
      croppedW,
      croppedH
    );

    const resizedCanvas = document.createElement("canvas");
    resizedCanvas.width = 28;
    resizedCanvas.height = 28;
    const resizedCtx = resizedCanvas.getContext("2d");
    resizedCtx.fillStyle = "white";
    resizedCtx.fillRect(0, 0, 28, 28);

    const scale = Math.min(22 / croppedW, 22 / croppedH);
    const newW = croppedW * scale;
    const newH = croppedH * scale;
    const dx = (28 - newW) / 2;
    const dy = (28 - newH) / 2;

    resizedCtx.drawImage(
      croppedCanvas,
      0,
      0,
      croppedW,
      croppedH,
      dx,
      dy,
      newW,
      newH
    );

    const data = resizedCtx.getImageData(0, 0, 28, 28).data;
    const input = new Float32Array(1 * 1 * 28 * 28);

    for (let i = 0; i < data.length; i += 4) {
      const gray = data[i];
      input[i / 4] = (gray / 255.0 - 0.5) / 0.5;
    }

    return input;
  };

  const predict = async () => {
    if (!sessionRef.current) {
      alert("Modèle non chargé.");
      return;
    }

    try {
      const input = preprocessImage(canvasRef.current);
      const tensor = new Tensor(input, "float32", [1, 1, 28, 28]);
      const outputMap = await sessionRef.current.run([tensor]);
      const key = Array.from(outputMap.keys())[0];
      const output = outputMap.get(key).data;

      const maxIndex = output.indexOf(Math.max(...output));
      const exp = output.map((x) => Math.exp(x));
      const softmax = exp.map((x) => x / exp.reduce((a, b) => a + b));
      const conf = softmax[maxIndex];

      setPrediction(maxIndex);
      setConfidence(conf);
    } catch (err) {
      console.error("Erreur lors de la prédiction :", err);
      alert("Erreur : " + err.message);
    }
  };

  return (
    <div className="drawer-container">
      <h2 className="drawer-title">Dessinez un chiffre</h2>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        className="drawer-canvas"
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseOut={stopDrawing}
        onMouseMove={draw}
      />
      <div className="drawer-controls">
        <Btn type="retry" onClick={clearCanvas} />
        <Btn type="generate" onClick={predict} />
      </div>
      {prediction !== null && (
        <div className="prediction-result">
          <div>Résultat : {prediction}</div>
          {confidence !== null && (
            <div>Confiance : {(confidence * 100).toFixed(1)}%</div>
          )}
        </div>
      )}
    </div>
  );
};

export default Drawer;
