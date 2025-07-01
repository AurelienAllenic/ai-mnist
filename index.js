const MODEL_PATH = "mnist_cnn.onnx";

let session = null;

const setupCanvas = () => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#000";

  let active = false;

  const position = (e) => {
    const r = canvas.getBoundingClientRect();
    if (e.touches) {
      const t = e.touches[0];
      return [t.clientX - r.left, t.clientY - r.top];
    }
    return [e.clientX - r.left, e.clientY - r.top];
  };

  const start = (e) => {
    e.preventDefault();
    active = true;
    const [x, y] = position(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    if (!active) return;
    e.preventDefault();
    const [x, y] = position(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const stop = () => {
    active = false;
    ctx.beginPath();
  };

  canvas.addEventListener("mousedown", start);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stop);
  canvas.addEventListener("mouseleave", stop);
  canvas.addEventListener("touchstart", start, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", stop);
  canvas.addEventListener("touchcancel", stop);

  return { canvas, ctx };
};

const extractInput = (sourceCanvas) => {
  const tmp = document.createElement("canvas");
  tmp.width = tmp.height = 28;
  const ctx = tmp.getContext("2d");
  ctx.drawImage(sourceCanvas, 0, 0, 28, 28);

  const data = ctx.getImageData(0, 0, 28, 28);
  const pixels = [];

  for (let i = 0; i < data.data.length; i += 4) {
    let brightness = 1 - data.data[i] / 255;
    brightness = (brightness - 0.1307) / 0.3081;
    pixels.push(brightness);
  }

  return pixels;
};

const initModel = async () => {
  session = await ort.InferenceSession.create(MODEL_PATH);
};

const recognizeDigit = async (pixels) => {
  const tensor = new ort.Tensor(
    "float32",
    Float32Array.from(pixels),
    [1, 1, 28, 28]
  );
  const output = await session.run({ input: tensor });
  const result = output.output.data;
  return result.indexOf(Math.max(...result));
};

const clearCanvas = (ctx, canvas, resultZone) => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultZone.textContent = "";
};

document.addEventListener("DOMContentLoaded", async () => {
  const { canvas, ctx } = setupCanvas();
  const predictBtn = document.getElementById("predict");
  const clearBtn = document.getElementById("clear");
  const result = document.getElementById("result");

  await initModel();

  predictBtn.onclick = async () => {
    const input = extractInput(canvas);
    const digit = await recognizeDigit(input);
    result.textContent = `Prédiction : ${digit}`;
  };

  clearBtn.onclick = () => clearCanvas(ctx, canvas, result);
});
