#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base";
const DEFAULT_DEVICE = "cuda:0";
const DEFAULT_DTYPE = "bfloat16";
const DEFAULT_PYTHON = "python3";

function printMainHelp() {
  console.log(`qwen3tts - Node.js CLI wrapper for Qwen3-TTS voice clone

Usage:
  node scripts/qwen3tts.mjs <command> [options]

Commands:
  clone    Build reusable ICL clone prompt from reference audio
  speak    Generate WAV from text using saved clone file

Run command help:
  node scripts/qwen3tts.mjs clone --help
  node scripts/qwen3tts.mjs speak --help
`);
}

function printCloneHelp() {
  console.log(`Usage:
  node scripts/qwen3tts.mjs clone --ref-audio <path|url|base64> (--ref-text <text> | --ref-text-file <file>) --clone-file <file> [options]

Required:
  --ref-audio <value>
  --ref-text <value> | --ref-text-file <file>
  --clone-file <file>

Options:
  --model <id>               Default: ${DEFAULT_MODEL}
  --device <device_map>      Default: ${DEFAULT_DEVICE}
  --dtype <dtype>            One of: bfloat16|bf16|float16|fp16|float32|fp32 (default: ${DEFAULT_DTYPE})
  --flash-attn               Enable FlashAttention-2 (default)
  --no-flash-attn            Disable FlashAttention-2
  (local non-WAV ref audio is auto-converted via ffmpeg)
  --python <path>            Python executable (default: ${DEFAULT_PYTHON})
  --help
`);
}

function printSpeakHelp() {
  console.log(`Usage:
  node scripts/qwen3tts.mjs speak --clone-file <file> (--text <text> | --text-file <file>) --out <wav> [options]

Required:
  --clone-file <file>
  --text <value> | --text-file <file>
  --out <wav>

Options:
  --language <name>          Default: Auto
  --max-new-tokens <int>
  --top-k <int>
  --top-p <float>
  --temperature <float>
  --repetition-penalty <float>
  --model <id>               Default: ${DEFAULT_MODEL}
  --device <device_map>      Default: ${DEFAULT_DEVICE}
  --dtype <dtype>            One of: bfloat16|bf16|float16|fp16|float32|fp32 (default: ${DEFAULT_DTYPE})
  --flash-attn               Enable FlashAttention-2 (default)
  --no-flash-attn            Disable FlashAttention-2
  --python <path>            Python executable (default: ${DEFAULT_PYTHON})
  --help
`);
}

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}

function parseOptions(argv, spec, defaults) {
  const options = { ...defaults };
  const positional = [];

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      positional.push(token);
      continue;
    }

    if (token === "--help") {
      options.help = true;
      continue;
    }

    if (token.startsWith("--no-")) {
      const name = token.slice(5);
      if (!spec[name] || spec[name].type !== "boolean") {
        fail(`Unknown option: ${token}`);
      }
      options[name] = false;
      continue;
    }

    const eq = token.indexOf("=");
    let rawName = token.slice(2);
    let rawValue = null;
    if (eq >= 0) {
      rawName = token.slice(2, eq);
      rawValue = token.slice(eq + 1);
    }

    const def = spec[rawName];
    if (!def) {
      fail(`Unknown option: --${rawName}`);
    }

    if (def.type === "boolean") {
      options[rawName] = true;
      continue;
    }

    let value = rawValue;
    if (value === null) {
      i += 1;
      if (i >= argv.length) {
        fail(`Option --${rawName} requires a value`);
      }
      value = argv[i];
    }

    if (def.type === "int") {
      const parsed = Number.parseInt(value, 10);
      if (!Number.isFinite(parsed)) {
        fail(`Invalid integer for --${rawName}: ${value}`);
      }
      options[rawName] = parsed;
      continue;
    }

    if (def.type === "float") {
      const parsed = Number.parseFloat(value);
      if (!Number.isFinite(parsed)) {
        fail(`Invalid number for --${rawName}: ${value}`);
      }
      options[rawName] = parsed;
      continue;
    }

    options[rawName] = value;
  }

  return { options, positional };
}

function pushOption(args, name, value) {
  if (value === undefined || value === null) {
    return;
  }
  args.push(`--${name}`, String(value));
}

function ensureBridgePath() {
  const thisFile = fileURLToPath(import.meta.url);
  const dir = path.dirname(thisFile);
  const bridge = path.join(dir, "qwen3tts_bridge.py");
  if (!fs.existsSync(bridge)) {
    fail(`Bridge script not found: ${bridge}`);
  }
  return bridge;
}

function runPython(python, args) {
  const child = spawn(python, args, { stdio: "inherit" });
  child.on("error", (err) => {
    fail(`Failed to run Python: ${err.message}`);
  });
  child.on("exit", (code, signal) => {
    if (signal) {
      process.exit(1);
    }
    process.exit(code ?? 1);
  });
}

function buildCommonArgs(opts) {
  return [
    "--model",
    opts.model,
    "--device",
    opts.device,
    "--dtype",
    opts.dtype,
    opts.flashAttn ? "--flash-attn" : "--no-flash-attn",
  ];
}

function handleClone(argv) {
  const spec = {
    "ref-audio": { type: "string" },
    "ref-text": { type: "string" },
    "ref-text-file": { type: "string" },
    "clone-file": { type: "string" },
    model: { type: "string" },
    device: { type: "string" },
    dtype: { type: "string" },
    "flash-attn": { type: "boolean" },
    python: { type: "string" },
  };

  const defaults = {
    model: DEFAULT_MODEL,
    device: DEFAULT_DEVICE,
    dtype: DEFAULT_DTYPE,
    flashAttn: true,
    python: DEFAULT_PYTHON,
    help: false,
  };

  const { options, positional } = parseOptions(argv, spec, defaults);
  if (options["flash-attn"] !== undefined) {
    options.flashAttn = options["flash-attn"];
  }

  if (options.help) {
    printCloneHelp();
    process.exit(0);
  }
  if (positional.length > 0) {
    fail(`Unexpected positional arguments: ${positional.join(" ")}`);
  }

  if (!options["ref-audio"]) {
    fail("Missing required --ref-audio");
  }
  if (!options["clone-file"]) {
    fail("Missing required --clone-file");
  }

  const hasRefText = Boolean(options["ref-text"]);
  const hasRefTextFile = Boolean(options["ref-text-file"]);
  if (hasRefText === hasRefTextFile) {
    fail("Provide exactly one of --ref-text or --ref-text-file");
  }

  const bridge = ensureBridgePath();
  const pyArgs = [bridge, "clone", ...buildCommonArgs(options)];
  pushOption(pyArgs, "ref-audio", options["ref-audio"]);
  if (hasRefText) {
    pushOption(pyArgs, "ref-text", options["ref-text"]);
  } else {
    pushOption(pyArgs, "ref-text-file", options["ref-text-file"]);
  }
  pushOption(pyArgs, "clone-file", options["clone-file"]);
  runPython(options.python, pyArgs);
}

function handleSpeak(argv) {
  const spec = {
    "clone-file": { type: "string" },
    text: { type: "string" },
    "text-file": { type: "string" },
    out: { type: "string" },
    language: { type: "string" },
    "max-new-tokens": { type: "int" },
    "top-k": { type: "int" },
    "top-p": { type: "float" },
    temperature: { type: "float" },
    "repetition-penalty": { type: "float" },
    model: { type: "string" },
    device: { type: "string" },
    dtype: { type: "string" },
    "flash-attn": { type: "boolean" },
    python: { type: "string" },
  };

  const defaults = {
    language: "Auto",
    model: DEFAULT_MODEL,
    device: DEFAULT_DEVICE,
    dtype: DEFAULT_DTYPE,
    flashAttn: true,
    python: DEFAULT_PYTHON,
    help: false,
  };

  const { options, positional } = parseOptions(argv, spec, defaults);
  if (options["flash-attn"] !== undefined) {
    options.flashAttn = options["flash-attn"];
  }

  if (options.help) {
    printSpeakHelp();
    process.exit(0);
  }
  if (positional.length > 0) {
    fail(`Unexpected positional arguments: ${positional.join(" ")}`);
  }

  if (!options["clone-file"]) {
    fail("Missing required --clone-file");
  }
  if (!options.out) {
    fail("Missing required --out");
  }

  const hasText = Boolean(options.text);
  const hasTextFile = Boolean(options["text-file"]);
  if (hasText === hasTextFile) {
    fail("Provide exactly one of --text or --text-file");
  }

  const bridge = ensureBridgePath();
  const pyArgs = [bridge, "speak", ...buildCommonArgs(options)];
  pushOption(pyArgs, "clone-file", options["clone-file"]);
  if (hasText) {
    pushOption(pyArgs, "text", options.text);
  } else {
    pushOption(pyArgs, "text-file", options["text-file"]);
  }
  pushOption(pyArgs, "out", options.out);
  pushOption(pyArgs, "language", options.language);
  pushOption(pyArgs, "max-new-tokens", options["max-new-tokens"]);
  pushOption(pyArgs, "top-k", options["top-k"]);
  pushOption(pyArgs, "top-p", options["top-p"]);
  pushOption(pyArgs, "temperature", options.temperature);
  pushOption(pyArgs, "repetition-penalty", options["repetition-penalty"]);

  runPython(options.python, pyArgs);
}

function main() {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv[0] === "--help" || argv[0] === "-h") {
    printMainHelp();
    process.exit(0);
  }

  const [command, ...rest] = argv;
  if (command === "clone") {
    handleClone(rest);
    return;
  }
  if (command === "speak") {
    handleSpeak(rest);
    return;
  }

  fail(`Unknown command: ${command}`);
}

main();
