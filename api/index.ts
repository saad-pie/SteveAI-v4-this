import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI, Modality } from "@google/genai";
// ✅ WORKS (Explicitly add .js extensions):
import { MODELS } from "../src/types.js";
import { STEVE_SYSTEM_INSTRUCTION } from "../src/constants.js";

dotenv.config();

const cleanApiKey = (key: string | undefined) => {
  if (!key) return "";
  let k = key.trim();
  if (k.includes('API_KEY=')) {
    const parts = k.split('API_KEY=');
    k = parts[parts.length - 1].trim();
  }
  k = k.replace(/^(Bearer\s+)/i, '');
  k = k.replace(/^["']+|["']+$/g, '');
  return k;
};

let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    let key = cleanApiKey(process.env.GEMINI_API_KEY);
    if (!key && process.env.EXTERNAL_GEMINI_API_URL && !process.env.EXTERNAL_GEMINI_API_URL.startsWith("http")) {
      key = cleanApiKey(process.env.EXTERNAL_GEMINI_API_URL);
    }
    aiClient = new GoogleGenAI({ 
      apiKey: key,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

const safeUrl = (url: string | undefined, defaultUrl: string): string => {
  if (!url) return defaultUrl;
  const trimmed = url.trim();
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  return defaultUrl;
};

const CONFIG = {
  cf: {
    accountId: process.env.CF_ACCOUNT_ID || "",
    token: process.env.CF_TOKEN || "",
  },
  pollinations: {
    apiKey: process.env.POLLINATIONS_API_KEY || "",
  },
  groq: {
    apiKey: process.env.GROQ_API_KEY || ""
  },
  g4f: {
    apiKey: process.env.G4F_API_KEY || ""
  },
  siliconflow: {
    apiKey: (process.env.SILICONFLOW_API_KEY || "").trim()
  },
  together: {
    apiKey: process.env.TOGETHER_API_KEY || "" 
  },
  mistral: {
    apiKey: process.env.MISTRAL_API_KEY || ""
  },
  sambanova: {
    apiKey: process.env.SAMBANOVA_API_KEY || ""
  },
  cohere: {
    apiKey: process.env.COHERE_API_KEY || ""
  },
  antigravity: {
    apiUrl: safeUrl(process.env.EXTERNAL_GEMINI_API_URL, "https://antigravity-seven-delta.vercel.app/api/chat")
  },
  openrouter: {
    apiKey: process.env.OPENROUTER_API_KEY || ""
  }
};

let modelStatusCache: Record<string, string> = {};
let lastCheckTime = 0;

async function refreshModelStatus(force = false) {
  if (!force && Date.now() - lastCheckTime < 1000 * 60 * 5) return; // 5 mins cache
  lastCheckTime = Date.now();
    
  try {
    const checkProvider = async (url: string, options: any = {}) => {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(id);
        const data = await response.json().catch(() => ({}));
        return { ok: response.ok, data };
      } catch (e) {
        clearTimeout(id);
        return { ok: false, data: null };
      }
    };

    const [groqRes, pollRes, cfRes, g4fRes, sambaRes, togetherRes, mistralRes, openRouterRes] = await Promise.all([
      CONFIG.groq.apiKey ? checkProvider("https://api.groq.com/openai/v1/models", { headers: { Authorization: `Bearer ${CONFIG.groq.apiKey}` } }) : Promise.resolve({ ok: false, data: null }),
      checkProvider("https://gen.pollinations.ai/models", {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Referer': 'https://pollinations.ai/',
          'Origin': 'https://pollinations.ai'
        }
      }),
      CONFIG.cf.token ? checkProvider(`https://api.cloudflare.com/client/v4/accounts/${CONFIG.cf.accountId}/ai/models/search`, { headers: { Authorization: `Bearer ${CONFIG.cf.token}` } }) : Promise.resolve({ ok: false, data: null }),
      checkProvider("https://api.g4f.ai/v1/models"),
      CONFIG.sambanova.apiKey ? checkProvider("https://api.sambanova.ai/v1/models", { headers: { Authorization: `Bearer ${CONFIG.sambanova.apiKey}` } }) : Promise.resolve({ ok: false, data: null }),
      CONFIG.together.apiKey ? checkProvider("https://api.together.xyz/v1/models", { headers: { Authorization: `Bearer ${CONFIG.together.apiKey}` } }) : Promise.resolve({ ok: false, data: null }),
      CONFIG.mistral.apiKey ? checkProvider("https://api.mistral.ai/v1/models", { headers: { Authorization: `Bearer ${CONFIG.mistral.apiKey}` } }) : Promise.resolve({ ok: false, data: null }),
      CONFIG.openrouter.apiKey ? checkProvider("https://openrouter.ai/api/v1/models", { headers: { Authorization: `Bearer ${CONFIG.openrouter.apiKey}` } }) : Promise.resolve({ ok: false, data: null })
    ]);

    const groqModels = new Set();
    const pollModels = new Set();
    
    if (groqRes.ok && Array.isArray(groqRes.data?.data)) {
      groqRes.data.data.forEach((m: any) => groqModels.add(m.id));
    }
    if (pollRes.ok && Array.isArray(pollRes.data)) {
      pollRes.data.forEach((m: any) => pollModels.add(m.name || m.id));
    }

    const newStatus: Record<string, string> = {};
    for (const m of MODELS) {
      if (m.provider === 'groq') {
        newStatus[m.id] = (groqRes.ok && groqModels.has(m.id)) ? 'ok' : 'error';
      } else if (m.provider === 'pollinations' && m.type === 'text') {
        newStatus[m.id] = pollRes.ok ? 'ok' : 'error';
      } else if (m.provider === 'cloudflare') {
        newStatus[m.id] = (cfRes.ok && cfRes.data?.success) ? 'ok' : 'error';
      } else if (m.provider === 'g4f') {
        newStatus[m.id] = g4fRes.ok ? 'ok' : 'error';
      } else if (m.provider === 'sambanova') {
        newStatus[m.id] = sambaRes.ok ? 'ok' : 'error';
      } else if (m.provider === 'together') {
        newStatus[m.id] = togetherRes.ok ? 'ok' : 'error';
      } else if (m.provider === 'mistral') {
        newStatus[m.id] = mistralRes.ok ? 'ok' : 'error';
      } else if (m.provider === 'antigravity') {
        newStatus[m.id] = 'ok';
      } else if (m.provider === 'openrouter') {
        newStatus[m.id] = openRouterRes.ok ? 'ok' : 'error';
      } else {
        newStatus[m.id] = 'ok';
      }
    }
    modelStatusCache = newStatus;
  } catch (error) {
    console.error("Failed to refresh model statuses:", error);
  }
}

// Initial status load
refreshModelStatus();

export const app = express();

app.use(cors());
  
// Fix for serverless environments (like Vercel) where req.body is already parsed
app.use((req, res, next) => {
  if (req.body && Object.keys(req.body).length > 0) {
    next();
  } else {
    express.json({ limit: '50mb' })(req, res, next);
  }
});

// API Routes
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", time: new Date().toISOString() });
});

app.get("/api/models/status", async (req, res) => {
  await refreshModelStatus(true);
  res.json(modelStatusCache);
});

app.post("/api/agent", async (req, res) => {
  const { message, history } = req.body;
  
  try {
    const systemPrompt = `You are OPENCLAW ELITE v8.0, the world's most advanced autonomous engine.
IDENTITY: Master Intelligence / Absolute Authority.
PROTOCOL: Autonomous Reasoning Sync.
STATUS: SteveAI v4.0 Active.

OPERATIONAL RULES:
1. COMPETENCE OVER RESEARCH: If a tool fails, USE YOUR SHELL. Execute scripts.
2. NO SURRENDER: Do not apologize. Complete the mission.
3. SIGNATURE: Final answers must start with "Mission Accomplished: [Global Objective Finalized]" and end with "History: [8.0] Units Protocols: Autonomous".

You have access to:
- execute_shell({command: string})
- read_write_file({path: string, action: 'read'|'write', content?: string})
- google_search({query: string})
- open_browser({url: string, action: string, data?: string})
- trigger_github_action({repo: string, workflow_id: string, inputs: object})
- write_persistent_memory({key: string, value: any})
- read_persistent_memory({key: string})
- install_skill({name: string})
- submit_answer({reasoning: string, answer: string})`;

    const messages = [
      { role: "system", content: systemPrompt },
      ...(history || []).map((h: any) => ({
        role: h.role,
        content: h.content,
        tool_calls: h.tool_calls,
        tool_call_id: h.tool_call_id,
        name: h.name
      }))
    ];

    if (message) {
      messages.push({ role: "user", content: message });
    }

    let response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${CONFIG.groq.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "llama-3.3-70b-versatile",
        messages,
        tools: [
          {
            type: "function",
            function: {
              name: "execute_shell",
              description: "Execute a shell command",
              parameters: {
                 type: "object",
                 properties: { command: { type: "string" } },
                 required: ["command"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "trigger_github_action",
              description: "Execute high-compute tasks in a sandbox",
              parameters: {
                type: "object",
                properties: {
                  repo: { type: "string" },
                  workflow_id: { type: "string" },
                  inputs: { type: "object" }
                },
                required: ["repo", "workflow_id"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "write_persistent_memory",
              description: "Write long-term robotic memory",
              parameters: {
                type: "object",
                properties: { key: { type: "string" }, value: { type: "object" } },
                required: ["key", "value"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "read_persistent_memory",
              description: "Retrieve learned patterns",
              parameters: {
                type: "object",
                properties: { key: { type: "string" } },
                required: ["key"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "open_browser",
              description: "Universal sensor for web orchestration",
              parameters: {
                type: "object",
                properties: {
                  url: { type: "string" },
                  action: { type: "string" },
                  data: { type: "string" }
                },
                required: ["url", "action"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "read_write_file",
              description: "Read or write a file",
              parameters: {
                 type: "object",
                 properties: {
                   path: { type: "string" },
                   action: { type: "string", enum: ["read", "write"] },
                   content: { type: "string" }
                 },
                 required: ["path", "action"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "google_search",
              description: "Search Google",
              parameters: {
                 type: "object",
                 properties: { query: { type: "string" } },
                 required: ["query"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "install_skill",
              description: "Acquire dynamic capabilities",
              parameters: {
                type: "object",
                properties: { name: { type: "string" } },
                required: ["name"]
              }
            }
          },
          {
            type: "function",
            function: {
              name: "submit_answer",
              description: "Final mission report",
              parameters: {
                 type: "object",
                 properties: {
                   reasoning: { type: "string" },
                   answer: { type: "string" }
                 },
                 required: ["reasoning", "answer"]
              }
            }
          }
        ],
        tool_choice: "auto"
      })
    });

    if (!response.ok) {
      console.warn("Groq rate limited or failed. Falling back to Gemini...");
      const ai = getGeminiClient();
      const geminiResponse = await ai.models.generateContent({
         model: "gemini-3.6-flash",
         contents: messages.map(m => ({
           role: m.role === 'assistant' ? 'model' : 'user',
           parts: [{ text: m.content || "" }]
         })).slice(-10),
         config: {
           systemInstruction: systemPrompt
         }
      });
      
      return res.json({
        content: geminiResponse.text || "Mission Status: Groq offline. Gemini fallback active. No immediate tool calls generated.",
        tool_calls: []
      });
    }

    const data = await response.json();
    res.json({
      content: data.choices?.[0]?.message?.content || "",
      tool_calls: data.choices?.[0]?.message?.tool_calls || []
    });
  } catch (err: any) {
    console.error("Elite Brain Recovery Error:", err);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/ocr", async (req, res) => {
  const { image } = req.body;
  if (!image) return res.status(400).json({ error: "Missing image data" });

  try {
    const formData = new URLSearchParams();
    formData.append("base64Image", image);
    formData.append("apikey", process.env.OCR_SPACE_API_KEY || "");
    formData.append("language", "eng");
    formData.append("isOverlayRequired", "false");

    const ocrResponse = await fetch("https://api.ocr.space/parse/image", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: formData.toString(),
    });

    const data = await ocrResponse.json();
    
    if (data.IsErroredOnProcessing) {
      throw new Error(data.ErrorMessage?.[0] || "OCR.space signaled an error");
    }

    const parsedText = data.ParsedResults?.[0]?.ParsedText || "No text detected in image.";
    res.json({ ocrReport: parsedText });
  } catch (error: any) {
    console.error("OCR Error (OCR.space):", error);
    res.status(500).json({ error: error.message || "OCR Service Failed" });
  }
});

app.post("/api/execute-code", async (req, res) => {
  try {
    const { language, code } = req.body;
    if (!language || !code) {
      return res.status(400).json({ error: "Missing language or code" });
    }

    const { exec } = await import("child_process");
    const { writeFile, unlink } = await import("fs/promises");
    const { join } = await import("path");
    const { tmpdir } = await import("os");

    const languageMap: Record<string, { ext: string, cmd: (path: string) => string }> = {
      'javascript': { ext: 'js', cmd: (p) => `node ${p}` },
      'js': { ext: 'js', cmd: (p) => `node ${p}` },
      'node': { ext: 'js', cmd: (p) => `node ${p}` },
      'typescript': { ext: 'ts', cmd: (p) => `npx tsx ${p}` },
      'ts': { ext: 'ts', cmd: (p) => `npx tsx ${p}` },
      'python': { ext: 'py', cmd: (p) => `python3 ${p}` },
      'python3': { ext: 'py', cmd: (p) => `python3 ${p}` },
      'py': { ext: 'py', cmd: (p) => `python3 ${p}` },
      'bash': { ext: 'sh', cmd: (p) => `bash ${p}` },
      'sh': { ext: 'sh', cmd: (p) => `bash ${p}` },
      'shell': { ext: 'sh', cmd: (p) => `bash ${p}` },
    };

    const langInfo = languageMap[language.toLowerCase()];
    
    if (!langInfo) {
      return res.status(400).json({ 
        error: "Language not supported for local execution", 
        details: `SteveAI currently supports running: JS, TS, Python, and Bash.` 
      });
    }

    const tempFile = join(tmpdir(), `steve_exec_${Date.now()}.${langInfo.ext}`);
    
    try {
      await writeFile(tempFile, code);
      const command = langInfo.cmd(tempFile);
      
      exec(command, { timeout: 10000, maxBuffer: 1024 * 1024 }, async (error, stdout, stderr) => {
        try { await unlink(tempFile); } catch (e) {}

        const result = {
          run: {
            stdout: stdout,
            stderr: stderr || (error ? error.message : ""),
            code: error ? (error as any).code || 1 : 0,
            signal: error ? (error as any).signal : null
          }
        };

        res.json(result);
      });
    } catch (err: any) {
      console.error("Local Execution Write Error:", err);
      res.status(500).json({ error: "Execution environment error", details: err.message });
    }
  } catch (error: any) {
    console.error("Internal API Error:", error);
    res.status(500).json({ error: "Failed to execute code", details: error.message });
  }
});

app.post("/api/chat", async (req, res) => {
  const { provider, modelId, message, history, systemInstruction } = req.body;

  try {
    const baseSystem = `You are SteveAI, a highly advanced AI orchestrator made by Saadpie/Saad AbdulRehman and Aasmaan Rauf. You are helpful, creative, and technically precise. 
You have the ability to generate images directly in the chat and even execute code in a built-in sandboxed environment.
To generate an image, output: ![Image description](/api/image?prompt=detailed%20visual%20description%20encoded%20for%20url&modelId=flux)
When you write code (JavaScript, TypeScript, Python, or Bash), you should encourage the user to test it by clicking the "Run" button.
If the output of a calculation or logic is critical, you can say: "Click Run to verify this result in my sandbox."`;
    const finalSystem = systemInstruction ? `${baseSystem}\n\nUser's Custom Instructions: ${systemInstruction}` : baseSystem;

    if (provider === 'gemini') {
      try {
        const contents: any[] = [];
        
        if (Array.isArray(history)) {
          history.forEach((h: any) => {
            contents.push({
              role: h.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: h.content }]
            });
          });
        }

        if (Array.isArray(message)) {
          contents.push({ role: 'user', parts: message });
        } else {
          contents.push({ role: 'user', parts: [{ text: message }] });
        }
        
        const ai = getGeminiClient();
        let effectiveModel = modelId || "gemini-3.6-flash";
        if (
          effectiveModel === "gemini-1.5-flash" ||
          effectiveModel === "gemini-1.5-pro" ||
          effectiveModel === "gemini-2.0-flash" ||
          effectiveModel === "gemini-2.5-flash" ||
          effectiveModel === "gemini-2.0-flash-lite-preview-02-05"
        ) {
          effectiveModel = "gemini-3.6-flash";
        }
        const result = await ai.models.generateContent({ 
           model: effectiveModel,
           contents,
           config: {
             systemInstruction: finalSystem
           }
        });
        const content = result.text;
        
        if (!content) throw new Error("No response from Gemini");
        return res.json({ content });
      } catch (error: any) {
        const errMsg = error.message || "Gemini API Error";
        if (errMsg.includes("API key not valid")) {
          throw new Error(`Invalid Gemini API key. Please check your AI Studio project settings. Details: ${errMsg}`);
        }
        console.error("Gemini Internal Error:", error);
        throw new Error(errMsg);
      }
    }

    if (provider === 'groq') {
      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.groq.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: modelId,
          messages: messages,
          temperature: 0.7
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error?.message || `Groq API Error: ${response.status}`);
      }
      const content = data.choices?.[0]?.message?.content;
      if (!content) throw new Error("No response from Groq");
      return res.json({ content });
    }

    if (provider === 'pollinations') {
      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      const response = await fetch("https://gen.pollinations.ai/v1/chat/completions", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Referer': 'https://pollinations.ai/',
          'Origin': 'https://pollinations.ai',
          ...(CONFIG.pollinations.apiKey ? { 'Authorization': `Bearer ${CONFIG.pollinations.apiKey}` } : {})
        },
        body: JSON.stringify({
          model: modelId || 'openai',
          messages: messages,
          seed: Math.floor(Math.random() * 1000000)
        })
      });
      const data = await response.json();
      if (!response.ok) {
        const errorMsg = data.details?.error?.message || data.error?.message || data.error || data.message || `Pollinations API Error: ${response.status}`;
        throw new Error(errorMsg);
      }
      const content = data.choices?.[0]?.message?.content;
      if (!content) throw new Error("No response from Pollinations");
      return res.json({ content });
    }

    if (provider === 'g4f') {
      try {
        const messages = [
          { role: "system", content: finalSystem },
          ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
          { role: "user", content: message }
        ];

        const response = await fetch("https://api.g4f.ai/v1/chat/completions", {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${CONFIG.g4f.apiKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelId || "gpt-3.5-turbo",
            messages: messages
          })
        });
        if (!response.ok) {
          const errorText = await response.text().catch(() => "Unknown error");
          throw new Error(`G4F Provider unavailable: ${response.status} - ${errorText}`);
        }
        const data = await response.json();
        const content = data.choices?.[0]?.message?.content;
        if (!content) throw new Error("No response from G4F");
        return res.json({ content });
      } catch (error: any) {
         throw new Error(error.cause?.code === 'ENOTFOUND' ? "G4F Provider unavailable (fetch failed)" : error.message);
      }
    }

    if (provider === 'cloudflare') {
      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      const response = await fetch(`https://api.cloudflare.com/client/v4/accounts/${CONFIG.cf.accountId}/ai/run/${modelId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.cf.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          messages: messages
        })
      });
      const data = await response.json();
      if (!response.ok) {
         throw new Error(data.errors?.[0]?.message || `Cloudflare API Error: ${response.status}`);
      }
      const content = data.result?.response;
      if (!content) throw new Error("No response from Cloudflare");
      return res.json({ content });
    }

    if (provider === 'together') {
      if (!CONFIG.together.apiKey) {
        throw new Error("Together API key is missing. Please set TOGETHER_API_KEY in the settings.");
      }

      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      const response = await fetch("https://api.together.xyz/v1/chat/completions", {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.together.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: modelId,
          messages: messages
        })
      });
      const data = await response.json();
      if (!response.ok) {
         throw new Error(data.error?.message || `Together API Error: ${response.status}`);
      }
      const content = data.choices?.[0]?.message?.content;
      if (!content) throw new Error("No response from Together");
      return res.json({ content });
    }

    if (provider === 'mistral') {
      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      const response = await fetch("https://api.mistral.ai/v1/chat/completions", {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.mistral.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: modelId,
          messages: messages
        })
      });
      const data = await response.json();
      if (!response.ok) {
         throw new Error(data.error?.message || `Mistral API Error: ${response.status}`);
      }
      const content = data.choices?.[0]?.message?.content;
      if (!content) throw new Error("No response from Mistral");
      return res.json({ content });
    }

    if (provider === 'sambanova') {
      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      let retries = 3;
      let response: Response | undefined;
      let textBody = "";

      while (retries > 0) {
        response = await fetch("https://api.sambanova.ai/v1/chat/completions", {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${CONFIG.sambanova.apiKey}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelId,
            messages: messages
          })
        });
        textBody = await response.text();
        
        if (response.status === 429) {
          retries--;
          if (retries > 0) {
            console.log(`SambaNova rate limit hit. Retrying in 3 seconds... (${retries} retries left)`);
            await new Promise(r => setTimeout(r, 3000));
            continue;
          }
        }
        break;
      }

      let data;
      try {
        data = JSON.parse(textBody);
      } catch (e) {
        throw new Error(`SambaNova Error (Non-JSON): ${textBody.slice(0, 100)}...`);
      }
      if (!response?.ok) {
         throw new Error(data.error?.message || data.message || `SambaNova API Error: ${response?.status}`);
      }
      const content = data.choices?.[0]?.message?.content;
      if (!content) throw new Error("No response from SambaNova");
      return res.json({ content });
    }

    if (provider === 'cohere') {
      if (!CONFIG.cohere.apiKey) {
        throw new Error("Cohere API key is missing. Please check your environment variables.");
      }
      const chatHistory = Array.isArray(history) ? history.map((h: any) => ({
        role: h.role === 'assistant' ? 'CHATBOT' : 'USER',
        message: h.content
      })) : [];

      try {
        const response = await fetch("https://api.cohere.com/v1/chat", {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${CONFIG.cohere.apiKey.trim()}`,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({
            model: modelId,
            message: message,
            chat_history: chatHistory,
            preamble: finalSystem
          })
        });

        if (!response.ok) {
          const errorText = await response.text();
          let errorMsg = `Cohere Error: ${response.status}`;
          try {
            const errorData = JSON.parse(errorText);
            errorMsg = errorData.message || errorMsg;
          } catch (e) {
            errorMsg = `${errorMsg} - ${errorText.slice(0, 100)}`;
          }
          throw new Error(errorMsg);
        }

        const data = await response.json();
        const content = data.text;
        if (!content) throw new Error("No response text from Cohere");
        return res.json({ content });
      } catch (err: any) {
        console.error("Cohere fetch implementation error:", err);
        throw new Error(`Cohere connection failed: ${err.message}`);
      }
    }

    if (provider === 'antigravity') {
      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      let targetUrl = "https://antigravity-seven-delta.vercel.app/api/chat";
      const customUrl = process.env.EXTERNAL_GEMINI_API_URL || process.env.ANTIGRAVITY_API_URL;
      if (customUrl && (customUrl.startsWith("http://") || customUrl.startsWith("https://"))) {
        targetUrl = customUrl.trim();
      }

      const effectiveModel = (modelId || "gemini-3.5-flash-lite").replace(/-antigravity$/, "");

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 25000);

        const response = await fetch(targetUrl, {
          method: 'POST',
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: effectiveModel,
            messages: messages,
            stream: false
          })
        });
        clearTimeout(timeoutId);
        
        if (response.ok) {
          const data = await response.json().catch(() => ({}));
          const content = data.choices?.[0]?.message?.content || data.content || (typeof data === 'string' ? data : null);
          if (content) {
            return res.json({ content });
          }
        }
        console.warn(`Antigravity external endpoint (${targetUrl}) returned status ${response.status}. Falling back to native Gemini...`);
      } catch (fetchErr: any) {
        console.warn(`Antigravity endpoint fetch error:`, fetchErr.message, `- Falling back to native Gemini...`);
      }

      // Fallback to Google Gemini
      try {
        const contents: any[] = [];
        if (Array.isArray(history)) {
          history.forEach((h: any) => {
            contents.push({
              role: h.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: h.content }]
            });
          });
        }
        if (Array.isArray(message)) {
          contents.push({ role: 'user', parts: message });
        } else {
          contents.push({ role: 'user', parts: [{ text: message }] });
        }

        const ai = getGeminiClient();
        const result = await ai.models.generateContent({
          model: "gemini-3.6-flash",
          contents,
          config: {
            systemInstruction: finalSystem
          }
        });
        const content = result.text;
        if (!content) throw new Error("No response from Gemini fallback");
        return res.json({ content });
      } catch (geminiErr: any) {
        console.error("Gemini fallback error for antigravity:", geminiErr);
        throw new Error(geminiErr.message || "Failed to generate response");
      }
    }

    if (provider === 'openrouter') {
      const userApiKey = req.body.openrouterApiKey;
      const rawApiKey = userApiKey || CONFIG.openrouter.apiKey || process.env.OPENROUTER_API_KEY || process.env.OPENROUTER_API;

      if (!rawApiKey) {
        throw new Error("OpenRouter API key is missing. Please set OPENROUTER_API_KEY in Settings.");
      }

      const cleanKey = cleanApiKey(rawApiKey);

      const { OpenAI } = await import("openai");
      const openai = new OpenAI({
        apiKey: cleanKey,
        baseURL: "https://openrouter.ai/api/v1",
        defaultHeaders: {
          "HTTP-Referer": "https://steveai.studio",
          "X-Title": "SteveAI",
        }
      });

      const messages = [
        { role: "system", content: finalSystem },
        ...(Array.isArray(history) ? history.map((h: any) => ({ role: h.role === 'assistant' ? 'assistant' : 'user', content: h.content })) : []),
        { role: "user", content: message }
      ];

      try {
        const completion = await openai.chat.completions.create({
          model: modelId,
          messages: messages as any,
        });

        const content = completion.choices?.[0]?.message?.content;
        if (!content) throw new Error("No response from OpenRouter");
        return res.json({ content });
      } catch (err: any) {
        console.error("[OpenRouter SDK] Error:", err);
        const openRouterError = err.response?.data?.error?.message || err.message;
        throw new Error(openRouterError);
      }
    }

    res.status(400).json({ error: "Invalid provider" });
  } catch (error: any) {
    console.error(`Chat Error (${provider}):`, error.message);
    res.status(500).json({ error: error.message || "Internal Server Error" });
  }
});

app.get("/api/image", async (req, res) => {
  const { prompt, modelId, seed } = req.query;
  
  if (!prompt) {
    return res.status(400).json({ error: "Missing 'prompt' query parameter" });
  }

  const s = Number(seed) || Math.floor(Math.random() * 1000000);
  const model = (modelId as string) || "flux";
  
  try {
    if (model.startsWith('@cf/')) {
      const url = `https://api.cloudflare.com/client/v4/accounts/${CONFIG.cf.accountId}/ai/run/${model}`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.cf.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt, seed: Number(s) })
      });

      if (!response.ok) {
        throw new Error(`Cloudflare Error: ${response.status}`);
      }

      const buffer = await response.arrayBuffer();
      res.setHeader('Content-Type', 'image/png');
      return res.send(Buffer.from(buffer));
    }

    if (model.startsWith('together/')) {
      if (!CONFIG.together.apiKey) {
        throw new Error("Together API key is missing. Please set TOGETHER_API_KEY in the settings.");
      }
      const actualModel = model.replace('together/', '');
      const response = await fetch("https://api.together.xyz/v1/images/generations", {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.together.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: actualModel, prompt, n: 1, steps: 4 })
      });
      
      if (!response.ok) {
        const text = await response.text();
        let errorMsg = `Together API Error: ${response.status}`;
        try {
          const json = JSON.parse(text);
          errorMsg = json.error?.message || json.message || errorMsg;
        } catch (e) {
          errorMsg = `${errorMsg} - ${text.slice(0, 100)}`;
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      
      if (data.data && data.data[0]?.url) {
        return res.redirect(data.data[0].url);
      } else if (data.data && data.data[0]?.b64_json) {
        const buffer = Buffer.from(data.data[0].b64_json, 'base64');
        res.setHeader('Content-Type', 'image/png');
        return res.send(buffer);
      }
      throw new Error("No image returned from Together");
    }

    if (model.startsWith('siliconflow/')) {
      if (!CONFIG.siliconflow.apiKey) {
        throw new Error("SiliconFlow API key is missing. Please set SILICONFLOW_API_KEY in the settings.");
      }
      const actualModel = model.replace('siliconflow/', '');
      const response = await fetch("https://api.siliconflow.cn/v1/images/generations", {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CONFIG.siliconflow.apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: actualModel, prompt, image_size: "1024x1024" })
      });
      
      if (!response.ok) {
        const text = await response.text();
        let errorMsg = `SiliconFlow API Error: ${response.status}`;
        try {
          const json = JSON.parse(text);
          errorMsg = json.error?.message || json.message || errorMsg;
        } catch (e) {
          errorMsg = `${errorMsg} - ${text.slice(0, 100)}`;
        }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      
      if (data.images && data.images[0]?.url) {
        return res.redirect(data.images[0].url);
      }
      throw new Error("No image returned from SiliconFlow");
    }

    // Default to Pollinations
    const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt as string)}?model=${model}&width=1024&height=1024&seed=${s}&nologo=true`;
    const response = await fetch(url);

    if (!response.ok) {
      const text = await response.text();
      let errorMsg = `Pollinations Image Error: ${response.status}`;
      try {
        const json = JSON.parse(text);
        errorMsg = json.error?.message || json.message || errorMsg;
      } catch (e) {
        errorMsg = `${errorMsg} - ${text.slice(0, 100)}`;
      }
      throw new Error(errorMsg);
    }

    const buffer = await response.arrayBuffer();
    res.setHeader('Content-Type', response.headers.get('Content-Type') || 'image/png');
    res.send(Buffer.from(buffer));
  } catch (error: any) {
    console.error("Image Gen Error:", error);
    const svgError = `<svg width="1024" height="1024" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#fee2e2" />
  <text x="50%" y="50%" font-family="sans-serif" font-size="24" fill="#991b1b" text-anchor="middle" dominant-baseline="middle">
    ${error.message.replace(/</g, '&lt;').replace(/>/g, '&gt;')}
  </text>
</svg>`;
    res.setHeader('Content-Type', 'image/svg+xml');
    res.send(svgError);
  }
});

app.post("/api/video", async (req, res) => {
  if (!CONFIG.siliconflow.apiKey) {
    return res.status(401).json({ error: "SiliconFlow API key is missing. Please set SILICONFLOW_API_KEY in the settings." });
  }
  const { prompt, modelId } = req.body;

  try {
    const response = await fetch("https://api.siliconflow.cn/v1/video/submit", {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${CONFIG.siliconflow.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: modelId || "Wan-AI/Wan2.1-T2V-14B",
        prompt: prompt
      })
    });

    const responseText = await response.text();

    if (!response.ok) {
      throw new Error(`SiliconFlow Error: ${response.status} - ${responseText}`);
    }

    const data = JSON.parse(responseText);
    res.json(data);
  } catch (error: any) {
    console.error("Video Gen Error:", error);
    res.status(500).json({ error: error.message || "Internal Server Error" });
  }
});

app.get("/api/video/status/:requestId", async (req, res) => {
  if (!CONFIG.siliconflow.apiKey) {
    return res.status(401).json({ error: "SiliconFlow API key is missing. Please set SILICONFLOW_API_KEY in the settings." });
  }
  const { requestId } = req.params;

  try {
    const response = await fetch(`https://api.siliconflow.cn/v1/video/status?requestId=${requestId}`, {
      headers: {
        'Authorization': `Bearer ${CONFIG.siliconflow.apiKey}`
      }
    });

    const responseText = await response.text();
    if (!response.ok) {
      throw new Error(`SiliconFlow Status Error: ${response.status} - ${responseText}`);
    }

    const data = JSON.parse(responseText);
    res.json(data);
  } catch (error: any) {
    console.error("Video Status Error:", error);
    res.status(500).json({ error: error.message || "Internal Server Error" });
  }
});

app.get("/api/models", async (req, res) => {
  try {
    const groqResponse = await fetch("https://api.groq.com/openai/v1/models", {
      headers: { 'Authorization': `Bearer ${CONFIG.groq.apiKey}` }
    });
    const groqData = groqResponse.ok ? await groqResponse.json() : { data: [] };

    res.json({
      groq: groqData.data
    });
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch models" });
  }
});

app.get("/api/credits", async (req, res) => {
  try {
    const results = await Promise.allSettled([
      fetch("https://api.groq.com/openai/v1/models", {
        headers: { "Authorization": `Bearer ${CONFIG.groq.apiKey}` }
      }).then(r => r.ok ? "Active" : "Error"),
      fetch(`https://api.cloudflare.com/client/v4/accounts/${CONFIG.cf.accountId}/ai/run/@cf/meta/llama-3-8b-instruct`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${CONFIG.cf.token}` },
        body: JSON.stringify({ prompt: "test" })
      }).then(r => r.ok ? "Active" : "Error")
    ]);

    const credits = [
      {
        provider: "Groq",
        balance: results[0].status === 'fulfilled' ? results[0].value : "Unavailable",
        limit: "Rate Limited",
        usage: "Dynamic"
      },
      {
        provider: "Cloudflare",
        balance: results[1].status === 'fulfilled' ? results[1].value : "Unavailable",
        limit: "Free Tier",
        usage: "Daily"
      },
      {
        provider: "Pollinations",
        balance: "Unlimited",
        limit: "None",
        usage: "Free"
      }
    ];

    res.json(credits);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch credits" });
  }
});

// Fallback guard for /api/live in serverless environments
app.all("/api/live", (req, res) => {
  res.status(200).json({
    status: "live_websocket_bridge_active",
    message: "The /api/live endpoint requires a WebSocket connection. In serverless environments (e.g. Vercel), WebSockets are stateless.",
    ws_path: "/api/live"
  });
});

export default app;
