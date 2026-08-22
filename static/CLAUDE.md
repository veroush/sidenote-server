# Sidenote — Session Handoff Prompt
> Paste this entire document at the start of a new Claude session. The AI will have complete context on the project, what's been built, what's currently working, and the exact state everything was left in — and can continue immediately without asking clarifying questions. At the end of every session, ask for an addendum to append to this document, following the same format.

---

## 0. WHO I AM & WHAT I'M DOING RIGHT NOW

I'm a software engineering student. I built this tool for myself — it listens to conversations I'm in and quietly surfaces my own notes on whatever topic comes up, so I have accurate specifics on hand instead of relying on memory alone, especially when I'm talking fast or switching between Dutch and English. I work in **PowerShell on Windows**. I need exact, copy-pasteable commands, and I want errors/output explained plainly before or after I run something — I am not a backend/infra expert, so don't assume I know what a given error means without it being spelled out. I debug interactively, one step at a time — don't batch five untested changes together and hope; give me one thing to try, then wait for what I actually see before the next step.

---

## 1. WHAT THE APP IS

**Sidenote** — a personal tool with two parts:
1. **Live listening**: it records short chunks of whatever's being said out loud (me or someone else in conversation) and transcribes them.
2. **Note matching**: I maintain a list of "topics," each with trigger keywords and my own written notes. When the transcribed speech matches a topic's keywords, it finds the single sentence in my notes that best answers whatever was actually asked (not the whole note block), and displays it on screen.

It supports **English and Dutch**, toggled manually (not auto-detected — see Section 4 for why).

---

## 2. HOW WE GOT TO THE CURRENT ARCHITECTURE (in order — don't repeat these dead ends)

1. **v1 — browser's built-in speech engine.** Used the Web Speech API (`webkitSpeechRecognition`), ran as a single HTML file. Worked for basic testing but: poor Dutch recognition, poor handling of fast speech, and quality varied unpredictably by device/browser.
2. **v2 — added AI-powered short-answer extraction.** Used Claude's built-in "call the Anthropic API with no key needed" artifact feature to turn a matched note into a concise answer. **This only works inside Claude's own chat preview** — once deployed to GitHub Pages, it failed with `Failed to fetch` (no key, no CORS, no proxy outside Claude's sandbox). Replaced with a fully offline local heuristic instead (tokenize the question, strip stopwords in both English and Dutch, score each sentence in the notes by keyword overlap, return the best-matching sentence, or the full note if nothing scores well). **This offline heuristic is still what's used today.**
3. **v3 — deployed to GitHub Pages** (`veroush/memory_help` repo, renamed the file to `index.html`, set Pages source to root). This surfaced a real bug: the app had been using `window.storage`, an API that **only exists inside Claude's own artifact preview** — meaning saved topics were silently not persisting anywhere else. **Fixed by switching to real `localStorage`.** ⚠️ Known remaining limitation from this fix — see Section 7, item 1.
4. **v4 — tried in-browser Whisper** (via `transformers.js`, model `Xenova/whisper-base` then `Xenova/whisper-tiny`, attempted WebGPU acceleration with CPU/WASM fallback) to fix the Dutch/fast-speech accuracy problem without needing a server. **Worked, but too slow** — `base` took ~30 seconds per chunk on the laptop's CPU via WebAssembly, `tiny` was faster but noticeably less accurate and prone to hallucinating text on quiet/ambiguous audio ("coughing", random unrelated phrases). Abandoned this path in favor of a real server.
5. **v5 — current architecture: a local Python server on the laptop.** See Section 3.

---

## 3. CURRENT ARCHITECTURE (as of end of this session)

```
sidenote-server/
├── server.py            ← FastAPI app: /transcribe endpoint (faster-whisper) + serves static/
├── requirements.txt      ← fastapi, uvicorn[standard], python-multipart, faster-whisper
└── static/
    └── index.html        ← the frontend (topics UI, listening control, language toggle, matching logic)
```

**Two things must be running simultaneously, in two separate terminal windows, every time this is used:**

```powershell
# Terminal 1 — from inside sidenote-server/
uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 2 — anywhere
tailscale serve 8000
```

- `uvicorn` runs the actual backend + serves the frontend, on port 8000, bound to all network interfaces (`0.0.0.0` — this must be literally those four zeros, never a specific IP address, real or redacted).
- `tailscale serve 8000` exposes that port over **real HTTPS** to devices on my personal Tailscale network (tailnet) only — not the public internet. This step is **required**, not optional: browsers block microphone access (`getUserMedia`) on any non-HTTPS origin except `localhost`, so a plain `http://100.x.x.x:8000` Tailscale address is not enough on its own — the mic permission silently fails and the Start button stays disabled.
- Confirmed working: opening the resulting `https://<device-name>.<tailnet-name>.ts.net` link on my phone loads the app and the mic prompt works, from anywhere (not just shared wifi), since Tailscale itself provides the tunnel.

**Requires (one-time setup, already done):**
- Tailscale installed and logged into the same account on both laptop and phone.
- **HTTPS Certificates** enabled at https://login.tailscale.com/admin/dns (separate from MagicDNS — both were checked this session).

### 3.1 — `server.py` key details
- Model: `faster-whisper`, `WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")`, currently `MODEL_SIZE = "small"`.
  - This is confirmed working via **automatic download from Hugging Face** on first run — it appeared stuck/silent for several minutes at one point (see Section 6), but ultimately completed successfully on its own. No manual model download was actually needed in the end.
  - ⚠️ A local-folder fallback (`MODEL_SIZE = "./model"`, for manually downloading model files through a browser if the automatic download ever stalls again) was drafted but **never actually applied to the real file** — the working copy still uses the automatic `"small"` download. If Hugging Face download issues come up again, that fallback approach (manually download `Systran/faster-whisper-small` files from Hugging Face into a local `model/` folder, point `MODEL_SIZE` at that folder path) is the documented next move.
- `vad_filter=True` is on — faster-whisper's built-in voice-activity detection, meant to skip silence and reduce hallucinated text on quiet/ambiguous audio.
- `/transcribe` endpoint: accepts multipart form-data (`file`: the audio blob, `lang`: `en-US` or `nl-NL`, mapped internally to `en`/`nl`), returns `{"text": "..."}`.
- CORS is wide open (`allow_origins=["*"]`) — not strictly needed since everything is same-origin, but harmless.
- No authentication anywhere on the server. Currently fine because Tailscale Serve (not Funnel) restricts access to only my own devices. **Would not be safe to switch to `tailscale funnel` (public internet) without adding real auth first.**

### 3.2 — `static/index.html` key details
- Records audio in **4-second chunks** via `MediaRecorder`, POSTs each chunk to `/transcribe`, displays the returned text as a live caption, then runs it through the local topic-matching + sentence-scoring logic (Section 2, step 2) to show a short answer.
- Topics (keyword sets + notes) are stored in `localStorage` under key `sidenote:topics`. Seeded by default with one example topic (chess rules: castling, en passant, stalemate, promotion).
- Language toggle (English / Nederlands pills in the UI) sets which language is passed to the server for that chunk — manual, not auto-detected, because browser/engine language switching mid-stream isn't reliable.
- Silence/near-empty transcriptions are handled server-side now (via `vad_filter`), not client-side — an earlier client-side volume-threshold approach was removed when the whole pipeline moved server-side.

---

## 4. WHY SOME THINGS ARE DESIGNED THE WAY THEY ARE (don't "fix" these back)

- **Manual EN/NL toggle instead of auto-detect** — deliberate. Browser and model-level language auto-detection for a live, code-switching, fast-speech use case was unreliable in testing; a manual toggle is simpler and actually works.
- **Local heuristic sentence-matching instead of an AI call** — deliberate, and forced by a real constraint: outside Claude's own sandbox, there is no free/keyless way to call an AI model from a client-side page without exposing an API key publicly. The offline heuristic is the permanent solution here, not a placeholder.
- **`localStorage`, not `window.storage`** — deliberate fix, not a downgrade. `window.storage` is a Claude-artifact-only API and silently does nothing outside Claude's preview.
- **A real Python server instead of in-browser Whisper** — deliberate trade-off, chosen explicitly by me after being shown the alternative: better accuracy and speed, at the cost of requiring my laptop to be on and running two terminal commands every time I want to use this on my phone.

---

## 5. STANDING PREFERENCES FOR HOW I LIKE TO WORK (carry these into every session)

- Exact, copy-pasteable PowerShell commands. Explain briefly what a command does or checks before I run it, especially anything destructive or hard to undo.
- One change/one test at a time when debugging — don't stack multiple untested fixes together.
- When something fails, get an actual error message or concrete symptom out of me before proposing a fix — don't guess-and-check repeatedly without new information.
- I will personally test things on my phone and laptop and report back exactly what I see (including pasted terminal output) — use that literally, don't assume standard/expected behavior occurred if I haven't confirmed it.
- Flag real trade-offs honestly (speed vs. accuracy, privacy vs. convenience, etc.) rather than picking silently on my behalf, unless it's a clearly reversible/low-stakes default.

---

## 6. NOTABLE BUGS/DEAD ENDS THIS SESSION (so they aren't re-tried)

- `window.storage` silently failing outside Claude's preview (Section 2, step 3) — real bug, fixed via `localStorage`.
- Anthropic API call (`fetch` to `api.anthropic.com`) failing with `Failed to fetch` once self-hosted — expected, not fixable client-side; replaced with the offline heuristic.
- `uvicorn --host <actual IP>` was accidentally used multiple times instead of `--host 0.0.0.0` — caused silent connection failures each time. If something isn't reachable, **check this first** before anything more complex.
- The model-loading step appeared to hang for 5+ minutes with **zero terminal output** — this turned out to be because `server.py` originally had no `print()` statements around the `WhisperModel(...)` line, so it was silent by design, not stuck. Fixed by adding print statements before/after model load. Confirmed separately (via `requests.get(...)` returning `200`) that this was **not** a network/SSL/proxy blocking issue.
- `tailscale serve status` reported "No serve config" in a second terminal window even while a first terminal was actively showing an "Available within your tailnet" link that worked. Never fully explained, but functionally the link worked once tested directly — worth a closer look if this recurs, but not currently blocking anything.

---

## 7. OUTSTANDING / KNOWN GAPS — TOP PRIORITY FOR NEXT SESSION

1. **⚠️ `localStorage` is per-browser, per-device — topics don't sync between phone and laptop.** This wasn't explicitly surfaced as a limitation during the session itself, but it's a real gap: any topic added while testing on the laptop's browser will **not** appear on the phone, and vice versa, since each browser has its own separate local storage. Given the entire point of this tool is having my notes available live on my phone, **this should probably be fixed next session** — most likely by moving topic storage server-side (e.g., a simple JSON file read/written by `server.py`, with a couple of new endpoints) so both devices see the same list.
2. Accuracy of the `small` faster-whisper model on real fast/Dutch speech has **not yet been properly stress-tested** end-to-end through the new server pipeline — the original motivating complaint (fast speech, Dutch) should be explicitly retested now that the whole pipeline actually works, not just confirmed "it transcribed the word hello."
3. Two terminal commands must be run manually every time (`uvicorn` + `tailscale serve`) — no combined launcher yet. A single `.ps1` or `.bat` script that starts both would remove a recurring manual step.
4. The local sentence-matching heuristic is pattern-matching, not true understanding — expect occasional wrong-sentence matches on oddly-phrased questions. Flag specific bad examples if/when they come up so the scoring can be tuned.
5. `CHUNK_MS` (currently 4000ms) is an adjustable trade-off between responsiveness and transcription accuracy/context — revisit if either direction (snappier vs. more accurate) becomes the priority.

---

## 8. EXACT NEXT STEPS FOR THE NEXT SESSION, IN ORDER

1. **Re-attach the actual project files at the start of the session** (`server.py`, `requirements.txt`, `static/index.html`) — a new chat starts with no file access, only this document's description of them.
2. Decide whether to tackle the **cross-device topic sync gap** (Section 7, item 1) first — this is the most important functional gap given the tool's whole purpose.
3. Actually stress-test transcription accuracy on fast/Dutch speech through the real server pipeline (Section 7, item 2), now that the infrastructure itself is confirmed working.
4. If it comes up again, revisit the unexplained `tailscale serve status` discrepancy (Section 6, last item) — low priority unless it starts actually blocking something.
5. Optional, lower priority: write a single launcher script for both terminal commands (Section 7, item 3).

---

*End of Session 1 addendum.*
