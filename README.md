# talk-meter

Measures how much of each minute you spend **talking, smiling or quiet**, from
your face alone, in the browser.

- **No audio.** Only the camera is requested (`audio: false`).
- **No backend.** Nothing is uploaded; results stay in the tab until you
  download them as CSV.
- **Measurement only.** No fatigue score. It is a tool for collecting labelled
  data and tuning the talking detector.

## How it works

MediaPipe FaceLandmarker runs in the browser on each video frame. Two numbers
are kept per frame:
- **MAR**, the mouth aspect ratio: how open the mouth is relative to its width;
- **smile**, the mean of the mouthSmileLeft/Right blendshapes.

Each second of face frames is then classified:

| State | Rule (starting values, in `src/lib/speech.ts`) |
|---|---|
| Talking | MAR swings ≥ 0.08 within the second **and** crosses its median ≥ 2 times (an open-close cycle), and the mouth is not held wide open (MAR > 0.6 for half the second = yawning) |
| Smiling | mean smile blendshape ≥ 0.5 |
| Quiet | neither |
| No face / too few frames | the second is not measured (needs ≥ 3 face frames) |

Speech opens and closes the mouth at about 2–7 Hz, while a yawn is one slow gape
of about 6 seconds. That is why oscillation counts as talking and a sustained
wide-open mouth does not.

Each minute reports `talk_frac`, `smile_frac` and `quiet_s` over its measured
seconds.

## Using it for a labelled recording

1. Press **Start** and allow the camera.
2. As you go, pick what you are doing: reading, typing, listening, speaking,
   laughing, chewing, yawning or other. You can also press **1–8**.
3. Keep the tab visible, because browsers slow down hidden tabs.
4. Press **Stop**, then download:
   - **minutes CSV:** one row per minute (activity, talk/smile share, quiet
     seconds, face visibility, fps);
   - **seconds CSV:** one row per second, with the label, the state and every
     raw statistic (MAR min/max/median/range, crossings, wide-open share, smile
     mean). This is the one to tune thresholds from offline.

Data lives only in the open tab. Reloading or closing it loses anything not
downloaded, and the page warns you first.

## Develop

```bash
npm install      # .npmrc sets legacy-peer-deps (works around an npm 10.8 bug)
npm run dev      # http://localhost:5180
npm test
npm run build    # typecheck + production build into dist/
```

Requires Node 20+. Camera access needs `localhost` or HTTPS.

## Deploy

`npm run build` produces a static site in `dist/` that uses relative paths, so it
works from any static host.

| Host | Settings |
|---|---|
| Vercel | Framework preset: Vite · Build: `npm run build` · Output: `dist` |
| Netlify | Build: `npm run build` · Publish directory: `dist` |
| Cloudflare Pages | Build: `npm run build` · Output: `dist` |
| GitHub Pages | Build locally or in an Action and publish `dist/` |

All of these serve over HTTPS, which the camera requires.

**Third-party downloads.** The browser fetches two files at startup:
- the MediaPipe runtime, from `cdn.jsdelivr.net`;
- the face model, from `storage.googleapis.com`.

These are downloads into the browser; no camera data is sent to either host. To
avoid them, copy the files into `public/` and change the two URLs in
`src/lib/tracker.ts`.

The WASM URL must match the installed `@mediapipe/tasks-vision` version, which
is pinned exactly in `package.json`.

## Files

- `src/lib/speech.ts` — MAR, per-second classification, per-minute summary (pure)
- `src/lib/recorder.ts` — per-second and per-minute records with activity labels
- `src/lib/tracker.ts` — camera + MediaPipe frame loop
- `src/lib/csv.ts` — CSV export
- `src/App.tsx` — the page
