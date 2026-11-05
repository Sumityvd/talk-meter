# gaze-collect

A website that collects webcam gaze-calibration sessions from invited friends, for training a gaze model
(non-commercial research). A friend opens the site, enters their invite code and does a 2-minute session; when it
ends, the session uploads by itself. It's a separate site from talk-meter, and unlike talk-meter **it uploads data**,
so it asks for consent first.

## How it works

1. **In the browser:** the positioning window, then the same calibration as `webcam_gaze.py --calibration quick`
   (16 dots, 6 guided head moves, 20 s moving dot). Each frame's face crop, landmarks and dot position go into one
   file of about 35 MB. It waits in the browser (IndexedDB) until the upload succeeds, so closing the tab or losing
   the network doesn't lose a finished session.
2. **On Supabase:** a private bucket and two `gaze_*` tables. The browser's anon key can't read or write any of it.
   Everything goes through the `gaze-upload` edge function, which checks the invite code, consent, the daily cap
   (3 sessions), session size and remaining storage.
3. **On your PC:** `sync/sync_sessions.py` downloads finished sessions, checks them, and converts them with the
   tracker's own `gaze_cnn` code into `D:\gaze-train\friends\<P01>\<session>\` (`patches/`, `session.npz`,
   `meta.json`, `raw/session.bin`), the format `train_gaze.session_items()` reads. It then deletes them from
   Supabase. The raw file is kept, so sessions can be reprocessed if a setting turns out to be wrong.

## Setup

**1. Supabase** (the commands use the Supabase CLI; run them from `gaze-collect/`):

```bash
npx supabase login
npx supabase link --project-ref YOUR-PROJECT-REF
npx supabase db push                                   # adds gaze_* tables and the private bucket; changes nothing else
npx supabase functions deploy gaze-upload --no-verify-jwt
npx supabase secrets set GAZE_ALLOWED_ORIGIN=https://YOUR-SITE-ADDRESS GAZE_MAX_STORED_MB=700
```

`db push` applies every migration in this folder only. If the project has other migrations, or you prefer not to
link it, paste `supabase/migrations/*.sql` into the dashboard's SQL editor instead.

`GAZE_MAX_STORED_MB` protects the Free plan's 1 GB, which this shares with whatever else the project stores. Set it
below what's actually free. When it's reached, new sessions wait in the participant's browser until you sync.

**2. The website:** create a **new, separate** site on your host with:

| Setting | Value |
|---|---|
| Root directory | `gaze-collect` |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment variables | `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` (see `.env.example`) |

Don't add these to the talk-meter site. It needs HTTPS for the camera, which these hosts provide.

**3. Your PC:** set these in your environment, never in a file in this repo. The service key can read everything:

```powershell
$env:GAZE_SUPABASE_URL = "https://YOUR-PROJECT-REF.supabase.co"
$env:GAZE_SUPABASE_SERVICE_KEY = "..."
```

## Running a study

Use the tracker's Python environment (`C:\Users\91790\GazePointer\tools\webcam-gaze\.venv`):

```powershell
python sync\sync_sessions.py --add-participant Rahul     # prints their invite code, once
python sync\sync_sessions.py                             # about daily: download, convert, delete from Supabase
python sync\sync_sessions.py --list                      # who has done what, and how much is held in Supabase
python sync\sync_sessions.py --delete-participant Rahul  # if someone withdraws: removes their data everywhere
```

Names stay on your PC, in `D:\gaze-train\friends\participants.json`. Supabase only knows `P01`, `P02`, and so on.

**Capacity on the Free plan:** 1 GB of storage holds about 28 sessions at a time, and 5 GB of downloads a month covers
about 140 sessions. Sync regularly. Each session takes about 50 MB on your PC, and the script stops when under 2 GB is
free. A Free project pauses after a week without activity and must be restored from the dashboard.

## Privacy

These are face images of real people, so:

- Review the consent text in `src/App.tsx` and make sure you'll keep its promises. Changing what's promised means
  changing `CONSENT_VERSION` in `supabase/functions/gaze-upload/logic.ts`, so people are asked again.
- Keep the service key private, and keep `D:\gaze-train\friends` out of any repository.
- Only invite adults, delete someone's data when they ask, and don't publish the images.

## Known limits

- **Labels are less exact than recordings on your own PC.** Screen size and camera position are self-reported, and an
  unknown camera's field of view uses a default of 65°. The raw files allow reprocessing if a better value is found.
- **Colour may differ:** a browser delivers camera images through a different path than the Python tracker.

Using this data in training (keeping some people out for honest testing) is a separate step.

## Develop

```bash
npm install
npm test             # protocol, positioning and server rules
npm run dev          # http://localhost:5181 (needs .env.local)
                     # http://localhost:5181/?offline=1 runs a whole session with no server
npm run build
```

Convert a session file without Supabase: `python sync\sync_sessions.py --convert session.bin OUTPUT_FOLDER`.

## Licences

MediaPipe Face Landmarker is Apache 2.0. The model trained from this data builds on ETH-XGaze and MPIIFaceGaze
(CC BY-NC-SA 4.0), so it's non-commercial.
