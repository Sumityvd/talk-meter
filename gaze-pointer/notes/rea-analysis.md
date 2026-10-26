# GazePointer — REA static analysis vs. source

REA 6.7.0, provider `rea-dotnet-static` (static only; no assembly loaded or executed).
Raw evidence: [`rea-evidence/`](rea-evidence/) — `*.members.json` (`inspect-managed-members`) and
`*.native.json` (`inspect-managed-native-boundaries`) for the four `Microsoft.HandsFree.*` DLLs
in `apps/GazePointerTest/GazePointerTest/bin/x64/Debug/`.

## Build used

```powershell
$msb = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
& $msb GazePointer.proj -p:TargetFrameworkVersion=v4.8   # no source edits; 4.6.1 pack is unusable on VS2022
```

Run without an eye tracker: `%APPDATA%\Microsoft\MSREnable\GazePointerTest.config` →
`<AppSettings><Mouse><Sensor Sensor="Mouse" /></Mouse></AppSettings>`.
Without it the default `TobiiEyeXSDK` fails to initialize and silently falls back to `NullSdk` (no pointer).

**Upstream crash (fixed on branch `webcam-gaze`):** the vendored `Tobii.EyeX.Client.Net20.dll` is v1.8.486
but `EyeXFramework.dll` references v1.8.504, so MSBuild (warning MSB3277) copies neither to the app output.
JIT-compiling `GazeDataProvider.InitializeGazeDataProvider` then throws `FileNotFoundException` at window
load for *every* sensor, Mouse included. Fix: construct the Tobii providers in `NoInlining` helper methods.

## Pipeline, as recovered from REA call edges

| Stage | REA evidence (call edges out of the method) | Source |
|---|---|---|
| **Sensor select** | `GazeDataProvider.InitializeGazeDataProvider` → `newobj TobiiEyeXSdk / TobiiGazeSdk / MouseSdk / LogPlaybackSdk / NullSdk`, `IGazeDataProvider::Initialize` | `lib/Microsoft.HandsFree.Sensors/GazeDataProvider.cs` |
| **Mouse emulation** | `MouseSdk.HandleMouseMove` → `new DispatcherTimer` + `ldftn MouseTick`; `MouseTick` → `Mouse::GetPosition`, `Visual::PointToScreen`, `Environment::get_TickCount`, `new GazeEventArgs`, `EventHandler::Invoke` | `MouseSdk.cs` — 15 Hz timer, starts only after the 2nd real mouse move |
| **Marshal to UI thread** | `GazePointer.OnGazeData` → `Dispatcher::BeginInvoke(GazeDataHandler)` | `GazePointer.cs:510` |
| **Per-sample handler** | `GazeDataHandler` → `IdleDetector::Tick`, `VerifyWindowContext`, `LogFilter::Update`, `IFilter::Update`, `UpdateCursorPosition`, `GetHitTargetWithMaxTime`, `HandleSameTarget` / `HandleDifferentTarget`, `InvokeTarget`, `FireEyesOn` | `GazePointer.cs:848-896` |
| **Smoothing** | `InitializeFilter` → `GainFilter / StampeFilter / NullFilter / OneEuroFilter`; `OneEuroFilter.Update` → `Settings::get_Beta/get_Cutoff`, `Alpha`, `LowpassFilter::Update`, `Math::Abs` | default `FilterType.OneEuroFilter` (`Filters/Settings.cs:8`) |
| **Dwell → click** | `InvokeTarget` → `IInvokeProvider::Invoke` (and `IToggleProvider::Toggle`) | `GazePointer.cs:565-610` — WPF **UI Automation peers**, not synthetic mouse input |

Dwell timing comes from `GazeClickParameters` (`MouseDownDelay` / `MouseUpDelay` / `RepeatMouseDownDelay`);
the test app uses 90/800 ms, and 90/2500 ms for elements tagged `DestructiveAction`.

## Findings

1. **`SendMouseInput` is dead code.** REA shows `User32::SendInput` is called only from
   `GazePointer.SendMouseInput`, and `SendMouseInput` has **no callers**. Confirmed in source
   (`GazePointer.cs:497`, never referenced). Clicking is done entirely through UI Automation, so the
   pointer only works on WPF elements inside the attached window — it cannot click other apps.
2. **P/Invoke surface is isolated to `Microsoft.HandsFree.Win32`** (65 imports: `SendInput`,
   `SetCursorPos`, low-level hooks, Magnification API, DWM, raw input…). `GazePointer`, `Filters` and
   `Sensors` declare zero P/Invokes. `SetCursorPos` is declared but has no callers in the analyzed assemblies.
3. **Silent sensor fallback.** Any provider whose `Initialize()` returns false is replaced by `NullSdk`
   with only a `Debug.Assert` — the reason the app shows nothing until the sensor is set to `Mouse`.
4. **Settings key names surface as user strings** (`Sensor`, `Filter`, `MouseEmulation`,
   `TrackActiveWindowOnly`, `MaxTrackCount`, `ShowCursor`, …), which is enough to write the config file
   without reading the serializer.

## Reproduce

```powershell
npx -y rea-agents@6.7.0 inspect-managed-members           <dll> --format json
npx -y rea-agents@6.7.0 inspect-managed-native-boundaries <dll> --format json
```
Then walk `normalized_result.call_edges` (`caller` is `Namespace.Type.Method`).
