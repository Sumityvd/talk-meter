using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Microsoft.HandsFree.Sensors
{
    public class GazeDataProvider
    {
        public static IGazeDataProvider InitializeGazeDataProvider(Settings settings = null)
        {

            if (settings == null)
            {
                settings = new Settings { Sensor = Sensors.TobiiEyeXSDK, UseFixationStream = false };
            }

            IGazeDataProvider gazeDataProvider;
            switch (settings.Sensor)
            {
                case Sensors.TobiiEyeXSDK:
                    gazeDataProvider = CreateTobiiEyeXSdk(settings.UseFixationStream);
                    break;
                case Sensors.TobiiGazeSDK:
                    gazeDataProvider = CreateTobiiGazeSdk();
                    break;
                case Sensors.Mouse:
                    gazeDataProvider = new MouseSdk();
                    break;
                case Sensors.LogPlayback:
                    gazeDataProvider = new LogPlaybackSdk(new LoggingSettings().GazeDataLogFile);
                    break;
                case Sensors.None:
                    gazeDataProvider = new NullSdk();
                    break;
                case Sensors.Network:
                    gazeDataProvider = new NetworkSdk(settings.NetworkPort);
                    break;
                default:
                    throw new ArgumentException("Unknown sensor type");
            }

            if (!gazeDataProvider.Initialize())
            {
                gazeDataProvider = new NullSdk();
                var ret = gazeDataProvider.Initialize();
                Debug.Assert(ret);               
            }
            return gazeDataProvider;
        }

        // Kept out of line so the Tobii assemblies are only loaded when a Tobii sensor is selected;
        // otherwise JIT-compiling InitializeGazeDataProvider fails when they are not deployed.
        [MethodImpl(MethodImplOptions.NoInlining)]
        static IGazeDataProvider CreateTobiiEyeXSdk(bool useFixationStream)
        {
            return new TobiiEyeXSdk(useFixationStream);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static IGazeDataProvider CreateTobiiGazeSdk()
        {
            return new TobiiGazeSdk();
        }
    }
}
