using System;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace Microsoft.HandsFree.Sensors
{
    /// <summary>
    /// Receives gaze points from an external tracker (e.g. a webcam tracker) over UDP on the
    /// loopback interface. Each datagram is "x,y" in physical screen pixels.
    /// Invoked targets are reported back to port + 1 as "target,left,top,right,bottom".
    /// </summary>
    internal class NetworkSdk : IGazeDataProvider, ITargetFeedback
    {
        readonly int _port;
        UdpClient _client;
        Thread _thread;
        volatile bool _running;

        public NetworkSdk(int port)
        {
            _port = port;
        }

        public event EventHandler<GazeEventArgs> GazeEvent;

        public Sensors Sensor { get { return Sensors.Network; } }

        public bool Initialize()
        {
            try
            {
                _client = new UdpClient(new IPEndPoint(IPAddress.Loopback, _port));
            }
            catch (SocketException)
            {
                return false;
            }

            _running = true;
            _thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "NetworkSdk receiver" };
            _thread.Start();
            return true;
        }

        void ReceiveLoop()
        {
            var remote = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                byte[] datagram;
                try
                {
                    datagram = _client.Receive(ref remote);
                }
                catch (SocketException)
                {
                    continue;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }

                double x, y;
                if (TryParse(Encoding.ASCII.GetString(datagram), out x, out y))
                {
                    GazeEvent?.Invoke(this, new GazeEventArgs(x, y, Environment.TickCount, Fixation.Unknown, false));
                }
            }
        }

        static bool TryParse(string message, out double x, out double y)
        {
            x = y = 0;
            var parts = message.Trim().Split(',');
            return parts.Length >= 2 &&
                double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out x) &&
                double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out y);
        }

        public void ReportInvokedTarget(Rect screenBounds)
        {
            var message = string.Format(CultureInfo.InvariantCulture, "target,{0:F0},{1:F0},{2:F0},{3:F0}",
                screenBounds.Left, screenBounds.Top, screenBounds.Right, screenBounds.Bottom);
            var datagram = Encoding.ASCII.GetBytes(message);
            try
            {
                _client?.Send(datagram, datagram.Length, new IPEndPoint(IPAddress.Loopback, _port + 1));
            }
            catch (SocketException)
            {
                // Feedback is best effort; the tracker may not be listening.
            }
            catch (ObjectDisposedException)
            {
            }
        }

        public Task<bool> CreateProfileAsync()
        {
            return Task.FromResult(true);
        }

        // Calibration is owned by the external tracker.
        public void LaunchRecalibration()
        {
        }

        public void BeginAddCalibrationPoint(int x, int y)
        {
        }

        public void EndAddCalibrationPoint()
        {
        }

        public void Terminate()
        {
            _running = false;
            _client?.Close();
        }
    }
}
