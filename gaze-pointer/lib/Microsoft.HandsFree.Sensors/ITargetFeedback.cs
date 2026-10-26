using System.Windows;

namespace Microsoft.HandsFree.Sensors
{
    /// <summary>
    /// Implemented by gaze providers that can use the bounds of invoked targets to correct
    /// their own calibration drift.
    /// </summary>
    public interface ITargetFeedback
    {
        /// <param name="screenBounds">Bounds of the invoked element, in physical screen pixels.</param>
        void ReportInvokedTarget(Rect screenBounds);
    }
}
