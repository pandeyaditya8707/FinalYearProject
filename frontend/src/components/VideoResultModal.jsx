import React, { useState, useEffect } from 'react';
import { API_ENDPOINTS } from '../config';

function VideoResultModal({ isOpen, onClose, resultData, originalVideoSrc }) {
  const [selectedPlate, setSelectedPlate] = useState(null);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);

  useEffect(() => {
    // Reset state when modal opens with new data
    if (isOpen && resultData) {
      setSelectedPlate(null);
      setCurrentFrameIndex(0);
    }
  }, [isOpen, resultData]);

  if (!isOpen || !resultData) return null;

  const { video_info, unique_plates, sample_frames, results_dir } = resultData;

  const handlePlateClick = (plate) => {
    setSelectedPlate(plate === selectedPlate ? null : plate);
  };

  const getFrameUrl = (frameInfo) => {
    // Construct URL to fetch the frame from the backend
    return `${API_ENDPOINTS.BASE_URL}/api/video_results/${results_dir}/${frameInfo.path}`;
  };

  const handleNextFrame = () => {
    if (sample_frames && currentFrameIndex < sample_frames.length - 1) {
      setCurrentFrameIndex(currentFrameIndex + 1);
    }
  };

  const handlePrevFrame = () => {
    if (sample_frames && currentFrameIndex > 0) {
      setCurrentFrameIndex(currentFrameIndex - 1);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-center items-center bg-black bg-opacity-75 overflow-y-auto">
      <div className="bg-gray-800 text-white rounded-lg w-4/5 max-w-6xl p-6 shadow-lg relative max-h-[90vh] overflow-y-auto">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white focus:outline-none"
        >
          ✖
        </button>

        <h2 className="text-2xl font-semibold text-center mb-4">Video Analysis Results</h2>

        {/* Video Information */}
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2">Video Info</h3>
          <div className="bg-gray-700 p-3 rounded-md">
            <p><strong>Duration:</strong> {video_info?.duration?.toFixed(2)}s</p>
            <p><strong>Processed Frames:</strong> {video_info?.processed_frames} of {video_info?.frame_count}</p>
            <p><strong>FPS:</strong> {video_info?.fps?.toFixed(2)}</p>
            <p><strong>Unique Plates Detected:</strong> {unique_plates?.length || 0}</p>
          </div>
        </div>

        {/* Two-column layout for larger screens */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left column: Sample frames */}
          <div>
            <h3 className="text-lg font-medium mb-2">Sample Frames</h3>
            
            {sample_frames && sample_frames.length > 0 ? (
              <div className="bg-gray-700 p-3 rounded-md">
                <div className="relative">
                  <img 
                    src={getFrameUrl(sample_frames[currentFrameIndex])} 
                    alt={`Frame ${sample_frames[currentFrameIndex].frame_number}`}
                    className="w-full h-auto rounded-md border border-gray-600"
                  />
                  
                  <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 px-2 py-1 rounded-md text-sm">
                    Frame {sample_frames[currentFrameIndex].frame_number} | 
                    Time: {sample_frames[currentFrameIndex].timestamp.toFixed(2)}s
                  </div>
                </div>
                
                {/* Frame navigation controls */}
                <div className="flex justify-center mt-2 space-x-4">
                  <button 
                    onClick={handlePrevFrame}
                    disabled={currentFrameIndex === 0}
                    className="bg-gray-600 px-3 py-1 rounded-md disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <span className="px-3 py-1">
                    {currentFrameIndex + 1} of {sample_frames.length}
                  </span>
                  <button 
                    onClick={handleNextFrame}
                    disabled={currentFrameIndex === sample_frames.length - 1}
                    className="bg-gray-600 px-3 py-1 rounded-md disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            ) : (
              <div className="bg-gray-700 p-3 rounded-md">
                <p>No sample frames available.</p>
              </div>
            )}
          </div>

          {/* Right column: Detected license plates */}
          <div>
            <h3 className="text-lg font-medium mb-2">Detected License Plates</h3>
            
            {unique_plates && unique_plates.length > 0 ? (
              <div className="bg-gray-700 p-3 rounded-md max-h-[400px] overflow-y-auto">
                <ul className="divide-y divide-gray-600">
                  {unique_plates.map((plate, index) => (
                    <li 
                      key={index} 
                      className={`py-2 cursor-pointer hover:bg-gray-600 transition-colors ${
                        selectedPlate === plate ? 'bg-gray-600' : ''
                      }`}
                      onClick={() => handlePlateClick(plate)}
                    >
                      <div className="flex justify-between items-center">
                        <div>
                          <span className="font-mono text-lg">{plate.plate_text}</span>
                          <span className="ml-2 text-sm text-green-400">
                            {Math.round(plate.confidence * 100)}%
                          </span>
                        </div>
                        <div className="text-sm text-gray-400">
                          Seen {plate.count} times
                        </div>
                      </div>
                      
                      {selectedPlate === plate && (
                        <div className="mt-2 text-sm">
                          <p>First seen: {new Date(plate.first_seen * 1000).toISOString().substr(11, 8)}</p>
                          {plate.vehicle_info && (
                            <p>Vehicle type: {plate.vehicle_info}</p>
                          )}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="bg-gray-700 p-3 rounded-md">
                <p>No license plates detected in this video.</p>
              </div>
            )}
          </div>
        </div>

        {/* Original video playback (optional) */}
        {originalVideoSrc && (
          <div className="mt-4">
            <h3 className="text-lg font-medium mb-2">Original Video</h3>
            <div className="bg-gray-700 p-3 rounded-md">
              <video 
                src={originalVideoSrc} 
                controls 
                className="w-full h-auto rounded-md"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoResultModal;
