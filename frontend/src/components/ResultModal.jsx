import React, { useState } from 'react';

function ResultModal({ isOpen, onClose, resultData, originalImage }) {
  const [selectedPlate, setSelectedPlate] = useState(null);
  const [showCarDetails, setShowCarDetails] = useState(false);
  
  // Sample car database - replace with your actual data
  const carDatabase = [
    {
      plateNumber: "MH20EE7602",
      make: "Škoda",
      model: "Octavia",
      year: "2020",
      color: "Gray",
      owner: "Rajesh Kumar",
      registrationStatus: "Valid",
      lastInspection: "2023-09-15"
    },
    
  ];
  
  if (!isOpen || !resultData) return null;
  
  const { plates, image_base64 } = resultData;
  
  const searchVehicle = (plateText) => {
    const foundCar = carDatabase.find(car => car.plateNumber === plateText);
    setSelectedPlate(foundCar || { plateNumber: plateText, error: "No vehicle record found" });
    setShowCarDetails(true);
  };
  
  return (
    <div className="fixed inset-0 z-50 flex justify-center items-center bg-black bg-opacity-75">
      <div className="bg-gray-800 text-white rounded-lg w-4/5 max-w-4xl p-6 shadow-lg relative">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white focus:outline-none"
        >
          ✖
        </button>
        
        {/* Extracted Text Section */}
        {plates && plates.length > 0 && (
          <div className="mb-6">
            <h2 className="text-2xl font-semibold text-center mb-4">Results</h2>
            
            {plates.map((plate, index) => (
              <div key={index} className="mb-4">
                <div className="bg-gray-900 border-2 border-green-500 rounded-lg p-3 flex justify-between items-center">
                  <span className="text-green-400 text-xl font-mono font-bold tracking-wider">
                    {plate.text}
                  </span>
                  <button
                    onClick={() => searchVehicle(plate.text)}
                    className="bg-blue-600 hover:bg-blue-700 text-white py-1 px-4 rounded-lg transition-colors duration-300 flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    Search
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
        
        {/* Images Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Original Image */}
          {originalImage && (
            <div>
              <h3 className="text-lg font-medium mb-2 text-center">Original Image</h3>
              <img
                src={originalImage}
                alt="Original"
                className="w-full h-auto rounded-md border border-gray-700"
              />
            </div>
          )}
          
          {/* Processed Image */}
          {image_base64 && (
            <div>
              <h3 className="text-lg font-medium mb-2 text-center">Processed Image</h3>
              <img
                src={`data:image/jpeg;base64,${image_base64}`}
                alt="Processed"
                className="w-full h-auto rounded-md border border-gray-700"
              />
            </div>
          )}
        </div>
      </div>
      
      {/* Car Details Popup */}
      {showCarDetails && (
        <div className="fixed inset-0 z-60 flex items-center justify-center bg-black bg-opacity-80">
          <div className="bg-gray-900 rounded-lg shadow-xl max-w-md w-full mx-4 border border-gray-700">
            <div className="flex justify-between items-center bg-gray-800 px-4 py-3 rounded-t-lg">
              <h3 className="text-xl font-semibold text-white">Vehicle Details</h3>
              <button 
                onClick={() => setShowCarDetails(false)}
                className="text-gray-300 hover:text-white"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <div className="p-6">
              {selectedPlate && !selectedPlate.error ? (
                <div className="space-y-4">
                  <div className="bg-gray-800 p-3 rounded-lg text-center mb-4">
                    <span className="text-green-400 font-mono text-xl font-bold">
                      {selectedPlate.plateNumber}
                    </span>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                    <div className="text-gray-400">Make:</div>
                    <div className="text-white font-medium text-right">{selectedPlate.make}</div>
                    
                    <div className="text-gray-400">Model:</div>
                    <div className="text-white font-medium text-right">{selectedPlate.model}</div>
                    
                    <div className="text-gray-400">Year:</div>
                    <div className="text-white font-medium text-right">{selectedPlate.year}</div>
                    
                    <div className="text-gray-400">Color:</div>
                    <div className="text-white font-medium text-right">{selectedPlate.color}</div>
                    
                    <div className="text-gray-400">Owner:</div>
                    <div className="text-white font-medium text-right">{selectedPlate.owner}</div>
                    
                    <div className="text-gray-400">Registration:</div>
                    <div className={`font-medium text-right ${selectedPlate.registrationStatus === 'Valid' ? 'text-green-400' : 'text-red-400'}`}>
                      {selectedPlate.registrationStatus}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="bg-gray-800 p-3 rounded-lg text-center mb-6">
                    <span className="text-yellow-400 font-mono text-xl font-bold">
                      {selectedPlate?.plateNumber}
                    </span>
                  </div>
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="mt-4 text-gray-300">{selectedPlate?.error || "No details available"}</p>
                </div>
              )}
            </div>
            
            <div className="bg-gray-800 px-4 py-3 text-right rounded-b-lg">
              <button
                onClick={() => setShowCarDetails(false)}
                className="bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg transition-colors duration-300"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ResultModal;